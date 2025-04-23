# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import opt, get_device

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.5, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Each head gets its own weight matrix
        self.W = nn.Parameter(torch.zeros(size=(n_heads, in_features, out_features // n_heads)))
        # Attention parameters
        self.a = nn.Parameter(torch.zeros(size=(n_heads, 2 * (out_features // n_heads))))
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
    
    def forward(self, features, adj_matrix=None):
        """
        Features shape: [batch_size, seq_len, feature_dim]
        adj_matrix: Either None (use self-attention) or [batch_size, batch_size] adjacency matrix
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Process each attention head
        heads = []
        for i in range(self.n_heads):
            # Linear transformation
            Wh = torch.matmul(features, self.W[i])  # Shape: [batch_size, seq_len, out_features//n_heads]
            
            # Calculate attention scores
            # For each node, compute attention with every other node
            q = Wh  # Query: [batch_size, seq_len, out_features//n_heads]
            k = Wh  # Key: [batch_size, seq_len, out_features//n_heads]
            
            # Compute attention matrix: [batch_size, seq_len, seq_len]
            attention = torch.bmm(q, k.transpose(1, 2)) / (self.out_features // self.n_heads) ** 0.5
            
            # Apply adjacency matrix mask if provided
            if adj_matrix is not None:
                # Create a mask for valid connections
                zero_vec = -9e15 * torch.ones_like(attention)
                # Make sure the adjacency matrix is broadcast correctly across batches
                if adj_matrix.dim() == 2:  # If it's [B, B] we need to expand it for each sequence element
                    # Create a new adjacency tensor of the right shape
                    expanded_adj = torch.zeros(batch_size, seq_len, seq_len, device=features.device)
                    for b in range(batch_size):
                        expanded_adj[b, :, :] = 1.0  # Default to fully connected within each batch
                    attention = torch.where(expanded_adj > 0, attention, zero_vec)
                else:
                    attention = torch.where(adj_matrix > 0, attention, zero_vec)
            
            # Apply softmax to normalize attention weights
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training)
            
            # Apply attention to get new node features
            h_prime = torch.bmm(attention, Wh)  # [batch_size, seq_len, out_features//n_heads]
            heads.append(h_prime)
        
        # Combine attention heads
        out = torch.cat(heads, dim=2)  # [batch_size, seq_len, out_features]
        return out

class CLASS_NN_Embed_cluster(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_num = opt.EmbeddingSize
        self.embedding_column = nn.Embedding(kwargs['bag_size'], self.embedding_num)

        # Self-attention for embeddings
        self.q = nn.Linear(self.embedding_num, self.embedding_num)
        self.k = nn.Linear(self.embedding_num, self.embedding_num)
        self.v = nn.Linear(self.embedding_num, self.embedding_num)
        self.att_dropout = nn.Dropout(0.6)  # Increased from 0.5
        self.layer_norm_att = nn.LayerNorm(self.embedding_num * kwargs['embedd_columns_num'], eps=1e-6)

        # Self-attention for numerical values
        self.q_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.k_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.v_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.value_layer_norm_att = nn.LayerNorm(256, eps=1e-6)

        # Combined input dimension from both pathways
        combined_input_dim = self.embedding_num * kwargs['embedd_columns_num'] + 256
        
        # GNN layers with multi-head attention
        self.gnn_layer_1 = GraphAttentionLayer(combined_input_dim, combined_input_dim, n_heads=4, dropout=0.6)
        self.layer_norm_gnn_1 = nn.LayerNorm(combined_input_dim, eps=1e-6)
        
        self.gnn_layer_2 = GraphAttentionLayer(combined_input_dim, combined_input_dim, n_heads=4, dropout=0.6)
        self.layer_norm_gnn_2 = nn.LayerNorm(combined_input_dim, eps=1e-6)
        
        # Alpha parameter to balance attention and graph outputs
        self.alpha_attention = torch.nn.Parameter(torch.tensor(0.5))  # Initialize to equal weighting
        
        # LSTM for sequence learning
        self.lstm_hidden_size = 512
        self.lstm = nn.LSTM(
            input_size=combined_input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        # Layer Normalization after LSTM
        self.lstm_norm = nn.LayerNorm(self.lstm_hidden_size, eps=1e-6)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.lstm_hidden_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512 + kwargs['values_columns_num'], 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.bn_fc3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 2)

        # Decoder for reconstruction
        self.decoder_1 = nn.Linear(self.lstm_hidden_size, 1024)
        self.decoder_2 = nn.Linear(1024, kwargs['embedd_columns_num'] + kwargs['values_columns_num'])

        # Increased dropout for regularization
        self.layer_dropout = nn.Dropout(0.6)  # Increased from 0.5

    def create_edge_features(self, node_embeddings):
        """Create edge features based on node similarities"""
        normed = F.normalize(node_embeddings, dim=-1)
        similarity = torch.matmul(normed, normed.transpose(-2, -1))
        
        # Create multiple edge features using different transforms
        edge_features = torch.stack([
            similarity,  # Raw similarity
            torch.exp(-torch.pow(1 - similarity, 2) / 0.1),  # Gaussian kernel
            torch.sigmoid(similarity * 5)  # Smooth threshold function
        ], dim=-1)
        
        return edge_features

    def dynamic_matrix_connection(self, embeddings):
        """Create dynamic adjacency matrix based on node similarity"""
        # Normalize embeddings for cosine similarity
        normed = F.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(normed, normed.T)
        
        # Create edge features for richer graph representation
        edge_features = self.create_edge_features(embeddings)
        
        # Use sigmoid to bound the threshold between 0 and 1
        threshold = torch.sigmoid(self.similarity_threshold)
        adjacency = (sim_matrix >= threshold).float()
        
        # Add self-loops to ensure all nodes are connected
        batch_size = adjacency.shape[0]
        adjacency = adjacency + torch.eye(batch_size, device=adjacency.device)
        
        # Normalize adjacency matrix for message passing
        degree = torch.sum(adjacency, dim=1)
        # Avoid division by zero
        degree = torch.clamp(degree, min=1e-6)
        degree_inv_sqrt = 1.0 / torch.sqrt(degree)
        
        # Create normalized adjacency using the degree matrix
        # This avoids explicit matrix inversion which can be unstable
        norm_adjacency = adjacency.clone()
        for i in range(batch_size):
            norm_adjacency[i] = adjacency[i] * degree_inv_sqrt[i]
        
        norm_adjacency = norm_adjacency * degree_inv_sqrt.unsqueeze(1)
        
        return norm_adjacency, edge_features

    def forward(self, value_batch, embedd_batch, clusters=None):
        batch_size, seq_len, _ = value_batch.shape

        # Embedding pathway
        embedd_batch = self.embedding_column(embedd_batch)
        query = self.q(embedd_batch)
        key = self.k(embedd_batch)
        value = self.v(embedd_batch)
        
        # Self-attention for embeddings
        attn_scores = torch.matmul(query, key.transpose(-1, -2))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.att_dropout(attn_probs)
        cat_context = torch.matmul(attn_probs, value).flatten(2)
        cat_context = self.layer_norm_att(cat_context)

        # Value pathway with self-attention
        value_q = self.q_value(value_batch)
        value_k = self.k_value(value_batch)
        value_v = self.v_value(value_batch)
        value_attn = F.softmax(value_q * value_k, dim=-1)
        value_attn = self.att_dropout(value_attn)
        cont_context = value_attn * value_v
        cont_context = self.value_layer_norm_att(cont_context)

        # Combine both pathways
        combined = torch.cat((cat_context, cont_context), dim=-1)
        
        # GNN processing using multi-head attention instead of explicit adjacency
        # Use self-attention GNN instead of explicit graph construction
        gnn_out_1 = self.gnn_layer_1(combined)  # Use self-attention within batch
        gnn_out_1 = self.layer_norm_gnn_1(gnn_out_1 + combined)  # Skip connection
        gnn_out_1 = F.relu(gnn_out_1)
        
        # Second GNN layer
        gnn_out_2 = self.gnn_layer_2(gnn_out_1)  # Use self-attention
        gnn_out_2 = self.layer_norm_gnn_2(gnn_out_2 + gnn_out_1)  # Skip connection
        gnn_out_2 = F.relu(gnn_out_2)
        
        # Adaptive fusion of attention and graph outputs
        final_intermediate = self.alpha_attention * combined + (1 - self.alpha_attention) * gnn_out_2

        # LSTM for sequential learning
        lstm_out, _ = self.lstm(final_intermediate)
        lstm_last = self.lstm_norm(lstm_out[:, -1, :])

        # Reconstruction pathway
        decoder_hidden = F.relu(self.decoder_1(lstm_last))
        decoder_hidden = self.layer_dropout(decoder_hidden)
        reconstruction_output = F.relu(self.decoder_2(decoder_hidden))

        # Classification pathway
        fc1_out = self.fc1(lstm_last)
        fc1_out = F.relu(self.bn_fc1(fc1_out))
        fc1_out = self.layer_dropout(fc1_out)
        
        # Incorporate raw values as skip connection
        value_agg = torch.mean(value_batch, dim=1)
        fc2_in = torch.cat((fc1_out, value_agg), dim=1)
        fc2_out = self.fc2(fc2_in)
        fc2_out = F.relu(self.bn_fc2(fc2_out))
        fc2_out = self.layer_dropout(fc2_out)
        
        fc3_out = self.fc3(fc2_out)
        fc3_out = F.relu(self.bn_fc3(fc3_out))
        fc3_out = self.layer_dropout(fc3_out)
        
        classification_output = self.fc4(fc3_out)  # No ReLU in final layer

        return classification_output, reconstruction_output, fc2_out