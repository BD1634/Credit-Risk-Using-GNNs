# ---- model.py ----
import torch
import torch.nn as nn
from config import opt
from config import get_device
import torch.nn.functional as F
from utils import matrix_connection

class CLASS_NN_Embed_cluster(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_num = opt.EmbeddingSize
        self.embedding_column = nn.Embedding(kwargs['bag_size'], self.embedding_num)

        # Self-attention for categorical attributes
        self.q = nn.Linear(self.embedding_num, self.embedding_num)
        self.k = nn.Linear(self.embedding_num, self.embedding_num)
        self.v = nn.Linear(self.embedding_num, self.embedding_num)
        self.att_dropout = nn.Dropout(0.35)
        self.layer_norm_att = nn.LayerNorm(self.embedding_num * kwargs['embedd_columns_num'], eps=1e-6)

        # Self-attention for continuous attributes
        self.q_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.k_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.v_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.value_layer_norm_att = nn.LayerNorm(256, eps=1e-6)

        combined_input_dim = self.embedding_num * kwargs['embedd_columns_num'] + 256

        # GNN layers
        self.gnn_layer_1 = nn.Linear(combined_input_dim, combined_input_dim)
        self.gnn_layer_2 = nn.Linear(combined_input_dim, combined_input_dim)
        self.alpha_attention = torch.nn.Parameter(torch.randn(1))

        # LSTM for temporal modeling
        self.lstm_hidden_size = 512
        self.lstm = nn.LSTM(
            input_size=combined_input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Classification path
        self.fc1 = nn.Linear(self.lstm_hidden_size, 512)
        self.fc2 = nn.Linear(512 + kwargs['values_columns_num'], 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

        # Decoder path
        self.decoder_1 = nn.Linear(self.lstm_hidden_size, 1024)
        self.decoder_2 = nn.Linear(1024, kwargs['embedd_columns_num'] + kwargs['values_columns_num'])

        self.layer_dropout = nn.Dropout(0.35)

    def forward(self, value_batch, embedd_batch, clusters):
        # value_batch: (batch_size, seq_len, num_cont_features)
        # embedd_batch: (batch_size, seq_len, num_categ_features)
        batch_size, seq_len, _ = value_batch.shape

        # GNN connections (static)
        connections_matrix, degree_matrix = matrix_connection(clusters, device=get_device())

        # Categorical Embedding + Self-Attention
        embedd_batch = self.embedding_column(embedd_batch)  # (B, T, F_cat, emb_dim)
        query = self.q(embedd_batch)
        key = self.k(embedd_batch)
        value = self.v(embedd_batch)
        attn_scores = torch.matmul(query, key.transpose(-1, -2))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.att_dropout(attn_probs)
        cat_context = torch.matmul(attn_probs, value)  # (B, T, F_cat, emb_dim)
        cat_context = cat_context.flatten(2)  # (B, T, F_cat * emb_dim)
        cat_context = self.layer_norm_att(cat_context)

        # Continuous Self-Attention
        value_q = self.q_value(value_batch)
        value_k = self.k_value(value_batch)
        value_v = self.v_value(value_batch)
        value_attn = F.softmax(value_q * value_k, dim=-1)
        value_attn = self.att_dropout(value_attn)
        cont_context = value_attn * value_v  # (B, T, 256)
        cont_context = self.value_layer_norm_att(cont_context)

        # Concatenate both contexts
        combined = torch.cat((cat_context, cont_context), dim=-1)  # (B, T, combined_dim)

        # GNN over combined context (flattened for graph op)
        combined_flat = combined.view(-1, combined.shape[-1])
        conn = torch.matmul(degree_matrix.float(), connections_matrix.float())
        gnn_out_1 = F.relu(self.gnn_layer_1(torch.matmul(conn, combined_flat)))
        gnn_out_2 = F.relu(self.gnn_layer_2(torch.matmul(conn, gnn_out_1)))
        gnn_out_2 = gnn_out_2.view(batch_size, seq_len, -1)

        # Attention blend
        final_intermediate = self.alpha_attention * combined + (1 - self.alpha_attention) * gnn_out_2  # (B, T, D)

        # LSTM
        lstm_out, _ = self.lstm(final_intermediate)  # (B, T, hidden)
        lstm_last = lstm_out[:, -1, :]  # (B, hidden)

        # Decoder
        decoder_hidden = F.relu(self.decoder_1(lstm_last))
        decoder_hidden = self.layer_dropout(decoder_hidden)
        reconstruction_output = F.relu(self.decoder_2(decoder_hidden))

        # Classification
        fc1_out = F.relu(self.fc1(lstm_last))
        fc1_out = self.layer_dropout(fc1_out)
        value_agg = torch.mean(value_batch, dim=1)  # mean pool over time
        fc2_in = torch.cat((fc1_out, value_agg), dim=1)
        fc2_out = F.relu(self.fc2(fc2_in))
        fc2_out = self.layer_dropout(fc2_out)
        fc3_out = F.relu(self.fc3(fc2_out))
        fc3_out = self.layer_dropout(fc3_out)
        classification_output = F.relu(self.fc4(fc3_out))

        return classification_output, reconstruction_output, fc2_out