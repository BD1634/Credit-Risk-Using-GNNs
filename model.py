# ---- model.py ----
# import torch
# import torch.nn as nn
# from config import opt
# from config import get_device
# import torch.nn.functional as F
# from utils import matrix_connection

# class CLASS_NN_Embed_cluster(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.embedding_num = opt.EmbeddingSize
#         self.embedding_column = nn.Embedding(kwargs['bag_size'], self.embedding_num)
#         self.q = nn.Linear(self.embedding_num, self.embedding_num)
#         self.k = nn.Linear(self.embedding_num, self.embedding_num)
#         self.v = nn.Linear(self.embedding_num, self.embedding_num)
#         self.att_dropout = nn.Dropout(0.35)
#         self.layer_norm_att = nn.LayerNorm(self.embedding_num * kwargs['embedd_columns_num'], eps=1e-6)

#         self.flatten_nn = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'], 512)
#         self.dropout_att = nn.Dropout(0.35)

#         self.q_value = nn.Linear(kwargs['values_columns_num'], 256)
#         self.k_value = nn.Linear(kwargs['values_columns_num'], 256)
#         self.v_value = nn.Linear(kwargs['values_columns_num'], 256)
#         self.value_layer_norm_att = nn.LayerNorm(256, eps=1e-6)

#         self.layer_concat_1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256, 512)
#         self.layer_concat_2 = nn.Linear(512 + kwargs['values_columns_num'], 128)
#         self.layer_concat_3 = nn.Linear(128, 32)
#         self.gnn_concat = nn.Linear(64, 32)
#         self.layer_concat_4 = nn.Linear(32, 2)

#         self.decoder_1 = nn.Linear(kwargs['embedd_columns_num'] * self.embedding_num + 256, 1024)
#         self.decoder_2 = nn.Linear(1024, kwargs['embedd_columns_num'] + kwargs['values_columns_num'])

#         self.gnn_layer_1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256,
#                                      self.embedding_num * kwargs['embedd_columns_num'] + 256)
#         self.gnn_layer_2 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256,
#                                      self.embedding_num * kwargs['embedd_columns_num'] + 256)

#         self.alpha_attention = torch.nn.Parameter(torch.randn(1))
#         self.layer_dropout = nn.Dropout(0.35)

#     def forward(self, value_batch, embedd_batch, clusters):
#         connections_matrix, degree_matrix = matrix_connection(clusters, device=get_device())
#         embedd_batch = self.embedding_column(embedd_batch)
#         query_layer = self.q(embedd_batch)
#         key_layer = self.k(embedd_batch)
#         value_layer = self.v(embedd_batch)
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         attention_probs = self.att_dropout(attention_probs)
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = self.layer_norm_att(context_layer.flatten(-2))

#         value_query_layer = self.q_value(value_batch)
#         value_key_layer = self.k_value(value_batch)
#         value_value_layer = self.v_value(value_batch)
#         value_attention_scores = nn.Softmax(dim=-1)(value_query_layer * value_key_layer)
#         value_attention_probs = self.att_dropout(value_attention_scores)
#         value_context_layer = (value_attention_probs * value_value_layer)
#         value_context_layer = self.value_layer_norm_att(value_context_layer)

#         self.output = torch.cat((context_layer, value_context_layer), 1)
#         connection = torch.matmul(torch.matmul(degree_matrix.float(), connections_matrix.float()), degree_matrix.float())
#         self.gnn_1 = F.relu(self.gnn_layer_1(torch.matmul(connection.float(), self.output.float())))
#         self.gnn = F.relu(self.gnn_layer_2(torch.matmul(connection.float(), self.gnn_1)))

#         self.output_0 = F.relu(self.layer_concat_1(self.alpha_attention * self.output + (1 - self.alpha_attention) * self.gnn))
#         self.output_0_df = self.layer_dropout(self.output_0)

#         self.output_1 = torch.cat((self.output_0_df, value_batch), 1)
#         self.output_2 = self.layer_dropout(F.relu(self.layer_concat_2(self.output_1)))
#         self.output_3 = self.layer_dropout(F.relu(self.layer_concat_3(self.output_2)))

#         self.output_4 = F.relu(self.layer_concat_4(self.output_3))

#         self.decoder_val_1 = self.layer_dropout(F.relu(self.decoder_1(self.output)))
#         self.decoder_val_2 = F.relu(self.decoder_2(self.decoder_val_1))

#         return self.output_4, self.decoder_val_2, self.output_2


import torch
import torch.nn as nn
import torch.nn.functional as F
from config import opt, get_device
from utils import matrix_connection

class CLASS_NN_Embed_cluster(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_num = opt.EmbeddingSize
        # Embedding layer for categorical features
        self.embedding_column = nn.Embedding(kwargs['bag_size'], self.embedding_num)
        
        #-----------------------------------------------------------------#
        # NEW: LSTM layer to process the embedded features as a sequence
        self.lstm_layer = nn.LSTM(input_size=self.embedding_num, hidden_size=self.embedding_num,
                                  num_layers=1, batch_first=True)
        #-----------------------------------------------------------------#
        
        # Attention layers for LSTM-refined embeddings
        self.q = nn.Linear(self.embedding_num, self.embedding_num)
        self.k = nn.Linear(self.embedding_num, self.embedding_num)
        self.v = nn.Linear(self.embedding_num, self.embedding_num)
        self.att_dropout = nn.Dropout(0.35)
        self.layer_norm_att = nn.LayerNorm(self.embedding_num * kwargs['embedd_columns_num'], eps=1e-6)

        self.flatten_nn = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'], 512)
        self.dropout_att = nn.Dropout(0.35)

        # Layers for value (numerical) features processing
        self.q_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.k_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.v_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.value_layer_norm_att = nn.LayerNorm(256, eps=1e-6)

        # Fusion and classification layers
        self.layer_concat_1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256, 512)
        self.layer_concat_2 = nn.Linear(512 + kwargs['values_columns_num'], 128)
        self.layer_concat_3 = nn.Linear(128, 32)
        self.gnn_concat = nn.Linear(64, 32)
        self.layer_concat_4 = nn.Linear(32, 2)

        self.decoder_1 = nn.Linear(kwargs['embedd_columns_num'] * self.embedding_num + 256, 1024)
        self.decoder_2 = nn.Linear(1024, kwargs['embedd_columns_num'] + kwargs['values_columns_num'])

        self.gnn_layer_1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256,
                                     self.embedding_num * kwargs['embedd_columns_num'] + 256)
        self.gnn_layer_2 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256,
                                     self.embedding_num * kwargs['embedd_columns_num'] + 256)

        self.alpha_attention = torch.nn.Parameter(torch.randn(1))
        self.layer_dropout = nn.Dropout(0.35)

    def forward(self, value_batch, embedd_batch, clusters):
        # Compute the graph connection matrices based on cluster assignments
        connections_matrix, degree_matrix = matrix_connection(clusters, device=get_device())
        # Lookup embeddings for categorical features
        embedded_features = self.embedding_column(embedd_batch)
        # Pass the embedded features through the LSTM to capture sequential dependencies
        lstm_out, _ = self.lstm_layer(embedded_features)
        # Use LSTM output as input for self-attention computation
        query_layer = self.q(lstm_out)
        key_layer = self.k(lstm_out)
        value_layer = self.v(lstm_out)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.att_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.layer_norm_att(context_layer.flatten(-2))

        # Process numerical (value) features via separate attention mechanism
        value_query_layer = self.q_value(value_batch)
        value_key_layer = self.k_value(value_batch)
        value_value_layer = self.v_value(value_batch)
        value_attention_scores = F.softmax(value_query_layer * value_key_layer, dim=-1)
        value_attention_probs = self.att_dropout(value_attention_scores)
        value_context_layer = (value_attention_probs * value_value_layer)
        value_context_layer = self.value_layer_norm_att(value_context_layer)

        self.output = torch.cat((context_layer, value_context_layer), 1)
        connection = torch.matmul(torch.matmul(degree_matrix.float(), connections_matrix.float()), degree_matrix.float())
        self.gnn_1 = F.relu(self.gnn_layer_1(torch.matmul(connection.float(), self.output.float())))
        self.gnn = F.relu(self.gnn_layer_2(torch.matmul(connection.float(), self.gnn_1)))

        self.output_0 = F.relu(self.layer_concat_1(self.alpha_attention * self.output + (1 - self.alpha_attention) * self.gnn))
        self.output_0_df = self.layer_dropout(self.output_0)

        self.output_1 = torch.cat((self.output_0_df, value_batch), 1)
        self.output_2 = self.layer_dropout(F.relu(self.layer_concat_2(self.output_1)))
        self.output_3 = self.layer_dropout(F.relu(self.layer_concat_3(self.output_2)))

        self.output_4 = F.relu(self.layer_concat_4(self.output_3))

        self.decoder_val_1 = self.layer_dropout(F.relu(self.decoder_1(self.output)))
        self.decoder_val_2 = F.relu(self.decoder_2(self.decoder_val_1))

        return self.output_4, self.decoder_val_2, self.output_2
    
    
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # -----------------------------
# # Flash Attention Layer Module
# # -----------------------------
# class FlashAttentionLayer(nn.Module):
#     """
#     This layer wraps the scaled dot-product attention function to use Flash Attention.
#     It assumes that queries, keys, and values all have the same embedding dimension.
#     """
#     def __init__(self, embed_dim, dropout=0.1):
#         super(FlashAttentionLayer, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         # Scaling factor as in the scaled dot-product attention (divide by sqrt(embed_dim))
#         self.scale_factor = embed_dim ** 0.5

#     def forward(self, Q, K, V):
#         # Using PyTorch's native scaled_dot_product_attention with Flash Attention enabled.
#         # Make sure you run on CUDA; otherwise, the flash-specific kernel will not be used.
#         with torch.backends.cuda.sdp_kernel(
#             enable_flash=True, enable_math=False, enable_mem_efficient=False
#         ):
#             # dropout_p is passed here; adjust if needed (here set to 0.1, same as dropout)
#             attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout.p, is_causal=False)
#         # Optionally apply dropout to the output (already applied inside scaled_dot_product_attention if dropout is nonzero)
#         attn_output = self.dropout(attn_output)
#         return attn_output

# # --------------------------------------------------
# # Model: LSTM + Flash Attention + Fully Connected Head
# # --------------------------------------------------
# class LSTMFlashAttentionModel(nn.Module):
#     def __init__(
#         self,
#         input_dim,        # Dimension of input features per time step
#         lstm_hidden,      # Hidden dimension of the LSTM
#         lstm_layers,      # Number of LSTM layers
#         attn_embed_dim,   # Dimension for query/key/value in attention
#         num_classes,      # Number of output classes for final prediction
#         dropout=0.1
#     ):
#         super(LSTMFlashAttentionModel, self).__init__()
#         # Define an LSTM layer (bidirectional to capture context from both sides)
#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=lstm_hidden,
#             num_layers=lstm_layers,
#             batch_first=True,
#             bidirectional=True
#         )
#         # Since LSTM is bidirectional, its output dimension is 2 * lstm_hidden.
#         lstm_output_dim = 2 * lstm_hidden

#         # Linear projections to create Q, K, V from LSTM outputs
#         self.linear_q = nn.Linear(lstm_output_dim, attn_embed_dim)
#         self.linear_k = nn.Linear(lstm_output_dim, attn_embed_dim)
#         self.linear_v = nn.Linear(lstm_output_dim, attn_embed_dim)

#         # Flash Attention layer
#         self.flash_attn = FlashAttentionLayer(attn_embed_dim, dropout=dropout)

#         # Final fully connected layer for classification
#         self.fc = nn.Linear(attn_embed_dim, num_classes)

#     def forward(self, x):
#         # x: [batch_size, seq_len, input_dim]
#         # Process the sequence with LSTM.
#         # lstm_out shape: [batch_size, seq_len, 2 * lstm_hidden]
#         lstm_out, _ = self.lstm(x)
#         # Create queries, keys, values from LSTM outputs.
#         # All will have shape: [batch_size, seq_len, attn_embed_dim]
#         Q = self.linear_q(lstm_out)
#         K = self.linear_k(lstm_out)
#         V = self.linear_v(lstm_out)

#         # Apply Flash Attention over the entire sequence.
#         # This computes the scaled dot-product attention using the efficient Flash Attention kernel.
#         attn_out = self.flash_attn(Q, K, V)  # Shape: [batch_size, seq_len, attn_embed_dim]

#         # Pool the output across the time dimension.
#         # Here we use mean pooling; alternative options (e.g., using the last time step) are also possible.
#         context = attn_out.mean(dim=1)  # Shape: [batch_size, attn_embed_dim]

#         # Compute final logits for classification.
#         logits = self.fc(context)  # Shape: [batch_size, num_classes]
#         return logits

# # -----------------------------
# # Training and Testing Procedure
# # -----------------------------
# def main():
#     # Use CUDA if available.
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Hyperparameters
#     batch_size = 32
#     seq_len = 50           # Length of each input sequence
#     input_dim = 16         # Dimensionality of input features per time step
#     lstm_hidden = 32       # Hidden dimension for LSTM
#     lstm_layers = 1        # Number of LSTM layers
#     attn_embed_dim = 64    # Dimension for projected Q, K, V
#     num_classes = 10       # Number of output classes
#     dropout = 0.1
#     learning_rate = 0.001
#     num_epochs = 10

#     # Instantiate the model and move it to the appropriate device.
#     model = LSTMFlashAttentionModel(
#         input_dim=input_dim,
#         lstm_hidden=lstm_hidden,
#         lstm_layers=lstm_layers,
#         attn_embed_dim=attn_embed_dim,
#         num_classes=num_classes,
#         dropout=dropout
#     ).to(device)

#     # Define an optimizer and a loss function.
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss()

#     # Dummy training loop using random data
#     for epoch in range(num_epochs):
#         model.train()
#         # Create random input data and random integer labels.
#         # Input: [batch_size, seq_len, input_dim]
#         x = torch.randn(batch_size, seq_len, input_dim).to(device)
#         # Labels: [batch_size] with integers in the range [0, num_classes-1]
#         labels = torch.randint(0, num_classes, (batch_size,)).to(device)

#         # Forward pass
#         logits = model(x)  # [batch_size, num_classes]
#         loss = criterion(logits, labels)

#         # Backward pass and optimization step.
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

#     # Switch to evaluation mode and do a forward pass on new dummy input.
#     model.eval()
#     with torch.no_grad():
#         test_x = torch.randn(batch_size, seq_len, input_dim).to(device)
#         test_logits = model(test_x)
#         print(f"Test logits shape: {test_logits.shape}")

# if __name__ == "__main__":
#     main()