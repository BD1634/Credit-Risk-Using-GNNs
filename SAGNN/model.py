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

        # GNN layers
        self.gnn_layer_1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256,
                                     self.embedding_num * kwargs['embedd_columns_num'] + 256)
        self.gnn_layer_2 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256,
                                     self.embedding_num * kwargs['embedd_columns_num'] + 256)
        self.alpha_attention = torch.nn.Parameter(torch.randn(1))

        # Classification path layers
        self.fc1 = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'] + 256, 512)
        self.fc2 = nn.Linear(512 + kwargs['values_columns_num'], 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

        # LSTM layer
        intermediate_size = self.embedding_num * kwargs['embedd_columns_num'] + 256
        self.lstm_hidden_size = 512
        self.lstm = nn.LSTM(
            input_size=intermediate_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Decoder layers
        self.decoder_1 = nn.Linear(self.lstm_hidden_size, 1024)
        self.decoder_2 = nn.Linear(1024, kwargs['embedd_columns_num'] + kwargs['values_columns_num'])

        # Dropouts
        self.layer_dropout = nn.Dropout(0.35)

    def forward(self, value_batch, embedd_batch, clusters):
        # Assume value_batch, embedd_batch are [batch, seq_len, feat_dim]
        batch_size, seq_len, _ = value_batch.shape
        intermediates = []
        device = get_device()

        # Get static connection matrix (could vary if needed)
        connections_matrix, degree_matrix = matrix_connection(clusters, device=device)
        connection = torch.matmul(torch.matmul(degree_matrix.float(), connections_matrix.float()), degree_matrix.float())

        for t in range(seq_len):
            v_t = value_batch[:, t, :]  # [batch, value_dim]
            e_t = embedd_batch[:, t, :]  # [batch, cat_dim]

            emb = self.embedding_column(e_t)  # [batch, cat_dim, emb_dim]
            query_layer = self.q(emb)
            key_layer = self.k(emb)
            value_layer = self.v(emb)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.att_dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = self.layer_norm_att(context_layer.flatten(-2))

            value_query_layer = self.q_value(v_t)
            value_key_layer = self.k_value(v_t)
            value_value_layer = self.v_value(v_t)
            value_attention_scores = nn.Softmax(dim=-1)(value_query_layer * value_key_layer)
            value_attention_probs = self.att_dropout(value_attention_scores)
            value_context_layer = (value_attention_probs * value_value_layer)
            value_context_layer = self.value_layer_norm_att(value_context_layer)

            intermediate_vector = torch.cat((context_layer, value_context_layer), 1)
            gnn_out_1 = F.relu(self.gnn_layer_1(torch.matmul(connection.float(), intermediate_vector.float())))
            gnn_out_2 = F.relu(self.gnn_layer_2(torch.matmul(connection.float(), gnn_out_1)))
            final_t = self.alpha_attention * intermediate_vector + (1 - self.alpha_attention) * gnn_out_2
            intermediates.append(final_t.unsqueeze(1))

        lstm_input_seq = torch.cat(intermediates, dim=1)  # [batch, seq_len, feat_dim]
        lstm_out, _ = self.lstm(lstm_input_seq)
        lstm_final = lstm_out[:, -1, :]  # or use mean over time

        # CLASSIFICATION PATH
        fc1_out = F.relu(self.fc1(lstm_final))
        fc1_out = self.layer_dropout(fc1_out)
        fc2_in = torch.cat((fc1_out, value_batch[:, -1, :]), 1)  # last time step
        fc2_out = F.relu(self.fc2(fc2_in))
        fc2_out = self.layer_dropout(fc2_out)
        fc3_out = F.relu(self.fc3(fc2_out))
        fc3_out = self.layer_dropout(fc3_out)
        classification_output = F.relu(self.fc4(fc3_out))

        # DECODER PATH
        decoder_hidden = F.relu(self.decoder_1(lstm_final))
        decoder_hidden = self.layer_dropout(decoder_hidden)
        reconstruction_output = F.relu(self.decoder_2(decoder_hidden))

        return classification_output, reconstruction_output, fc2_out
