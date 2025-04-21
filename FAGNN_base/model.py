# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import opt, get_device

class CLASS_NN_Embed_cluster(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_num = opt.EmbeddingSize
        self.embedding_column = nn.Embedding(kwargs['bag_size'], self.embedding_num)

        self.q = nn.Linear(self.embedding_num, self.embedding_num)
        self.k = nn.Linear(self.embedding_num, self.embedding_num)
        self.v = nn.Linear(self.embedding_num, self.embedding_num)
        self.att_dropout = nn.Dropout(0.35)
        self.layer_norm_att = nn.LayerNorm(self.embedding_num * kwargs['embedd_columns_num'], eps=1e-6)

        self.q_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.k_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.v_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.value_layer_norm_att = nn.LayerNorm(256, eps=1e-6)

        combined_input_dim = self.embedding_num * kwargs['embedd_columns_num'] + 256
        self.gnn_layer_1 = nn.Linear(combined_input_dim, combined_input_dim)
        self.gnn_layer_2 = nn.Linear(combined_input_dim, combined_input_dim)
        self.alpha_attention = torch.nn.Parameter(torch.randn(1))

        self.lstm_hidden_size = 512
        self.lstm = nn.LSTM(
            input_size=combined_input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc1 = nn.Linear(self.lstm_hidden_size, 512)
        self.fc2 = nn.Linear(512 + kwargs['values_columns_num'], 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

        self.decoder_1 = nn.Linear(self.lstm_hidden_size, 1024)
        self.decoder_2 = nn.Linear(1024, kwargs['embedd_columns_num'] + kwargs['values_columns_num'])

        self.layer_dropout = nn.Dropout(0.35)

    def dynamic_matrix_connection(self, embeddings, threshold=0.85):
        normed = F.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(normed, normed.T)
        adjacency = (sim_matrix >= threshold).float()
        degree = torch.diag(torch.sum(adjacency, dim=1))
        degree_inv_sqrt = torch.linalg.inv(torch.sqrt(degree + 1e-6 * torch.eye(degree.shape[0], device=degree.device)))
        return adjacency, degree_inv_sqrt

    def forward(self, value_batch, embedd_batch, clusters=None):
        batch_size, seq_len, _ = value_batch.shape

        embedd_batch = self.embedding_column(embedd_batch)
        query = self.q(embedd_batch)
        key = self.k(embedd_batch)
        value = self.v(embedd_batch)
        attn_scores = torch.matmul(query, key.transpose(-1, -2))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.att_dropout(attn_probs)
        cat_context = torch.matmul(attn_probs, value).flatten(2)
        cat_context = self.layer_norm_att(cat_context)

        value_q = self.q_value(value_batch)
        value_k = self.k_value(value_batch)
        value_v = self.v_value(value_batch)
        value_attn = F.softmax(value_q * value_k, dim=-1)
        value_attn = self.att_dropout(value_attn)
        cont_context = value_attn * value_v
        cont_context = self.value_layer_norm_att(cont_context)

        combined = torch.cat((cat_context, cont_context), dim=-1)

        combined_flat = combined.view(-1, combined.shape[-1])

        with torch.no_grad():
            node_repr = torch.mean(combined, dim=1)  # (B, D)
            conn_matrix, degree_matrix = self.dynamic_matrix_connection(node_repr)

        conn = torch.matmul(degree_matrix, conn_matrix)
        gnn_out_1 = F.relu(self.gnn_layer_1(torch.matmul(conn, combined_flat)))
        gnn_out_2 = F.relu(self.gnn_layer_2(torch.matmul(conn, gnn_out_1)))
        gnn_out_2 = gnn_out_2.view(batch_size, seq_len, -1)

        final_intermediate = self.alpha_attention * combined + (1 - self.alpha_attention) * gnn_out_2

        lstm_out, _ = self.lstm(final_intermediate)
        lstm_last = lstm_out[:, -1, :]

        decoder_hidden = F.relu(self.decoder_1(lstm_last))
        decoder_hidden = self.layer_dropout(decoder_hidden)
        reconstruction_output = F.relu(self.decoder_2(decoder_hidden))

        fc1_out = F.relu(self.fc1(lstm_last))
        fc1_out = self.layer_dropout(fc1_out)
        value_agg = torch.mean(value_batch, dim=1)
        fc2_in = torch.cat((fc1_out, value_agg), dim=1)
        fc2_out = F.relu(self.fc2(fc2_in))
        fc2_out = self.layer_dropout(fc2_out)
        fc3_out = F.relu(self.fc3(fc2_out))
        fc3_out = self.layer_dropout(fc3_out)
        classification_output = F.relu(self.fc4(fc3_out))

        return classification_output, reconstruction_output, fc2_out
    
    
    