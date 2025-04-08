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
        self.q = nn.Linear(self.embedding_num, self.embedding_num)
        self.k = nn.Linear(self.embedding_num, self.embedding_num)
        self.v = nn.Linear(self.embedding_num, self.embedding_num)
        self.att_dropout = nn.Dropout(0.35)
        self.layer_norm_att = nn.LayerNorm(self.embedding_num * kwargs['embedd_columns_num'], eps=1e-6)

        self.flatten_nn = nn.Linear(self.embedding_num * kwargs['embedd_columns_num'], 512)
        self.dropout_att = nn.Dropout(0.35)

        self.q_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.k_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.v_value = nn.Linear(kwargs['values_columns_num'], 256)
        self.value_layer_norm_att = nn.LayerNorm(256, eps=1e-6)

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
        connections_matrix, degree_matrix = matrix_connection(clusters, device=get_device())
        embedd_batch = self.embedding_column(embedd_batch)
        query_layer = self.q(embedd_batch)
        key_layer = self.k(embedd_batch)
        value_layer = self.v(embedd_batch)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.att_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.layer_norm_att(context_layer.flatten(-2))

        value_query_layer = self.q_value(value_batch)
        value_key_layer = self.k_value(value_batch)
        value_value_layer = self.v_value(value_batch)
        value_attention_scores = nn.Softmax(dim=-1)(value_query_layer * value_key_layer)
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