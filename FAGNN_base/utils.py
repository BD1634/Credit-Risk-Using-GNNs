# ---- utils.py ----
import torch
import numpy as np


def containsAny(seq, aset):
    return any(i in seq for i in aset)


def intermediate_feature_distance(intermediate_features, label_batch):
    positive_vector = torch.mean(intermediate_features * label_batch.unsqueeze(-1).float(), dim=0)
    zero = torch.zeros_like(label_batch)
    label_temp = label_batch + 1
    label_negative = torch.where(label_temp == 2, zero, label_temp)
    negative_vector = torch.mean(intermediate_features * label_negative.unsqueeze(-1).float(), dim=0)
    similarity = abs(torch.cosine_similarity(positive_vector, negative_vector, dim=0))
    return similarity


def matrix_connection(a, device):
    a = a.to('cpu')
    a_array = a.numpy()
    dict_index = {i.numpy().tolist(): sum(np.argwhere(a_array == i.numpy()).tolist(), []) for i in a.unique()}
    matrix_connect = np.zeros((len(a), len(a)))
    degree_matrix = np.zeros((len(a), len(a)))
    for index_column, i in enumerate(a):
        for j in dict_index[i.numpy().tolist()]:
            matrix_connect[index_column][j] = 1
        degree_matrix[index_column][index_column] = len(dict_index[i.numpy().tolist()])
    matrix_connect = torch.tensor(matrix_connect, dtype=torch.float32)
    degree_matrix = torch.tensor(degree_matrix, dtype=torch.float32)
    degree_matrix = torch.inverse(torch.sqrt(degree_matrix))
    return matrix_connect.to(device), degree_matrix.to(device)