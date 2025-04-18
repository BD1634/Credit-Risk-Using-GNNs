import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numba import njit
import torch

try:
    import cupy as cnp        
    backend = "cupy"
except ImportError:
    import numpy as cnp         
    backend = "numpy"

import sys
sys.modules["np"] = cnp         
np = cnp

@njit(cache=True)
def containsAny_numba(seq_arr, aset_arr):
    """Numba‑JIT accelerated version that checks membership of any char in set."""
    for c in seq_arr:
        for a in aset_arr:
            if c == a:
                return True
    return False

def containsAny(seq: str, aset):
    # Fast path using Python set intersection, fallback to JIT if needed
    return bool(set(seq) & set(aset))


def intermediate_feature_distance(intermediate_features, label_batch):
    positive_vector = torch.mean(intermediate_features * label_batch.unsqueeze(-1).float(), dim=0)
    zero = torch.zeros_like(label_batch)
    label_temp = label_batch + 1
    label_negative = torch.where(label_temp == 2, zero, label_temp)
    negative_vector = torch.mean(intermediate_features * label_negative.unsqueeze(-1).float(), dim=0)
    similarity = abs(torch.cosine_similarity(positive_vector, negative_vector, dim=0))
    return similarity


def matrix_connection(a, device):
    """Build adjacency & degree matrices on GPU if possible."""
    a = a.to(device)
    a_array = a.cpu().numpy()
    # build dict of cluster id to indices
    dict_index = {}
    for idx, cid in enumerate(a_array):
        dict_index.setdefault(cid, []).append(idx)

    num_nodes = len(a)
    # use cupy if device is cuda
    if device.type == 'cuda':
        import cupy as cp
        matrix_connect = cp.zeros((num_nodes, num_nodes), dtype=cp.float32)
        degree_matrix = cp.zeros((num_nodes, num_nodes), dtype=cp.float32)
    else:
        matrix_connect = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        degree_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for row_idx, cid in enumerate(a_array):
        neighbors = dict_index[cid]
        if device.type == 'cuda':
            matrix_connect[row_idx, neighbors] = 1
        else:
            matrix_connect[row_idx][neighbors] = 1
        degree_matrix[row_idx][row_idx] = len(neighbors)

    # convert to torch tensors
    tc = torch.as_tensor(matrix_connect, device=device)
    td = torch.as_tensor(degree_matrix, device=device)
    td = torch.inverse(torch.sqrt(td))
    return tc, td