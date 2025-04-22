# monitor.py
import os
import torch
import numpy as np

def save_lstm_outputs(embeddings, sk_ids, clusters, epoch, folder="node_embeddings", labels=None):
    """
    Saves intermediate LSTM embeddings, SK_ID_CURR, cluster IDs, and classification labels.
    """
    os.makedirs(folder, exist_ok=True)
    torch.save(embeddings, os.path.join(folder, f"lstm_epoch_{epoch}.pt"))
    np.save(os.path.join(folder, f"sk_ids_epoch_{epoch}.npy"), np.array(sk_ids))
    np.save(os.path.join(folder, f"clusters_epoch_{epoch}.npy"), np.array(clusters))
    if labels is not None:
        np.save(os.path.join(folder, f"labels_epoch_{epoch}.npy"), np.array(labels))