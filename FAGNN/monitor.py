# monitor.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def track_node_representations(epoch, intermediate_tensor, sk_ids, output_dir='node_transitions'):
    """
    Saves 2D PCA-projected node embeddings per epoch for tracking transitions.

    Args:
        epoch (int): Current epoch
        intermediate_tensor (Tensor): Shape (batch_size, embedding_dim)
        sk_ids (List[int]): Corresponding SK_ID_CURR values
        output_dir (str): Folder to store PCA plots and tensors
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save raw tensor for later cosine tracking or t-SNE
    np.save(os.path.join(output_dir, f'intermediate_epoch_{epoch}.npy'), intermediate_tensor.detach().cpu().numpy())

    # Perform PCA (2D) for visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(intermediate_tensor.detach().cpu().numpy())

    # Plot with SK_ID_CURR labels
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
    for i, sk_id in enumerate(sk_ids):
        if i % max(1, len(sk_ids) // 100) == 0:
            plt.text(reduced[i, 0], reduced[i, 1], str(sk_id), fontsize=6, alpha=0.6)
    plt.title(f"Node Embeddings PCA - Epoch {epoch}")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'pca_epoch_{epoch}.png'))
    plt.close()