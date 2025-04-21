import os
import torch
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity

# Set backend for headless environments (e.g., VSCode terminals, servers)
import matplotlib
matplotlib.use('Agg')  

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_saved_data(epoch, folder="node_embeddings"):
    embeddings = torch.load(os.path.join(folder, f"lstm_epoch_{epoch}.pt"), map_location="cpu")
    sk_ids = np.load(os.path.join(folder, f"sk_ids_epoch_{epoch}.npy"))
    clusters = np.load(os.path.join(folder, f"clusters_epoch_{epoch}.npy"))
    return embeddings, sk_ids, clusters


def plot_tsne(embeddings, sk_ids, clusters, epoch, save_dir="graphs"):
    ensure_dir(save_dir)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(embeddings.detach().numpy())

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.title(f"t-SNE Visualization - Epoch {epoch}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"tsne_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] t-SNE saved to: {save_path}")


def plot_graph_dynamic_pyvis(embeddings, sk_ids, epoch, threshold=0.85, save_dir="graphs"):
    ensure_dir(save_dir)
    normed_embeddings = F.normalize(embeddings, dim=1).cpu().numpy()
    cosine_sim = cosine_similarity(normed_embeddings)

    net = Network(height='800px', width='100%', notebook=False, cdn_resources='in_line')
    net.force_atlas_2based(gravity=-50)

    # Add nodes
    for sk in sk_ids:
        net.add_node(int(sk), label=str(sk), size=10)

    # Add edges based on similarity
    for i in range(len(sk_ids)):
        for j in range(i + 1, len(sk_ids)):
            sim = cosine_sim[i][j]
            if sim >= threshold:
                net.add_edge(int(sk_ids[i]), int(sk_ids[j]), value=sim)

    save_path = os.path.join(save_dir, f"graph_epoch_{epoch}.html")
    net.save_graph(save_path)
    print(f"[✓] Interactive graph saved to: {save_path}")


def visualize_epoch(epoch, node_folder="node_embeddings", save_dir="graphs", threshold=0.85):
    embeddings, sk_ids, clusters = load_saved_data(epoch, folder=node_folder)
    plot_tsne(embeddings, sk_ids, clusters, epoch, save_dir)
    plot_graph_dynamic_pyvis(embeddings, sk_ids, epoch, threshold=threshold, save_dir=save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True, help="Epoch number to visualize")
    parser.add_argument('--save_dir', type=str, default='graphs')
    parser.add_argument('--node_folder', type=str, default='node_embeddings')
    parser.add_argument('--threshold', type=float, default=0.85)

    args = parser.parse_args()

    visualize_epoch(
        epoch=args.epoch,
        node_folder=args.node_folder,
        save_dir=args.save_dir,
        threshold=args.threshold
    )