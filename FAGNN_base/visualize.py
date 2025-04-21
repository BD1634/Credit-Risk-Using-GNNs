#visualize.py
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

# Set backend for headless environments
import matplotlib
matplotlib.use('Agg')  

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_saved_data(epoch, folder="node_embeddings"):
    embeddings = torch.load(os.path.join(folder, f"lstm_epoch_{epoch}.pt"), map_location="cpu")
    sk_ids = np.load(os.path.join(folder, f"sk_ids_epoch_{epoch}.npy"))
    clusters = np.load(os.path.join(folder, f"clusters_epoch_{epoch}.npy"))
    return embeddings, sk_ids, clusters

def plot_tsne(embeddings, sk_ids, clusters, epoch, save_dir="graphs", sample_size=None):
    ensure_dir(save_dir)
    
    # Optional sampling for large datasets
    if sample_size and len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        sample_clusters = clusters[indices]
    else:
        sample_embeddings = embeddings
        sample_clusters = clusters
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(sample_embeddings)-1), 
                random_state=42, n_jobs=-1)  # Use all cores
    reduced = tsne.fit_transform(sample_embeddings.detach().numpy())

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=sample_clusters, cmap='tab10', alpha=0.7)
    plt.title(f"t-SNE Visualization - Epoch {epoch}" + 
              (f" (Sampled {sample_size} nodes)" if sample_size else ""))
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"tsne_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] t-SNE saved to: {save_path}")

def plot_graph_dynamic_pyvis(embeddings, sk_ids, epoch, threshold=0.85, 
                            save_dir="graphs", max_nodes=500, max_edges=1000):
    ensure_dir(save_dir)
    
    # Sample nodes if too many
    if len(sk_ids) > max_nodes:
        indices = np.random.choice(len(sk_ids), max_nodes, replace=False)
        embeddings = embeddings[indices]
        sk_ids = sk_ids[indices]
    
    # Compute similarities more efficiently
    normed_embeddings = F.normalize(embeddings, dim=1).cpu().numpy()
    cosine_sim = cosine_similarity(normed_embeddings)
    
    # Create network
    net = Network(height='800px', width='100%', notebook=False, cdn_resources='in_line')
    net.force_atlas_2based(gravity=-50, spring_length=100)
    
    # Add nodes - more efficient approach
    for i, sk in enumerate(sk_ids):
        net.add_node(int(sk), label=str(sk), size=10)
    
    # Add edges - vectorized approach with edge limit
    edge_count = 0
    edge_indices = np.where(cosine_sim >= threshold)
    # Use upper triangle to avoid duplicates
    edge_pairs = [(i, j) for i, j in zip(edge_indices[0], edge_indices[1]) if i < j]
    
    # Sort by similarity strength and take top max_edges
    edge_pairs = sorted(edge_pairs, key=lambda pair: cosine_sim[pair[0]][pair[1]], reverse=True)
    edge_pairs = edge_pairs[:max_edges]
    
    for i, j in edge_pairs:
        net.add_edge(int(sk_ids[i]), int(sk_ids[j]), value=float(cosine_sim[i][j]))
        edge_count += 1
    
    # Use physics simulation settings for better performance
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "maxVelocity": 50,
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100
        },
        "timestep": 0.5,
        "adaptiveTimestep": true
      }
    }
    """)
    
    save_path = os.path.join(save_dir, f"graph_epoch_{epoch}.html")
    net.save_graph(save_path)
    print(f"[✓] Interactive graph saved to: {save_path} with {len(sk_ids)} nodes and {edge_count} edges")

def visualize_epoch(epoch, node_folder="node_embeddings", save_dir="graphs", 
                   threshold=0.85, max_nodes=500, max_edges=1000, sample_size=None):
    embeddings, sk_ids, clusters = load_saved_data(epoch, folder=node_folder)
    
    print(f"Dataset size: {len(sk_ids)} nodes")
    
    # t-SNE visualization
    plot_tsne(embeddings, sk_ids, clusters, epoch, save_dir, sample_size)
    
    # Graph visualization
    plot_graph_dynamic_pyvis(embeddings, sk_ids, epoch, threshold=threshold, 
                           save_dir=save_dir, max_nodes=max_nodes, max_edges=max_edges)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True, help="Epoch number to visualize")
    parser.add_argument('--save_dir', type=str, default='graphs')
    parser.add_argument('--node_folder', type=str, default='node_embeddings')
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--max_nodes', type=int, default=500, help="Max nodes to visualize")
    parser.add_argument('--max_edges', type=int, default=1000, help="Max edges to visualize")
    parser.add_argument('--sample_size', type=int, default=None, help="Sample size for t-SNE")

    args = parser.parse_args()

    visualize_epoch(
        epoch=args.epoch,
        node_folder=args.node_folder,
        save_dir=args.save_dir,
        threshold=args.threshold,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        sample_size=args.sample_size
    )