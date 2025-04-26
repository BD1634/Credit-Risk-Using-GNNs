# visualize.py
import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from pyvis.network import Network
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_interpretability_outputs(embeddings, sk_ids, clusters, epoch, folder="node_embeddings", 
                                  risk_levels=None, shap_values=None, shap_indices=None,
                                  shap_sk_ids=None, shap_clusters=None, shap_risk_levels=None,
                                  feature_names=None):
    """
    Saves embeddings, IDs, clusters, risk levels and SHAP values for interpretability
    
    Args:
        embeddings: Tensor of node embeddings
        sk_ids: List of node IDs
        clusters: List of cluster assignments
        epoch: Current epoch number
        folder: Directory to save outputs
        risk_levels: Array of risk level assignments (0=low, 1=medium, 2=high)
        shap_values: SHAP values for selected nodes
        shap_indices: Indices of nodes selected for SHAP analysis
        shap_sk_ids: IDs of nodes selected for SHAP analysis
        shap_clusters: Cluster assignments of nodes selected for SHAP analysis
        shap_risk_levels: Risk levels of nodes selected for SHAP analysis
        feature_names: Names of input features
    """
    os.makedirs(folder, exist_ok=True)
    
    # Save embeddings and metadata
    torch.save(embeddings, os.path.join(folder, f"lstm_epoch_{epoch}.pt"))
    np.save(os.path.join(folder, f"sk_ids_epoch_{epoch}.npy"), np.array(sk_ids))
    np.save(os.path.join(folder, f"clusters_epoch_{epoch}.npy"), np.array(clusters))
    
    # Save risk levels if provided
    if risk_levels is not None:
        np.save(os.path.join(folder, f"labels_epoch_{epoch}.npy"), np.array(risk_levels))
    
    # Save SHAP values and related data if provided
    if shap_values is not None:
        np.save(os.path.join(folder, f"shap_values_epoch_{epoch}.npy"), shap_values)
        
    if shap_indices is not None:
        np.save(os.path.join(folder, f"shap_indices_epoch_{epoch}.npy"), np.array(shap_indices))
    
    if shap_sk_ids is not None:
        np.save(os.path.join(folder, f"shap_sk_ids_epoch_{epoch}.npy"), np.array(shap_sk_ids))
    
    if shap_clusters is not None:
        np.save(os.path.join(folder, f"shap_clusters_epoch_{epoch}.npy"), np.array(shap_clusters))
    
    if shap_risk_levels is not None:
        np.save(os.path.join(folder, f"shap_risk_levels_epoch_{epoch}.npy"), np.array(shap_risk_levels))
    
    if feature_names is not None:
        np.save(os.path.join(folder, f"feature_names_epoch_{epoch}.npy"), np.array(feature_names))
    
    print(f"[✓] All visualization and interpretability data saved for epoch {epoch}")

def load_saved_data(epoch, folder="node_embeddings"):
    """Load all data saved for visualization and interpretability"""
    embeddings = torch.load(os.path.join(folder, f"lstm_epoch_{epoch}.pt"), map_location="cpu")
    sk_ids = np.load(os.path.join(folder, f"sk_ids_epoch_{epoch}.npy"))
    clusters = np.load(os.path.join(folder, f"clusters_epoch_{epoch}.npy"))
    
    # Try to load risk labels
    labels_path = os.path.join(folder, f"labels_epoch_{epoch}.npy")
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Risk levels loaded with distribution:")
        for level, count in zip(unique, counts):
            print(f"  Level {level}: {count} samples ({count/len(labels)*100:.2f}%)")
    else:
        print("No risk levels file found!")
        labels = None
    
    # Try to load SHAP values and related data
    shap_data = {}
    
    shap_values_path = os.path.join(folder, f"shap_values_epoch_{epoch}.npy")
    if os.path.exists(shap_values_path):
        shap_data["shap_values"] = np.load(shap_values_path)
        print(f"SHAP values loaded with shape: {shap_data['shap_values'].shape}")
        
        # Load other SHAP-related data
        shap_data["shap_indices"] = np.load(os.path.join(folder, f"shap_indices_epoch_{epoch}.npy"))
        shap_data["shap_sk_ids"] = np.load(os.path.join(folder, f"shap_sk_ids_epoch_{epoch}.npy"))
        shap_data["shap_clusters"] = np.load(os.path.join(folder, f"shap_clusters_epoch_{epoch}.npy"))
        shap_data["shap_risk_levels"] = np.load(os.path.join(folder, f"shap_risk_levels_epoch_{epoch}.npy"))
        shap_data["feature_names"] = np.load(os.path.join(folder, f"feature_names_epoch_{epoch}.npy"), allow_pickle=True)
    else:
        print("No SHAP data found!")
        shap_data = None
    
    return embeddings, sk_ids, clusters, labels, shap_data

def plot_tsne(embeddings, sk_ids, clusters, epoch, save_dir="graphs", sample_size=None, labels=None):
    """Create t-SNE visualization of embeddings colored by risk level"""
    ensure_dir(save_dir)
    
    # Optional sampling for large datasets
    if sample_size and len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        sample_clusters = clusters[indices]
        sample_labels = labels[indices] if labels is not None else None
    else:
        sample_embeddings = embeddings
        sample_clusters = clusters
        sample_labels = labels
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(sample_embeddings)-1), 
                random_state=42, n_jobs=-1)  # Use all cores
    reduced = tsne.fit_transform(sample_embeddings.detach().numpy())

    plt.figure(figsize=(10, 8))
    
    # Color based on risk levels if labels are provided
    if sample_labels is not None:
        # Create a custom colormap for risk levels
        colors = []
        for label in sample_labels:
            if label == 0:  # Low risk
                colors.append('green')
            elif label == 1:  # Medium risk
                colors.append('yellow')
            else:  # High risk
                colors.append('red')
        
        plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.7)
        
        # Create legend for risk levels
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Low Risk'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Medium Risk'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='High Risk')
        ]
        plt.legend(handles=legend_elements)
    else:
        # Default coloring by clusters
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

def plot_feature_importance_by_risk(shap_data, epoch, save_dir="graphs", top_n=10):
    """Create visualization of feature importance by risk level"""
    ensure_dir(save_dir)
    
    if shap_data is None:
        print("No SHAP data available for feature importance visualization")
        return
    
    try:
        shap_values = shap_data["shap_values"]
        feature_names = shap_data["feature_names"]
        risk_levels = shap_data["shap_risk_levels"]
        
        # Handle multi-dimensional SHAP values
        if len(shap_values.shape) > 2:
            # If SHAP values have multiple dimensions (e.g., for multi-class models),
            # use the values for class 1 (positive class) or sum across classes
            if shap_values.shape[2] == 2:  # Binary classification, shape is (n_samples, n_features, 2)
                shap_values = shap_values[:, :, 1]  # Use values for positive class
            else:
                shap_values = np.sum(shap_values, axis=2)  # Sum across all classes
        
        # Convert feature_names to list of strings
        if isinstance(feature_names, np.ndarray):
            feature_names = [str(f) for f in feature_names]
        
        risk_names = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_colors = ['green', 'yellow', 'red']
        
        plt.figure(figsize=(15, 12))
        
        # For each risk level
        for risk in range(3):
            indices = np.where(risk_levels == risk)[0]
            if len(indices) == 0:
                continue
                
            # Get average SHAP values for this risk level
            risk_shap = np.abs(shap_values[indices]).mean(axis=0)
            
            # Get top features
            top_indices = np.argsort(-risk_shap)[:top_n]
            # Convert feature names to strings to ensure they're hashable
            top_features = [str(feature_names[i]) if i < len(feature_names) else f"Feature_{i}" 
                           for i in top_indices]
            top_values = [float(risk_shap[i]) for i in top_indices]
            
            # Create subplot
            plt.subplot(3, 1, risk + 1)
            bars = plt.barh(top_features, top_values, color=risk_colors[risk], alpha=0.7)
            plt.title(f'Top Features for {risk_names[risk]}')
            plt.xlabel('Mean |SHAP value|')
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"feature_importance_by_risk_epoch{epoch}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[✓] Feature importance by risk level saved to: {save_path}")
    except Exception as e:
        print(f"Error in plot_feature_importance_by_risk: {e}")
        print("Continuing without feature importance visualization...")

def create_feature_heatmap(shap_data, epoch, save_dir="graphs"):
    """Create heatmap visualization showing feature importance by risk level"""
    ensure_dir(save_dir)
    
    if shap_data is None:
        print("No SHAP data available for heatmap visualization")
        return
    
    try:
        shap_values = shap_data["shap_values"]
        feature_names = shap_data["feature_names"]
        risk_levels = shap_data["shap_risk_levels"]
        
        # Handle multi-dimensional SHAP values
        if len(shap_values.shape) > 2:
            # If SHAP values have multiple dimensions (e.g., for multi-class models),
            # use the values for class 1 (positive class)
            if shap_values.shape[2] == 2:  # Binary classification
                shap_values = shap_values[:, :, 1]  # Use values for positive class
            else:
                shap_values = np.sum(shap_values, axis=2)  # Sum across all classes
        
        # Convert feature names to strings if needed
        if isinstance(feature_names, np.ndarray):
            feature_names = [str(f) for f in feature_names]
        
        # Calculate top 20 features by overall importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(-mean_abs_shap)[:20]
        
        # Create heatmap data
        heatmap_data = []
        for risk in range(3):
            indices = np.where(risk_levels == risk)[0]
            if len(indices) > 0:
                risk_shap = shap_values[indices][:, top_indices].mean(axis=0)
                heatmap_data.append(risk_shap)
        
        if not heatmap_data:
            print("No data available for heatmap (no samples in any risk level)")
            return
            
        heatmap_data = np.array(heatmap_data)
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        plt.imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
        
        # Add labels
        plt.yticks(range(len(heatmap_data)), ['Low Risk', 'Medium Risk', 'High Risk'][:len(heatmap_data)])
        plt.xticks(range(len(top_indices)), 
                   [str(feature_names[i]) if i < len(feature_names) else f"Feature_{i}" for i in top_indices], 
                   rotation=45, ha='right')
        
        plt.colorbar(label='SHAP value (impact on prediction)')
        plt.title('Feature Importance Heatmap by Risk Level')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"feature_heatmap_epoch{epoch}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[✓] Feature importance heatmap saved to: {save_path}")
    except Exception as e:
        print(f"Error creating feature heatmap: {e}")

def plot_graph_with_explanations(embeddings, sk_ids, epoch, shap_data=None, threshold=0.85, 
                                save_dir="graphs", max_nodes=500, max_edges=1000, labels=None):
    """Create interactive graph visualization with feature importance tooltips"""
    ensure_dir(save_dir)
    
    # Sample nodes if too many
    if len(sk_ids) > max_nodes:
        indices = np.random.choice(len(sk_ids), max_nodes, replace=False)
        embeddings = embeddings[indices]
        sk_ids = sk_ids[indices]
        if labels is not None:
            labels = labels[indices]
    
    # Compute similarities
    normed_embeddings = F.normalize(embeddings, dim=1).cpu().numpy()
    cosine_sim = cosine_similarity(normed_embeddings)
    
    # Create network
    net = Network(height='800px', width='100%', notebook=False, cdn_resources='in_line')
    net.force_atlas_2based(gravity=-50, spring_length=100)
    
    # Map SHAP data to node IDs if available
    shap_explanations = {}
    if shap_data is not None:
        try:
            for i, sk_id in enumerate(shap_data["shap_sk_ids"]):
                node_shap = shap_data["shap_values"][i]
                feature_names = shap_data["feature_names"]
                
                # Handle multi-dimensional SHAP values
                if len(node_shap.shape) > 1:
                    # If SHAP values have multiple dimensions (e.g., for multi-class models),
                    # use the values for class 1 (positive class) or sum across classes
                    if node_shap.shape[1] == 2:  # Binary classification
                        node_shap = node_shap[:, 1]  # Use values for positive class
                    else:
                        node_shap = np.sum(node_shap, axis=1)  # Sum across all classes
                
                # Get top 5 features for this node
                sorted_indices = np.argsort(-np.abs(node_shap))[:5]
                
                # Create top features list with proper scalar conversion
                top_features = []
                for j in sorted_indices:
                    feature_name = str(feature_names[j]) if j < len(feature_names) else f"Feature_{j}"
                    feature_value = float(node_shap[j]) if isinstance(node_shap[j], (np.number, float, int)) else 0.0
                    top_features.append((feature_name, feature_value))
                
                shap_explanations[str(sk_id)] = top_features
        except Exception as e:
            print(f"Error processing SHAP data: {e}")
            print("Continuing without SHAP explanations...")
    
    # Add nodes with explanations in tooltips
    for i, sk in enumerate(sk_ids):
        # Add color based on risk level if labels are provided
        if labels is not None:
            if labels[i] == 0:  # Low risk
                color = '#00cc00'  # Green
            elif labels[i] == 1:  # Medium risk
                color = '#ffcc00'  # Yellow
            else:  # High risk
                color = '#cc0000'  # Red
        else:
            color = '#1f77b4'  # Default blue
            
        # Create tooltip with feature explanations if available
        tooltip = f"ID: {sk}<br>"
        if str(sk) in shap_explanations:
            tooltip += "Top influential features:<br>"
            for feat, val in shap_explanations[str(sk)]:
                direction = "increases risk" if val > 0 else "decreases risk"
                tooltip += f"{feat}: {abs(val):.4f} ({direction})<br>"
        
        # Add node with explanation
        net.add_node(int(sk), label=str(sk), size=10, color=color, title=tooltip)
    
    # Add edges
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
    
    # Add legend for risk levels if labels is not None
    if labels is not None:
        legend_html = """
        <div style="position: absolute; top: 10px; left: 10px; background-color: white; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
        <div style="margin-bottom: 5px;"><span style="display: inline-block; width: 15px; height: 15px; background-color: #00cc00; margin-right: 5px;"></span> Low Risk</div>
        <div style="margin-bottom: 5px;"><span style="display: inline-block; width: 15px; height: 15px; background-color: #ffcc00; margin-right: 5px;"></span> Medium Risk</div>
        <div><span style="display: inline-block; width: 15px; height: 15px; background-color: #cc0000; margin-right: 5px;"></span> High Risk</div>
        </div>
        """
        net.html = net.html.replace("</body>", legend_html + "</body>")
    
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
    
    save_path = os.path.join(save_dir, f"interpretable_graph_epoch{epoch}.html")
    net.save_graph(save_path)
    print(f"[✓] Interpretable graph saved to: {save_path} with {len(sk_ids)} nodes and {edge_count} edges")

def visualize_epoch(epoch, node_folder="node_embeddings", save_dir="graphs", 
                   threshold=0.85, max_nodes=500, max_edges=1000, sample_size=None):
    """Generate all visualizations for the specified epoch"""
    embeddings, sk_ids, clusters, labels, shap_data = load_saved_data(epoch, folder=node_folder)
    
    print(f"Dataset size: {len(sk_ids)} nodes")
    
    # t-SNE visualization
    plot_tsne(embeddings, sk_ids, clusters, epoch, save_dir, sample_size, labels=labels)
    
    # Feature importance by risk level
    if shap_data is not None:
        try:
            plot_feature_importance_by_risk(shap_data, epoch, save_dir)
            create_feature_heatmap(shap_data, epoch, save_dir)
        except Exception as e:
            print(f"Error generating feature importance plots: {e}")
            print("Continuing with other visualizations...")
    
    # Interactive graph visualization with explanations
    plot_graph_with_explanations(embeddings, sk_ids, epoch, shap_data=shap_data,
                               threshold=threshold, save_dir=save_dir, 
                               max_nodes=max_nodes, max_edges=max_edges, labels=labels)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True, help="Epoch number to visualize")
    parser.add_argument('--save_dir', type=str, default='graphs')
    parser.add_argument('--node_folder', type=str, default='node_embeddings')
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--max_nodes', type=int, default=500, help="Max nodes to visualize")
    parser.add_argument('--max_edges', type=int, default=1000, help="Max edges to visualize")
    parser.add_argument('--sample_size', type=int, default=None, help="Sample size for t-SNE")
    parser.add_argument('--skip_shap', action='store_true', help="Skip SHAP visualizations")

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