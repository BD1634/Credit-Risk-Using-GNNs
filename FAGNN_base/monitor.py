# monitor.py
import os
import torch
import numpy as np

def save_interpretability_outputs(embeddings, sk_ids, clusters, epoch, folder="node_embeddings", 
                                  risk_levels=None, shap_values=None, shap_indices=None,
                                  shap_sk_ids=None, shap_clusters=None, shap_risk_levels=None,
                                  feature_names=None):
    """
    Saves embeddings, IDs, clusters, risk levels and SHAP values for interpretability
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