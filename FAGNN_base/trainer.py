# trainer.py
import os
import time
import copy
import shap
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from dataset import auc_calculate
from config import opt, get_device
from monitor import save_interpretability_outputs
from model import CLASS_NN_Embed_cluster
from utils import intermediate_feature_distance
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def create_sequence_tensor(df, value_column, embed_column):
    if 'DAYS_INSTALMENT' in df.columns:
        df = df.sort_values(by=['SK_ID_CURR', 'DAYS_INSTALMENT'])

    grouped = df.groupby('SK_ID_CURR')
    value_seqs, embed_seqs, label_seqs = [], [], []

    for _, group in grouped:
        value_seqs.append(torch.tensor(group[value_column].values).float())
        embed_seqs.append(torch.tensor(group[embed_column].values).long())
        label_seqs.append(torch.tensor(group['TARGET'].values[0]).long())

    val_tensor = nn.utils.rnn.pad_sequence(value_seqs, batch_first=True)
    emb_tensor = nn.utils.rnn.pad_sequence(embed_seqs, batch_first=True)
    label_tensor = torch.stack(label_seqs)

    return val_tensor, emb_tensor, label_tensor


def create_loader(df, batch_size, value_column, embed_column, weighted_sampler=None):
    val_tensor, emb_tensor, label_tensor = create_sequence_tensor(df, value_column, embed_column)
    dataset = Data.TensorDataset(val_tensor, emb_tensor, label_tensor)

    if weighted_sampler is not None:
        sampler = Data.WeightedRandomSampler(weighted_sampler, num_samples=len(weighted_sampler), replacement=True)
        train_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    eval_loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader


def analyze_with_shap(model, val_data, feature_names, device, save_dir="interpretability", samples=100):
    """
    Analyze model predictions using SHAP values
    
    Args:
        model: Trained model
        val_data: Validation dataset (val_tensor, emb_tensor, label_tensor)
        feature_names: List of feature names
        device: Computation device (cuda/cpu)
        save_dir: Directory to save results
        samples: Number of background samples for SHAP
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Get background data for SHAP
    val_tensor, emb_tensor, labels = val_data
    
    # Subsample if needed
    if len(val_tensor) > samples:
        indices = np.random.choice(len(val_tensor), samples, replace=False)
        background_val = val_tensor[indices].to(device)
        background_emb = emb_tensor[indices].to(device)
    else:
        background_val = val_tensor.to(device)
        background_emb = emb_tensor.to(device)
    
    # Create explainer
    def model_predict(x):
        # Convert numpy array to torch tensor
        x_tensor = torch.FloatTensor(x)
        
        # Format input for model
        batch_size = x_tensor.shape[0]
        seq_len = background_val.shape[1]
        val_dim = background_val.shape[2]
        emb_dim = background_emb.shape[2]
        
        # Reshape x back into val and emb tensors
        val_input = x_tensor[:, :seq_len*val_dim].reshape(batch_size, seq_len, val_dim).to(device)
        # For embedding tensor, convert to long after creating torch tensor
        emb_input = x_tensor[:, seq_len*val_dim:].reshape(batch_size, seq_len, emb_dim).to(device).long()
        
        # Get predictions
        with torch.no_grad():
            outputs, _, _ = model(val_input, emb_input)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()
    
    # Flatten background data for SHAP
    flat_background_val = background_val.reshape(background_val.shape[0], -1)
    flat_background_emb = background_emb.reshape(background_emb.shape[0], -1)
    background_data = torch.cat([flat_background_val, flat_background_emb], dim=1).cpu().numpy()
    
    print("Preparing SHAP explainer with background data...")
    
    # Try a more lightweight approach with Explainer class
    try:
        # Use a small subset of validation data for SHAP values to save time
        eval_indices = np.random.choice(len(val_tensor), min(20, len(val_tensor)), replace=False)
        eval_val = val_tensor[eval_indices].to(device)
        eval_emb = emb_tensor[eval_indices].to(device)
        
        # Flatten evaluation data
        flat_eval_val = eval_val.reshape(eval_val.shape[0], -1)
        flat_eval_emb = eval_emb.reshape(eval_emb.shape[0], -1)
        eval_data = torch.cat([flat_eval_val, flat_eval_emb], dim=1).cpu().numpy()
        
        # For speed and simplicity, use a more basic explainer
        print("Computing SHAP values with sampling approach...")
        explainer = shap.KernelExplainer(model_predict, background_data[:10])  # Use fewer background samples
        shap_values = explainer.shap_values(eval_data[:10], nsamples=50)  # Calculate for fewer samples
        
        # If successful with small sample, try with full data
        if isinstance(shap_values, list):
            # For multi-class output, index 1 is for the positive class
            shap_values = shap_values[1]
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, eval_data[:10], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"shap_summary.png"))
        plt.close()
        
        return shap_values, eval_indices[:10]  # Return SHAP values for positive class
    
    except Exception as e:
        print(f"Error in SHAP calculation: {e}")
        print("Continuing without SHAP values...")
        # Return placeholder values so training can continue
        dummy_shape = (min(10, len(val_tensor)), len(feature_names))
        return np.zeros(dummy_shape), np.arange(min(10, len(val_tensor)))


def training_model_classification(data_all, clusters, value_column, embed_column, bag_size, sk_ids, f=None):
    device = get_device()
    print("Preparing training data...")

    if opt.up_sample:
        pos_idx = data_all[data_all['TARGET'] == 1].index.tolist()
        neg_idx = data_all[data_all['TARGET'] == 0].index.tolist()
        sampler_weights = torch.FloatTensor([
            int(opt.up_sample) if i in pos_idx else 1.0 for i in range(len(data_all))
        ])
    elif opt.down_sample:
        pos_idx = data_all[data_all['TARGET'] == 1].index.tolist()
        neg_idx = data_all[data_all['TARGET'] == 0].index.tolist()
        neg_sample_idx = random.sample(neg_idx, int(opt.down_sample * len(neg_idx)))
        keep_idx = pos_idx + neg_sample_idx
        data_all = data_all.loc[keep_idx].reset_index(drop=True)
        clusters = [clusters[i] for i in keep_idx]
        sk_ids = [sk_ids[i] for i in keep_idx]
        sampler_weights = None
    else:
        sampler_weights = None

    # Train/val split
    indices = list(range(len(data_all)))
    random.shuffle(indices)
    val_size = int(len(data_all) * opt.valid_portion)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_df = data_all.iloc[train_indices].reset_index(drop=True)
    val_df = data_all.iloc[val_indices].reset_index(drop=True)
    train_sk_ids = [sk_ids[i] for i in train_indices]
    val_sk_ids = [sk_ids[i] for i in val_indices]
    val_clusters = [clusters[i] for i in val_indices]

    train_loader, _ = create_loader(train_df, opt.batchSize, value_column, embed_column, weighted_sampler=sampler_weights)
    val_loader, _ = create_loader(val_df, opt.batchSize, value_column, embed_column)

    model = CLASS_NN_Embed_cluster(
        embedd_columns_num=len(embed_column),
        values_columns_num=len(value_column),
        bag_size=bag_size
    ).to(device)
    
    # Create EMA model
    ema_model = copy.deepcopy(model)
    ema_decay = 0.999

    # Lower initial learning rate further
    initial_lr = 2e-4  # Reduced from 5e-4 to further stabilize training
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    # Cosine annealing schedule with warm restarts - increased T_0 for more gradual changes
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

    writer = SummaryWriter(log_dir=os.path.join("runs", time.strftime("%Y-%m-%d_%H-%M-%S")))
    best_auc_val = 0
    best_val_loss = float('inf')
    patience = 0
    early_stop_patience = 5
    
    # Initialize validation metrics history for adaptive weights
    val_loss_history = []
    val_cls_loss_history = []
    val_ae_loss_history = []
    
    # Create feature names for interpretability
    feature_names = []
    for col in value_column:
        for i in range(bag_size):
            feature_names.append(f"{col}_t{i}")
    for col in embed_column:
        for i in range(bag_size):
            feature_names.append(f"{col}_t{i}")
    
    # Warmup epochs and total epochs for scheduling
    warmup_epochs = 5  # Increased from 3 to 5 for more stability
    total_epochs = opt.epoch

    for epoch in range(opt.epoch):
        model.train()
        total_loss = 0
        
        # Improved weight scheduling with warmup and cosine annealing
        if epoch < warmup_epochs:
            # During warmup, keep weights stable
            lambda_weight = opt.lambda_
            alpha_weight = opt.alpha_
        else:
            # After warmup, use cosine annealing for smoother transitions
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lambda_weight = opt.lambda_ * 0.5 * (1 + np.cos(np.pi * progress))
            alpha_weight = opt.alpha_ * (1.5 - 0.5 * np.cos(np.pi * progress))
            
            # More conservative adaptive adjustment based on validation loss trends
            if len(val_cls_loss_history) >= 3:
                cls_trend = sum(val_cls_loss_history[-i] - val_cls_loss_history[-i-1] for i in range(1, 3))
                ae_trend = sum(val_ae_loss_history[-i] - val_ae_loss_history[-i-1] for i in range(1, 3))
                
                # Smaller adjustment steps (0.05 instead of 0.1)
                lambda_adjustment = np.sign(cls_trend) * 0.05
                alpha_adjustment = np.sign(ae_trend) * 0.05
                
                lambda_weight = lambda_weight * (1 + lambda_adjustment)
                alpha_weight = alpha_weight * (1 + alpha_adjustment)
                
                # Tighter bounds to prevent large oscillations
                lambda_weight = max(0.4, min(0.8, lambda_weight))
                alpha_weight = max(1.2, min(1.8, alpha_weight))

        # Print current weights for monitoring
        print(f"Epoch {epoch}: lambda_weight={lambda_weight:.4f}, alpha_weight={alpha_weight:.4f}")
        
        epoch_cls_loss = 0
        epoch_ae_loss = 0
        epoch_cos_loss = 0

        for val, emb, lab in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epoch}", leave=False):
            val, emb, lab = val.to(device), emb.to(device), lab.to(device)
            optimizer.zero_grad()
            out_cls, out_ae, inter = model(val, emb)

            loss_cls = criterion_cls(out_cls, lab)
            flat_ae_target = torch.cat((val, emb.float()), dim=2).mean(dim=1)
            loss_ae = criterion_ae(out_ae, flat_ae_target) / val.size(0)
            loss_cos = intermediate_feature_distance(inter, lab)
            
            # Apply dynamic weights
            loss = lambda_weight * loss_cls + alpha_weight * loss_ae + opt.beta_ * loss_cos
            
            loss.backward()
            
            # Stronger gradient clipping to prevent spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # Update EMA model
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            
            total_loss += loss.item()
            
            # Track loss components
            epoch_cls_loss += loss_cls.item()
            epoch_ae_loss += loss_ae.item()
            epoch_cos_loss += loss_cos.item()

        # Step the scheduler
        scheduler.step()
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_cls_loss = epoch_cls_loss / len(train_loader)
        avg_ae_loss = epoch_ae_loss / len(train_loader)
        avg_cos_loss = epoch_cos_loss / len(train_loader)
        
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
        print(f"  - Classification loss: {avg_cls_loss:.4f}")
        print(f"  - Reconstruction loss: {avg_ae_loss:.4f}")
        print(f"  - Contrastive loss: {avg_cos_loss:.4f}")
        
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)
        
        # Additional metrics tracking
        writer.add_scalar("Loss/classification", avg_cls_loss, epoch)
        writer.add_scalar("Loss/reconstruction", avg_ae_loss, epoch)
        writer.add_scalar("Loss/contrastive", avg_cos_loss, epoch)
        writer.add_scalar("LossWeights/classification", lambda_weight, epoch)
        writer.add_scalar("LossWeights/reconstruction", alpha_weight, epoch)

        # Validation - use EMA model for more stable evaluation
        ema_model.eval()
        all_probs, all_labels, all_inter = [], [], []
        val_loss = 0
        val_cls_loss = 0
        val_ae_loss = 0
        
        # Store validation tensors for SHAP analysis
        val_tensors, emb_tensors = [], []
        
        with torch.no_grad():
            for val, emb, lab in tqdm(val_loader, desc="Validation", leave=False):
                val, emb, lab = val.to(device), emb.to(device), lab.to(device)
                # Store for SHAP analysis
                val_tensors.append(val.cpu())
                emb_tensors.append(emb.cpu())
                
                # Use EMA model for validation
                outputs, out_ae, inter_val = ema_model(val, emb)
                
                # Calculate validation loss components
                loss_cls_val = criterion_cls(outputs, lab)
                flat_ae_target = torch.cat((val, emb.float()), dim=2).mean(dim=1)
                loss_ae_val = criterion_ae(out_ae, flat_ae_target) / val.size(0)
                
                # Total validation loss
                loss_val = lambda_weight * loss_cls_val + alpha_weight * loss_ae_val
                val_loss += loss_val.item()
                val_cls_loss += loss_cls_val.item()
                val_ae_loss += loss_ae_val.item()
                
                probs = F.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(lab.cpu().numpy())
                all_inter.append(inter_val.cpu())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls_loss = val_cls_loss / len(val_loader)
        avg_val_ae_loss = val_ae_loss / len(val_loader)
        
        # Store validation loss components for adaptive weight adjustment
        val_loss_history.append(avg_val_loss)
        val_cls_loss_history.append(avg_val_cls_loss)
        val_ae_loss_history.append(avg_val_ae_loss)
        
        # Validation-based learning rate adjustment
        if len(val_loss_history) > 0 and avg_val_loss > val_loss_history[-1]:
            # Reduce learning rate when validation loss increases
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
                print(f"Reducing learning rate to {param_group['lr']:.6f} due to validation loss increase")
        
        val_auc = auc_calculate(np.array(all_labels), np.array(all_probs))
        
        print(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}")
        print(f"  - Val Classification loss: {avg_val_cls_loss:.4f}")
        print(f"  - Val Reconstruction loss: {avg_val_ae_loss:.4f}")
        print(f"Epoch {epoch}: Validation AUC = {val_auc:.4f}")
        
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Loss/val_classification", avg_val_cls_loss, epoch)
        writer.add_scalar("Loss/val_reconstruction", avg_val_ae_loss, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)

        # Save visualization data every 5 epochs with adjusted percentile-based risk levels
        if epoch % 5 == 0:
            full_inter_tensor = torch.cat(all_inter, dim=0)
            all_probs_np = np.array(all_probs)
            all_labels_np = np.array(all_labels)
            
            # Adjusted percentiles for more business-appropriate risk distribution
            low_thresh = np.percentile(all_probs_np, 40)  # Bottom 40%
            high_thresh = np.percentile(all_probs_np, 80)  # Top 20%
            
            risk_levels = np.zeros_like(all_labels_np)
            risk_levels[all_probs_np > low_thresh] = 1  # Medium risk (40% to 80%)
            risk_levels[all_probs_np > high_thresh] = 2  # High risk (top 20%)
            
            # Print risk level distribution and thresholds
            print(f"Adaptive thresholds: low={low_thresh:.3f}, high={high_thresh:.3f}")
            unique, counts = np.unique(risk_levels, return_counts=True)
            print(f"Risk level distribution at epoch {epoch}:")
            for level, count in zip(unique, counts):
                print(f"  Level {level}: {count} samples ({count/len(risk_levels)*100:.2f}%)")
            
            # Skip SHAP analysis if user doesn't have shap installed or version is incompatible
            try:
                # Concatenate validation tensors for SHAP analysis
                val_tensor = torch.cat(val_tensors, dim=0)
                emb_tensor = torch.cat(emb_tensors, dim=0)
                
                # Calculate SHAP values with error handling
                print("Calculating SHAP values for interpretability...")
                shap_values, shap_indices = analyze_with_shap(
                    ema_model, (val_tensor, emb_tensor, all_labels_np), 
                    feature_names, device
                )
                
                # Get the correct SK_IDs for the SHAP samples
                shap_sk_ids = [val_sk_ids[i] for i in shap_indices]
                shap_clusters = [val_clusters[i] for i in shap_indices]
                shap_risk_levels = risk_levels[shap_indices]
                
                # Save embeddings and SHAP values
                save_interpretability_outputs(
                    full_inter_tensor, val_sk_ids, val_clusters, epoch, 
                    risk_levels=risk_levels, 
                    shap_values=shap_values,
                    shap_indices=shap_indices, 
                    shap_sk_ids=shap_sk_ids,
                    shap_clusters=shap_clusters,
                    shap_risk_levels=shap_risk_levels,
                    feature_names=feature_names
                )
            except Exception as e:
                print(f"Error in SHAP analysis: {e}")
                print("Saving data without SHAP values...")
                # Save embeddings without SHAP values
                save_interpretability_outputs(
                    full_inter_tensor, val_sk_ids, val_clusters, epoch, 
                    risk_levels=risk_levels
                )

        # Save model based on both metrics
        saved_model = False
        if val_auc > best_auc_val:
            best_auc_val = val_auc
            # Save both regular and EMA models
            torch.save(model.state_dict(), os.path.join(opt.data_load_path, f"best_model_auc_epoch{epoch}.pth"))
            torch.save(model.state_dict(), os.path.join(opt.data_load_path, "best_model_auc.pth"))
            torch.save(ema_model.state_dict(), os.path.join(opt.data_load_path, "best_ema_model_auc.pth"))
            patience = 0
            saved_model = True
            print(f"New best model saved with validation AUC: {val_auc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save both regular and EMA models
            torch.save(model.state_dict(), os.path.join(opt.data_load_path, f"best_model_loss_epoch{epoch}.pth"))
            torch.save(model.state_dict(), os.path.join(opt.data_load_path, "best_model_loss.pth"))
            torch.save(ema_model.state_dict(), os.path.join(opt.data_load_path, "best_ema_model_loss.pth"))
            patience = 0
            saved_model = True
            print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
            
        if not saved_model:
            patience += 1
            if patience > early_stop_patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    writer.close()
    
    # Load the best EMA model (AUC-based) before returning
    try:
        model.load_state_dict(torch.load(os.path.join(opt.data_load_path, "best_ema_model_auc.pth")))
        print("Loaded best EMA model based on AUC")
    except:
        try:
            model.load_state_dict(torch.load(os.path.join(opt.data_load_path, "best_model_auc.pth")))
            print("Loaded best model based on AUC")
        except:
            print("Could not load best AUC model, using current model state")
    
    return best_auc_val