# ---- trainer.py (optimized) ----
import os, time, torch, random, numpy as np, pandas as pd, torch.nn as nn, torch.utils.data as Data
import torch.nn.functional as F
from dataset import auc_calculate
from config import opt, get_device
from model import CLASS_NN_Embed_cluster
from utils import intermediate_feature_distance
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True  # auto‑tune
device = get_device()

def _compile_if_available(model):
    # PyTorch 2.0+ dynamic graph compilation
    if hasattr(torch, 'compile'):
        try:
            return torch.compile(model, mode='reduce-overhead')
        except Exception:
            pass
    return model

def create_loader(df, cluster_list, value_column, embed_column):
    pin = device.type == 'cuda'
    value_tensor = torch.tensor(df[value_column].values, dtype=torch.float32)
    embed_tensor = torch.tensor(df[embed_column].values, dtype=torch.long)
    cluster_tensor = torch.tensor(cluster_list, dtype=torch.long)
    label_tensor = torch.tensor(df['TARGET'].values, dtype=torch.long)
    dataset = Data.TensorDataset(value_tensor, embed_tensor, cluster_tensor, label_tensor)
    return Data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
                           num_workers=os.cpu_count()//2, pin_memory=pin, persistent_workers=True)

def training_model_classification(data_all, clusters, value_column, embed_column, bag_size, sk_ids):
    global device
    if torch.cuda.is_available():
        print("Using CUDA GPU for training.")
    elif torch.backends.mps.is_available():
        print("Using Apple MPS GPU for training.")
    else:
        print("Using CPU for training.")

    # ----- sampling (unchanged) -----
    pos_idx = data_all[data_all['TARGET'] == 1].index.tolist()
    neg_idx = data_all[data_all['TARGET'] == 0].index.tolist()
    # simple validation split
    val_mask = np.zeros(len(data_all), dtype=bool)
    val_mask[::int(1/opt.valid_portion)] = True

    train_df = data_all[~val_mask]
    val_df = data_all[val_mask]
    train_clusters = [clusters[i] for i in np.where(~val_mask)[0]]
    val_clusters   = [clusters[i] for i in np.where(val_mask)[0]]

    train_loader = create_loader(train_df, train_clusters, value_column, embed_column)
    val_loader   = create_loader(val_df,   val_clusters,   value_column, embed_column)

    model = CLASS_NN_Embed_cluster(
        embedd_columns_num=len(embed_column),
        values_columns_num=len(value_column),
        bag_size=bag_size
    ).to(device)
    model = _compile_if_available(model)

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

    writer = SummaryWriter(log_dir=os.path.join("runs", time.strftime("%Y-%m-%d_%H-%M-%S")))
    best_auc_val, best_epoch = 0, -1
    best_model_path = None
    patience, trigger = 10, 0

    for epoch in range(opt.epoch):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        for val_batch, emb_batch, clu_batch, lab_batch in train_loader:
            val_batch, emb_batch, clu_batch, lab_batch = val_batch.to(device), emb_batch.to(device), clu_batch.to(device), lab_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out_cls, out_ae, inter = model(val_batch, emb_batch, clu_batch)
            loss_cls = criterion_cls(out_cls, lab_batch)
            loss_ae  = criterion_ae(out_ae, torch.cat((val_batch, emb_batch.float()), dim=1)) / val_batch.size(0)
            loss_cos = intermediate_feature_distance(inter, clu_batch)
            loss     = opt.lambda_ * loss_cls + opt.alpha_ * loss_ae + opt.beta_ * loss_cos
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            all_probs, all_labels = [], []
            for val_batch, emb_batch, clu_batch, lab_batch in val_loader:
                val_batch, emb_batch, clu_batch = val_batch.to(device), emb_batch.to(device), clu_batch.to(device)
                outputs, _, _ = model(val_batch, emb_batch, clu_batch)
                probs = F.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(lab_batch.numpy())
        val_auc = auc_calculate(np.array(all_labels), np.array(all_probs))
        print(f"Epoch {epoch+1:03d} | Loss {total_loss/len(train_loader):.4f} | Val AUC {val_auc:.4f} | Time {(time.time()-epoch_start):.1f}s")
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_scalar("Loss/train", total_loss/len(train_loader), epoch)

        if val_auc > best_auc_val:
            best_auc_val, best_epoch = val_auc, epoch
            best_model_path = os.path.join(opt.data_load_path, f"best_model_epoch{best_epoch}.pth")
            torch.save(model.state_dict(), best_model_path)
            trigger = 0
        else:
            trigger += 1
        if trigger >= patience:
            print("Early stopping triggered.")
            break

    writer.close()
    return best_auc_val, best_model_path