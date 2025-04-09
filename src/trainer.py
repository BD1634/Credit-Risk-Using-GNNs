#trainer.py
import os
import time
import torch
import random
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from .dataset import auc_calculate
from .config import opt, get_device
from .model import CLASS_NN_Embed_cluster
from .utils import intermediate_feature_distance
from torch.utils.tensorboard import SummaryWriter



def training_model_classification(data_all, clusters, value_column, embed_column, bag_size, f=None):
    device = get_device()

    if opt.up_sample:
        pos_idx = data_all[data_all['TARGET'] == 1].index.tolist()
        neg_idx = data_all[data_all['TARGET'] == 0].index.tolist()
        positive_df = pd.DataFrame(np.repeat(data_all.loc[pos_idx].values, int(opt.up_sample), axis=0), columns=data_all.columns)
        data_all = pd.concat([positive_df, data_all.loc[neg_idx]])
        clusters = [clusters[i] for i in (pos_idx * int(opt.up_sample) + neg_idx)]

    if opt.down_sample:
        pos_idx = data_all[data_all['TARGET'] == 1].index.tolist()
        neg_idx = data_all[data_all['TARGET'] == 0].index.tolist()
        neg_sample_idx = random.sample(neg_idx, int(opt.down_sample * len(neg_idx)))
        data_all = pd.concat([data_all.loc[pos_idx], data_all.loc[neg_sample_idx]])
        clusters = [clusters[i] for i in pos_idx + neg_sample_idx]

    val_index, train_index = [], []
    val_clusters, train_clusters = [], []
    for i, cluster in enumerate(clusters):
        if i % int(1 / opt.valid_portion) == 0:
            val_index.append(i)
            val_clusters.append(cluster)
        else:
            train_index.append(i)
            train_clusters.append(cluster)

    val_df = data_all.iloc[val_index]
    train_df = data_all.iloc[train_index]

    def create_loader(df, cluster_list):
        value_tensor = torch.tensor(df[value_column].values).float()
        embed_tensor = torch.tensor(df[embed_column].values).long()
        cluster_tensor = torch.tensor(cluster_list)
        label_tensor = torch.tensor(df['TARGET'].values).long()
        dataset = Data.TensorDataset(value_tensor, embed_tensor, cluster_tensor, label_tensor)
        return Data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True), Data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False)

    train_loader, train_loader_eval = create_loader(train_df, train_clusters)
    val_loader, val_loader_eval = create_loader(val_df, val_clusters)

    model = CLASS_NN_Embed_cluster(
        embedd_columns_num=len(embed_column),
        values_columns_num=len(value_column),
        bag_size=bag_size
    ).to(device)

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

    # 🔥 Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join("runs", time.strftime("%Y-%m-%d_%H-%M-%S")))

    best_auc_val = 0
    count = 0

    for epoch in range(opt.epoch):
        model.train()
        total_loss = 0
        for val, emb, clu, lab in train_loader:
            val, emb, clu, lab = val.to(device), emb.to(device), clu.to(device), lab.to(device)
            optimizer.zero_grad()
            out_cls, out_ae, inter = model(val, emb, clu)
            loss_cls = criterion_cls(out_cls, lab)
            loss_ae = criterion_ae(out_ae, torch.cat((val, emb.float()), dim=1)) / val.size(0)
            loss_cos = intermediate_feature_distance(inter, clu)
            loss = opt.lambda_ * loss_cls + opt.alpha_ * loss_ae + opt.beta_ * loss_cos
            loss.backward()

            # Log gradient histogram
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, epoch)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        print(f"Loss in Epoch {epoch}: {avg_loss}")
        if f:
            f.write(f"Loss in Epoch {epoch}: {avg_loss}\n")

        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Validation AUC
        auc_val = []
        model.eval()
        with torch.no_grad():
            for val, emb, clu, lab in val_loader:
                val, emb, clu = val.to(device), emb.to(device), clu.to(device)
                outputs, _, _ = model(val, emb, clu)
                probs = F.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
                auc_val.append(auc_calculate(lab.numpy(), probs))

        val_auc = np.mean(auc_val)
        print(f"Val AUC in Epoch {epoch}: {val_auc}")
        if f:
            f.write(f"Val AUC in Epoch {epoch}: {val_auc}\n")

        writer.add_scalar("AUC/val", val_auc, epoch)

        if val_auc > best_auc_val:
            best_auc_val = val_auc
            best_model = model
            count = 0
            print(f"Best Val AUC in Epoch {epoch}: {best_auc_val}")
        else:
            count += 1

        if count > 10:
            path = os.path.join(opt.data_load_path, f"best_model_epoch{epoch}.pth")
            torch.save(best_model.state_dict(), path)
            print(f"Early stopping triggered at Epoch {epoch}. Best model saved.")
            break

    writer.close()
    return best_auc_val