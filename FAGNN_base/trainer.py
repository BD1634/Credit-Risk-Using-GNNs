# ---- trainer.py ----
import os
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from dataset import auc_calculate
from config import opt, get_device
from model import CLASS_NN_Embed_cluster
from utils import intermediate_feature_distance
from torch.utils.tensorboard import SummaryWriter


def create_sequence_tensor(df, value_column, embed_column):
    grouped = df.groupby('SK_ID_CURR')
    value_seqs, embed_seqs, label_seqs = [], [], []
    for _, group in grouped:
        group = group.sort_values(by='DAYS_INSTALMENT', ascending=True) if 'DAYS_INSTALMENT' in group.columns else group
        value_seqs.append(torch.tensor(group[value_column].values).float())
        embed_seqs.append(torch.tensor(group[embed_column].values).long())
        label_seqs.append(torch.tensor(group['TARGET'].values[0]).long())

    value_tensor = nn.utils.rnn.pad_sequence(value_seqs, batch_first=True)
    embed_tensor = nn.utils.rnn.pad_sequence(embed_seqs, batch_first=True)
    label_tensor = torch.stack(label_seqs)
    return value_tensor, embed_tensor, label_tensor


def training_model_classification(data_all, clusters, value_column, embed_column, bag_size, sk_ids, f=None):
    device = get_device()

    if opt.up_sample:
        pos_idx = data_all[data_all['TARGET'] == 1].index.tolist()
        neg_idx = data_all[data_all['TARGET'] == 0].index.tolist()
        positive_df = pd.DataFrame(np.repeat(data_all.loc[pos_idx].values, int(opt.up_sample), axis=0), columns=data_all.columns)
        data_all = pd.concat([positive_df, data_all.loc[neg_idx]])
        clusters = [clusters[i] for i in (pos_idx * int(opt.up_sample) + neg_idx)]
        sk_ids = [sk_ids[i] for i in (pos_idx * int(opt.up_sample) + neg_idx)]

    if opt.down_sample:
        pos_idx = data_all[data_all['TARGET'] == 1].index.tolist()
        neg_idx = data_all[data_all['TARGET'] == 0].index.tolist()
        neg_sample_idx = random.sample(neg_idx, int(opt.down_sample * len(neg_idx)))
        data_all = pd.concat([data_all.loc[pos_idx], data_all.loc[neg_sample_idx]])
        clusters = [clusters[i] for i in pos_idx + neg_sample_idx]
        sk_ids = [sk_ids[i] for i in pos_idx + neg_sample_idx]

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
    val_sk_ids = [sk_ids[i] for i in val_index]
    train_sk_ids = [sk_ids[i] for i in train_index]

    def create_loader(df):
        val_tensor, emb_tensor, label_tensor = create_sequence_tensor(df, value_column, embed_column)
        dataset = Data.TensorDataset(val_tensor, emb_tensor, label_tensor)
        return Data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True), Data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False)

    train_loader, train_loader_eval = create_loader(train_df)
    val_loader, val_loader_eval = create_loader(val_df)

    model = CLASS_NN_Embed_cluster(
        embedd_columns_num=len(embed_column),
        values_columns_num=len(value_column),
        bag_size=bag_size
    ).to(device)

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

    writer = SummaryWriter(log_dir=os.path.join("runs", time.strftime("%Y-%m-%d_%H-%M-%S")))
    best_auc_val = 0
    count = 0

    for epoch in range(opt.epoch):
        model.train()
        total_loss = 0
        for val, emb, lab in train_loader:
            val, emb, lab = val.to(device), emb.to(device), lab.to(device)
            clusters_tensor = torch.zeros(val.size(0)).to(device)
            optimizer.zero_grad()
            out_cls, out_ae, inter = model(val, emb, clusters_tensor)
            loss_cls = criterion_cls(out_cls, lab)
            flat_ae_target = torch.cat((val, emb.float()), dim=2).mean(dim=1)
            loss_ae = criterion_ae(out_ae, flat_ae_target) / val.size(0)
            loss_cos = intermediate_feature_distance(inter, clusters_tensor)
            loss = opt.lambda_ * loss_cls + opt.alpha_ * loss_ae + opt.beta_ * loss_cos
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, epoch)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Loss in Epoch {epoch}: {avg_loss}")
        if f: f.write(f"Loss in Epoch {epoch}: {avg_loss}\n")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/classification", loss_cls.item(), epoch)
        writer.add_scalar("Loss/autoencoder", loss_ae.item(), epoch)
        writer.add_scalar("Loss/cosine", loss_cos.item(), epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        all_probs, all_labels = [], []
        all_inter = []

        model.eval()
        with torch.no_grad():
            for val, emb, lab in val_loader:
                val, emb = val.to(device), emb.to(device)
                clusters_tensor = torch.zeros(val.size(0)).to(device)
                outputs, _, inter_val = model(val, emb, clusters_tensor)
                probs = F.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(lab.numpy())
                all_inter.append(inter_val.cpu())

        val_auc = auc_calculate(np.array(all_labels), np.array(all_probs))
        print(f"Val AUC in Epoch {epoch}: {val_auc}")
        if f: f.write(f"Val AUC in Epoch {epoch}: {val_auc}\n")
        writer.add_scalar("AUC/val", val_auc, epoch)

        if epoch % 5 == 0:
            os.makedirs("node_embeddings", exist_ok=True)
            full_inter_tensor = torch.cat(all_inter, dim=0)
            torch.save(full_inter_tensor, f"node_embeddings/val_embeddings_epoch_{epoch}.pt")
            np.save(f"node_embeddings/val_sk_ids_epoch_{epoch}.npy", np.array(val_sk_ids))
            np.save(f"node_embeddings/val_clusters_epoch_{epoch}.npy", np.array(val_clusters))

        if val_auc > best_auc_val:
            best_auc_val = val_auc
            torch.save(model.state_dict(), os.path.join(opt.data_load_path, f"best_model_epoch{epoch}.pth"))
            count = 0
        else:
            count += 1

        if count > 10:
            print(f"Early stopping triggered at Epoch {epoch}. Best model saved.")
            break

    writer.close()
    return best_auc_val