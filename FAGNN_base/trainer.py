# trainer.py
import os
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data as Data
from dataset import auc_calculate
from config import opt, get_device
from monitor import save_lstm_outputs
from model import CLASS_NN_Embed_cluster
from utils import intermediate_feature_distance
from torch.utils.tensorboard import SummaryWriter


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

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.1)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

    writer = SummaryWriter(log_dir=os.path.join("runs", time.strftime("%Y-%m-%d_%H-%M-%S")))
    best_auc_val = 0
    patience = 0

    for epoch in range(opt.epoch):
        model.train()
        total_loss = 0

        for val, emb, lab in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epoch}", leave=False):
            val, emb, lab = val.to(device), emb.to(device), lab.to(device)
            optimizer.zero_grad()
            out_cls, out_ae, inter = model(val, emb)

            loss_cls = criterion_cls(out_cls, lab)
            flat_ae_target = torch.cat((val, emb.float()), dim=2).mean(dim=1)
            loss_ae = criterion_ae(out_ae, flat_ae_target) / val.size(0)
            loss_cos = intermediate_feature_distance(inter, lab)
            loss = opt.lambda_ * loss_cls + opt.alpha_ * loss_ae + opt.beta_ * loss_cos
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Validation
        model.eval()
        all_probs, all_labels, all_inter = [], [], []
        with torch.no_grad():
            for val, emb, lab in tqdm(val_loader, desc="Validation", leave=False):
                val, emb = val.to(device), emb.to(device)
                outputs, _, inter_val = model(val, emb)
                probs = F.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(lab.numpy())
                all_inter.append(inter_val.cpu())

        val_auc = auc_calculate(np.array(all_labels), np.array(all_probs))
        print(f"Epoch {epoch}: Validation AUC = {val_auc:.4f}")
        writer.add_scalar("AUC/val", val_auc, epoch)

        if epoch % 5 == 0:
            full_inter_tensor = torch.cat(all_inter, dim=0)
            save_lstm_outputs(full_inter_tensor, val_sk_ids, val_clusters, epoch)

        if val_auc > best_auc_val:
            best_auc_val = val_auc
            torch.save(model.state_dict(), os.path.join(opt.data_load_path, f"best_model_epoch{epoch}.pth"))
            patience = 0
        else:
            patience += 1
            if patience > 10:
                print("Early stopping triggered.")
                break

    writer.close()
    return best_auc_val