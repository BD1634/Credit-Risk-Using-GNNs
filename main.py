# ---- main.py ----
import os
from config import opt
from trainer import training_model_classification
from data_preprocessing import data_loaded, feature_engineering
from dataset import data_preprocess, cluster_analysis, data_cluster


def main():
    os.makedirs(opt.data_load_path, exist_ok=True)
    
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = data_loaded()
    data_all = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    data_all, val_cols, emb_cols, bag_size = data_preprocess(data_all)
    selected_attrs = cluster_analysis(data_all, opt.theta_m, opt.theta_u)
    clusters = data_cluster(data_all, selected_attrs)

    # Now just pass None instead of file handle
    best_auc = training_model_classification(data_all, clusters, val_cols, emb_cols, bag_size, f=None)
    print(f"\nBest Validation AUC: {best_auc}")

if __name__ == '__main__':
    main()