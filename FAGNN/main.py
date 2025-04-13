# ---- main.py ----
import os
from config import opt
from trainer import training_model_classification
from data_preprocessing import data_loaded, feature_engineering
from dataset import data_preprocess, cluster_analysis, data_cluster


def check_time_features(df):
    time_cols = [col for col in df.columns if 'DAYS_' in col or 'MONTHS_' in col]
    print(f"\n🕒 Time-related columns in data:")
    for col in time_cols:
        print(f"{col}: min={df[col].min()}, max={df[col].max()}, nulls={df[col].isnull().sum()}")


def main():
    os.makedirs(opt.data_load_path, exist_ok=True)

    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = data_loaded()
    data_all = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)

    check_time_features(data_all)

    data_all, val_cols, emb_cols, bag_size = data_preprocess(data_all)
    selected_attrs = cluster_analysis(data_all, opt.theta_m, opt.theta_u)
    clusters = data_cluster(data_all, selected_attrs)

    sk_ids = data_all['SK_ID_CURR'].tolist()
    best_auc = training_model_classification(data_all, clusters, val_cols, emb_cols, bag_size, sk_ids, f=None)
    print(f"\nBest Validation AUC: {best_auc}")


if __name__ == '__main__':
    main()