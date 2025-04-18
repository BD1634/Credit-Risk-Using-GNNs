# ---- main.py (optimized) ----
import os, time, torch, logging
from config import opt
from trainer import training_model_classification
from data_preprocessing import data_loaded, feature_engineering
from dataset import data_preprocess, cluster_analysis, data_cluster

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def check_time_features(df):
    time_cols = [col for col in df.columns if 'DAYS_' in col or 'MONTHS_' in col]
    logging.info(f"Time‑related columns count: {len(time_cols)}")
    for col in time_cols[:10]:  # print first 10 to avoid clogging output
        logging.debug(f"{col}: min={df[col].min()}, max={df[col].max()}, nulls={df[col].isnull().sum()}")

def main():
    start_time = time.time()
    os.makedirs(opt.data_load_path, exist_ok=True)

    logging.info("Loading data ...")
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = data_loaded()
    logging.info("Feature engineering ...")
    data_all = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)

    check_time_features(data_all)
    data_all, val_cols, emb_cols, bag_size = data_preprocess(data_all)

    selected_attrs = cluster_analysis(data_all, opt.theta_m, opt.theta_u)
    clusters = data_cluster(data_all, selected_attrs)
    sk_ids = data_all['SK_ID_CURR'].tolist()

    logging.info("Starting training ...")
    best_auc, best_model_path = training_model_classification(
        data_all, clusters, val_cols, emb_cols, bag_size, sk_ids
    )

    best_model_weights = torch.load(best_model_path, weights_only=True)

    total_time = time.time() - start_time
    logging.info(f"Finished in {total_time/60:.2f} mins ‑ Best AUC: {best_auc}")
    return best_model_weights

if __name__ == '__main__':
    main()