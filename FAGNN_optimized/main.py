# # ---- main.py ----
import os
import time
import pickle
import atexit
import logging
from tqdm import tqdm
import multiprocessing
from config import opt
from trainer import training_model_classification
from data_preprocessing import data_loaded, feature_engineering
from dataset import data_preprocess, cluster_analysis, data_cluster

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

# Add this function to clean up multiprocessing resources
def cleanup_resources():
    # Force cleanup of multiprocessing resources
    if hasattr(multiprocessing, '_resource_tracker'):
        if hasattr(multiprocessing._resource_tracker, '_resource_tracker'):
            multiprocessing._resource_tracker._resource_tracker.clear()

# Register the cleanup function to run at exit
atexit.register(cleanup_resources)


def check_time_features(df):
    time_cols = [col for col in df.columns if 'DAYS_' in col or 'MONTHS_' in col]
    print(f"\n🕒 Time-related columns in data: {len(time_cols)} columns total")
    # Optional: you could still print min/max/null statistics as a summary
    print(f"Range examples - min: {min([df[col].min() for col in time_cols])}, " 
          f"max: {max([df[col].max() for col in time_cols])}")
    print(f"Total null values across time columns: {sum([df[col].isnull().sum() for col in time_cols])}")


def main():
    os.makedirs(opt.data_load_path, exist_ok=True)
    
    print("Loading and preprocessing data...")
    t0 = time.time()
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = data_loaded()
    data_all = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    print(f"Data loading and preprocessing took {time.time() - t0:.2f} seconds")

    check_time_features(data_all)

    print("Starting model training preparation...")
    with tqdm(total=4, desc="Setup Progress") as pbar:
        t1 = time.time()
        data_all, val_cols, emb_cols, bag_size = data_preprocess(data_all)
        print(f"Data preprocess took {time.time() - t1:.2f} seconds")
        pbar.update(1)
        
        # Cache cluster analysis results
        cache_file = os.path.join(opt.data_load_path, "cluster_cache.pkl")
        t2 = time.time()
        if os.path.exists(cache_file):
            print("Loading cached cluster analysis...")
            with open(cache_file, 'rb') as f:
                selected_attrs, clusters = pickle.load(f)
        else:
            print("Running cluster analysis...")
            selected_attrs = cluster_analysis(data_all, opt.theta_m, opt.theta_u)
            clusters = data_cluster(data_all, selected_attrs)
            with open(cache_file, 'wb') as f:
                pickle.dump((selected_attrs, clusters), f)
        print(f"Cluster processing took {time.time() - t2:.2f} seconds")
        pbar.update(2)  # Update by 2 since we did two operations
        
        sk_ids = data_all['SK_ID_CURR'].tolist()
        pbar.update(1)
    
    print("Starting model training...")
    t3 = time.time()
    best_auc = training_model_classification(data_all, clusters, val_cols, emb_cols, bag_size, sk_ids, f=None)
    print(f"Training process took {time.time() - t3:.2f} seconds")
    print(f"\nBest Validation AUC: {best_auc}")


if __name__ == '__main__':
    main()