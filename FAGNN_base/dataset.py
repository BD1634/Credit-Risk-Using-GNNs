# ---- dataset.py ----
import numpy as np
import pandas as pd
from utils import containsAny
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pd.set_option('future.no_silent_downcasting', True)

def data_preprocess(df_data):
    list_columns = df_data.columns.tolist()
    list_columns.remove('SK_ID_CURR')
    list_columns.remove('TARGET')

    value_keywords = ['MEAN', 'SUM', 'MIN', 'MAX', 'COUNT', 'RATIO', 'STD', 'AVG', 'TOTAL', 'MODE', 'MEDI']
    columns_value_selected = [
        'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE',
        'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'EXT_SOURCE_1',
        'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE', 'index'
    ]
    columns_embedd_selected = []

    for col in list_columns:
        if containsAny(col, value_keywords) or len(df_data[col].unique()) > 50:
            columns_value_selected.append(col)
        else:
            columns_embedd_selected.append(col)

    columns_value_selected = list(set(col for col in columns_value_selected if col in df_data.columns))
    columns_embedd_selected = [col for col in columns_embedd_selected if col in df_data.columns]

    for col in columns_value_selected:
        # df_data[col] = df_data[col].fillna(0)
        df_data[col] = df_data[col].fillna(0).infer_objects(copy=False)

    df_data[columns_value_selected] = (
        (df_data[columns_value_selected] - df_data[columns_value_selected].min()) /
        (df_data[columns_value_selected].max() - df_data[columns_value_selected].min())
    ).fillna(0)

    start_encode = 0
    for column in columns_embedd_selected:
        unique_vals = df_data[column].unique()
        mapping = {val: i + start_encode for i, val in enumerate(unique_vals)}
        df_data[column] = df_data[column].map(mapping)
        start_encode += len(unique_vals)

    return df_data, columns_value_selected, columns_embedd_selected, start_encode


def cluster_analysis(data_all, theta_m, theta_u):
    nums = len(data_all)
    dict_column = {col: (len(data_all[col][data_all[col] == 0]) / nums, len(data_all[col].unique())) for col in data_all.columns}
    return [col for col, (zero_ratio, unique_vals) in dict_column.items() if zero_ratio < theta_m and unique_vals < theta_u]


def data_cluster(df_data, columns_object, columns_embedd_selected=None, n_clusters=100):
    """
    Improved clustering function using PCA + K-means with backward compatibility
    
    Args:
        df_data: DataFrame containing the data
        columns_object: List of categorical columns (for backward compatibility)
        columns_embedd_selected: List of encoded categorical columns (optional)
        n_clusters: Target number of clusters (default 100)
        
    Returns:
        List of cluster assignments for each row in df_data
    """
    # Check if we're being called with the old signature
    if columns_embedd_selected is None:
        # We're in legacy mode - columns_object contains the categorical columns
        # Extract numerical columns for clustering
        numerical_cols = [col for col in df_data.columns 
                         if col not in columns_object 
                         and col not in ['SK_ID_CURR', 'TARGET']
                         and df_data[col].dtype in ['int64', 'float64']]
        
        # Apply PCA + K-means on the available numerical columns
        features = numerical_cols
        X = df_data[features].fillna(0)
        
        # Only proceed if we have features to work with
        if len(features) > 0:
            # Apply PCA for dimensionality reduction
            n_components = min(min(50, len(features)), len(X) - 1)
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(X) - 1), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_reduced)
            
            # Create mapping from SK_ID_CURR to cluster ID
            cluster_mapping = {id_val: cluster for id_val, cluster in zip(df_data['SK_ID_CURR'], clusters)}
            
            # Return cluster assignments
            return df_data['SK_ID_CURR'].map(cluster_mapping).tolist()
        else:
            # Fallback to a single cluster if no features available
            return [0] * len(df_data)
    else:
        # We're in new mode - using the improved clustering as originally designed
        # Prepare features (both numerical and encoded categorical)
        features = columns_object + columns_embedd_selected
        X = df_data[features].fillna(0)
        
        # Apply PCA for dimensionality reduction
        n_components = min(min(50, len(features)), len(X) - 1)
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(X) - 1), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_reduced)
        
        # Create mapping from SK_ID_CURR to cluster ID
        cluster_mapping = {id_val: cluster for id_val, cluster in zip(df_data['SK_ID_CURR'], clusters)}
        
        # Return cluster assignments
        return df_data['SK_ID_CURR'].map(cluster_mapping).tolist()


def SMOTE_data(train_df):
    label = train_df['TARGET']
    columns = list(train_df.columns)
    columns_copy = columns.copy()
    columns.remove('TARGET')
    
    # For a 70:30 ratio, we want minority:majority = 30:70 = 3:7 = 0.429
    sm = SMOTE(sampling_strategy=0.429, random_state=42)
    X_res, y_res = sm.fit_resample(train_df[columns], label)
    train_df = pd.DataFrame(data=X_res, columns=columns)
    train_df['TARGET'] = y_res
    
    return train_df[columns_copy]


def auc_calculate(groundtruth, predicted_prob):
    fpr, tpr, _ = roc_curve(groundtruth, predicted_prob, pos_label=1)
    return auc(fpr, tpr)