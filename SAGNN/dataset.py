# ---- dataset.py ----
import numpy as np
import pandas as pd
from utils import containsAny
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc


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
        df_data[col] = df_data[col].fillna(0)

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


def data_cluster(df_data, columns_object):
    df_data[columns_object] = df_data[columns_object].fillna(0)
    groups = df_data.groupby(columns_object)
    dict_group = {}
    for count, (_, group) in enumerate(groups):
        for id_val in group['SK_ID_CURR']:
            dict_group[id_val] = count
    return df_data['SK_ID_CURR'].map(dict_group).tolist()


def SMOTE_data(train_df):
    label = train_df['TARGET']
    columns = list(train_df.columns)
    columns_copy = columns.copy()
    columns.remove('TARGET')
    sm = SMOTE(sampling_strategy=1, random_state=42)
    X_res, y_res = sm.fit_resample(train_df[columns], label)
    train_df = pd.DataFrame(data=X_res, columns=columns)
    train_df['TARGET'] = y_res
    return train_df[columns_copy]


def auc_calculate(groundtruth, predicted_prob):
    fpr, tpr, _ = roc_curve(groundtruth, predicted_prob, pos_label=1)
    return auc(fpr, tpr)