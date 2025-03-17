import polars as pl
import numpy as np
import os


def load_and_cast_csv(file_path, dtype_dict):
    df = pl.scan_csv(file_path)
    casted_df = df.with_columns([
        pl.col(col).cast(dtype_dict[col]) for col in dtype_dict
    ])
    return casted_df

def load_data(PATH):

    application_data = pl.scan_csv(os.path.join(PATH,'application_train.csv'),encoding='utf8')
    bureau_data = pl.scan_csv(os.path.join(PATH,'bureau.csv'))
    bureau_balance_data = pl.scan_csv(os.path.join(PATH,'bureau_balance.csv'))
    previous_application_data = pl.scan_csv(os.path.join(PATH,'previous_application.csv'))
    

    pos_dtype = {
    'SK_ID_PREV': pl.UInt32, 'SK_ID_CURR': pl.UInt32, 'MONTHS_BALANCE': pl.Int32, 'SK_DPD': pl.Int32,
    'SK_DPD_DEF': pl.Int32, 'CNT_INSTALMENT': pl.Float32, 'CNT_INSTALMENT_FUTURE': pl.Float32
    }

    install_dtype = {
    'SK_ID_PREV': pl.UInt32, 'SK_ID_CURR': pl.UInt32, 'NUM_INSTALMENT_NUMBER': pl.Int32,
    'NUM_INSTALMENT_VERSION': pl.Float32, 'DAYS_INSTALMENT': pl.Float32, 'DAYS_ENTRY_PAYMENT': pl.Float32,
    'AMT_INSTALMENT': pl.Float32, 'AMT_PAYMENT': pl.Float32
    }   

    card_dtype = {
    'SK_ID_PREV': pl.UInt32, 'SK_ID_CURR': pl.UInt32, 'MONTHS_BALANCE': pl.Int16,
    'AMT_CREDIT_LIMIT_ACTUAL': pl.Int32, 'CNT_DRAWINGS_CURRENT': pl.Int32, 'SK_DPD': pl.Int32,
    'SK_DPD_DEF': pl.Int32, 'AMT_BALANCE': pl.Float32, 'AMT_DRAWINGS_ATM_CURRENT': pl.Float32,
    'AMT_DRAWINGS_CURRENT': pl.Float32, 'AMT_DRAWINGS_OTHER_CURRENT': pl.Float32,
    'AMT_DRAWINGS_POS_CURRENT': pl.Float32, 'AMT_INST_MIN_REGULARITY': pl.Float32,
    'AMT_PAYMENT_CURRENT': pl.Float32, 'AMT_PAYMENT_TOTAL_CURRENT': pl.Float32,
    'AMT_RECEIVABLE_PRINCIPAL': pl.Float32, 'AMT_RECIVABLE': pl.Float32,
    'AMT_TOTAL_RECEIVABLE': pl.Float32, 'CNT_DRAWINGS_ATM_CURRENT': pl.Float32,
    'CNT_DRAWINGS_OTHER_CURRENT': pl.Float32, 'CNT_DRAWINGS_POS_CURRENT': pl.Float32,
    'CNT_INSTALMENT_MATURE_CUM': pl.Float32
    }

    pos_cash_bal_data = load_and_cast_csv(os.path.join(PATH, 'POS_CASH_balance.csv'), pos_dtype)
    installments_payments_data = load_and_cast_csv(os.path.join(PATH, 'installments_payments.csv'), install_dtype)
    credit_card_bal_data = load_and_cast_csv(os.path.join(PATH, 'credit_card_balance.csv'), card_dtype)
    

    return application_data , bureau_data , bureau_balance_data, previous_application_data, pos_cash_bal_data, installments_payments_data, credit_card_bal_data

def process_app_data(apps):

    apps = apps.with_columns([
    (pl.col('EXT_SOURCE_1') + pl.col('EXT_SOURCE_2') + pl.col('EXT_SOURCE_3')) / 3.0
    .alias('APPS_EXT_SOURCE_MEAN'),
    
    (pl.col('EXT_SOURCE_1')**2 + pl.col('EXT_SOURCE_2')**2 + pl.col('EXT_SOURCE_3')**2 - 
     ((pl.col('EXT_SOURCE_1') + pl.col('EXT_SOURCE_2') + pl.col('EXT_SOURCE_3'))**2) / 3.0)
    .sqrt().alias('APPS_EXT_SOURCE_STD')
    ])

    # Fill missing values for standard deviation
    apps = apps.with_columns([
        pl.when(pl.col('APPS_EXT_SOURCE_STD').is_null())
        .then(pl.col('APPS_EXT_SOURCE_STD').mean())
        .otherwise(pl.col('APPS_EXT_SOURCE_STD'))
        .alias('APPS_EXT_SOURCE_STD')
    ])

    # AMT_CREDIT features
    apps = apps.with_columns([
        (pl.col('AMT_ANNUITY') / pl.col('AMT_CREDIT')).alias('APPS_ANNUITY_CREDIT_RATIO'),
        (pl.col('AMT_GOODS_PRICE') / pl.col('AMT_CREDIT')).alias('APPS_GOODS_CREDIT_RATIO')
    ])

    # AMT_INCOME_TOTAL features
    apps = apps.with_columns([
        (pl.col('AMT_ANNUITY') / pl.col('AMT_INCOME_TOTAL')).alias('APPS_ANNUITY_INCOME_RATIO'),
        (pl.col('AMT_CREDIT') / pl.col('AMT_INCOME_TOTAL')).alias('APPS_CREDIT_INCOME_RATIO'),
        (pl.col('AMT_GOODS_PRICE') / pl.col('AMT_INCOME_TOTAL')).alias('APPS_GOODS_INCOME_RATIO'),
        (pl.col('AMT_INCOME_TOTAL') / pl.col('CNT_FAM_MEMBERS')).alias('APPS_CNT_FAM_INCOME_RATIO')
    ])

    # DAYS_BIRTH, DAYS_EMPLOYED features
    apps = apps.with_columns([
        (pl.col('DAYS_EMPLOYED') / pl.col('DAYS_BIRTH')).alias('APPS_EMPLOYED_BIRTH_RATIO'),
        (pl.col('AMT_INCOME_TOTAL') / pl.col('DAYS_EMPLOYED')).alias('APPS_INCOME_EMPLOYED_RATIO'),
        (pl.col('AMT_INCOME_TOTAL') / pl.col('DAYS_BIRTH')).alias('APPS_INCOME_BIRTH_RATIO'),
        (pl.col('OWN_CAR_AGE') / pl.col('DAYS_BIRTH')).alias('APPS_CAR_BIRTH_RATIO'),
        (pl.col('OWN_CAR_AGE') / pl.col('DAYS_EMPLOYED')).alias('APPS_CAR_EMPLOYED_RATIO')
    ])

    return apps


def process_prev_data(prev):

    prev = prev.with_columns([
        (pl.col('AMT_APPLICATION') - pl.col('AMT_CREDIT')).alias('PREV_CREDIT_DIFF'),
        (pl.col('AMT_APPLICATION') - pl.col('AMT_GOODS_PRICE')).alias('PREV_GOODS_DIFF'),
        (pl.col('AMT_CREDIT') / pl.col('AMT_APPLICATION')).alias('PREV_CREDIT_APPL_RATIO'),
        (pl.col('AMT_GOODS_PRICE') / pl.col('AMT_APPLICATION')).alias('PREV_GOODS_APPL_RATIO')
    ])

    # Data Cleansing - Replace specific values with NaN
    prev = prev.with_columns([
        pl.when(pl.col('DAYS_FIRST_DRAWING') == 365243).then(None).otherwise(pl.col('DAYS_FIRST_DRAWING')).alias('DAYS_FIRST_DRAWING'),
        pl.when(pl.col('DAYS_FIRST_DUE') == 365243).then(None).otherwise(pl.col('DAYS_FIRST_DUE')).alias('DAYS_FIRST_DUE'),
        pl.when(pl.col('DAYS_LAST_DUE_1ST_VERSION') == 365243).then(None).otherwise(pl.col('DAYS_LAST_DUE_1ST_VERSION')).alias('DAYS_LAST_DUE_1ST_VERSION'),
        pl.when(pl.col('DAYS_LAST_DUE') == 365243).then(None).otherwise(pl.col('DAYS_LAST_DUE')).alias('DAYS_LAST_DUE'),
        pl.when(pl.col('DAYS_TERMINATION') == 365243).then(None).otherwise(pl.col('DAYS_TERMINATION')).alias('DAYS_TERMINATION')
    ])

    # Subtraction between DAYS_LAST_DUE_1ST_VERSION and DAYS_LAST_DUE
    prev = prev.with_columns([
        (pl.col('DAYS_LAST_DUE_1ST_VERSION') - pl.col('DAYS_LAST_DUE')).alias('PREV_DAYS_LAST_DUE_DIFF')
    ])

    # Calculate the interest rate
    all_pay = pl.col('AMT_ANNUITY') * pl.col('CNT_PAYMENT')
    prev = prev.with_columns([
        ((all_pay / pl.col('AMT_CREDIT') - 1) / pl.col('CNT_PAYMENT')).alias('PREV_INTERESTS_RATE')
    ])

    return prev






    