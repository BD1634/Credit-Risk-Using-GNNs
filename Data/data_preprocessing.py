""" 
"""
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

    apps = apps.with_columns([
        pl.when(pl.col('APPS_EXT_SOURCE_STD').is_null())
        .then(pl.col('APPS_EXT_SOURCE_STD').mean())
        .otherwise(pl.col('APPS_EXT_SOURCE_STD'))
        .alias('APPS_EXT_SOURCE_STD')
    ])

    apps = apps.with_columns([
        (pl.col('AMT_ANNUITY') / pl.col('AMT_CREDIT')).alias('APPS_ANNUITY_CREDIT_RATIO'),
        (pl.col('AMT_GOODS_PRICE') / pl.col('AMT_CREDIT')).alias('APPS_GOODS_CREDIT_RATIO')
    ])


    apps = apps.with_columns([
        (pl.col('AMT_ANNUITY') / pl.col('AMT_INCOME_TOTAL')).alias('APPS_ANNUITY_INCOME_RATIO'),
        (pl.col('AMT_CREDIT') / pl.col('AMT_INCOME_TOTAL')).alias('APPS_CREDIT_INCOME_RATIO'),
        (pl.col('AMT_GOODS_PRICE') / pl.col('AMT_INCOME_TOTAL')).alias('APPS_GOODS_INCOME_RATIO'),
        (pl.col('AMT_INCOME_TOTAL') / pl.col('CNT_FAM_MEMBERS')).alias('APPS_CNT_FAM_INCOME_RATIO')
    ])

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


    prev = prev.with_columns([
        pl.when(pl.col('DAYS_FIRST_DRAWING') == 365243).then(None).otherwise(pl.col('DAYS_FIRST_DRAWING')).alias('DAYS_FIRST_DRAWING'),
        pl.when(pl.col('DAYS_FIRST_DUE') == 365243).then(None).otherwise(pl.col('DAYS_FIRST_DUE')).alias('DAYS_FIRST_DUE'),
        pl.when(pl.col('DAYS_LAST_DUE_1ST_VERSION') == 365243).then(None).otherwise(pl.col('DAYS_LAST_DUE_1ST_VERSION')).alias('DAYS_LAST_DUE_1ST_VERSION'),
        pl.when(pl.col('DAYS_LAST_DUE') == 365243).then(None).otherwise(pl.col('DAYS_LAST_DUE')).alias('DAYS_LAST_DUE'),
        pl.when(pl.col('DAYS_TERMINATION') == 365243).then(None).otherwise(pl.col('DAYS_TERMINATION')).alias('DAYS_TERMINATION')
    ])

    prev = prev.with_columns([
        (pl.col('DAYS_LAST_DUE_1ST_VERSION') - pl.col('DAYS_LAST_DUE')).alias('PREV_DAYS_LAST_DUE_DIFF')
    ])

    all_pay = pl.col('AMT_ANNUITY') * pl.col('CNT_PAYMENT')
    prev = prev.with_columns([
        ((all_pay / pl.col('AMT_CREDIT') - 1) / pl.col('CNT_PAYMENT')).alias('PREV_INTERESTS_RATE')
    ])

    return prev


def process_prev_amt_agg(prev):

    agg_expressions = [
    pl.col("SK_ID_CURR").count().alias("PREV_SK_ID_CURR_COUNT"),
    
    pl.col("AMT_CREDIT").mean().alias("PREV_AMT_CREDIT_MEAN"),
    pl.col("AMT_CREDIT").max().alias("PREV_AMT_CREDIT_MAX"),
    pl.col("AMT_CREDIT").sum().alias("PREV_AMT_CREDIT_SUM"),
    
    pl.col("AMT_ANNUITY").mean().alias("PREV_AMT_ANNUITY_MEAN"),
    pl.col("AMT_ANNUITY").max().alias("PREV_AMT_ANNUITY_MAX"),
    pl.col("AMT_ANNUITY").sum().alias("PREV_AMT_ANNUITY_SUM"),
    
    pl.col("AMT_APPLICATION").mean().alias("PREV_AMT_APPLICATION_MEAN"),
    pl.col("AMT_APPLICATION").max().alias("PREV_AMT_APPLICATION_MAX"),
    pl.col("AMT_APPLICATION").sum().alias("PREV_AMT_APPLICATION_SUM"),
    
    pl.col("AMT_DOWN_PAYMENT").mean().alias("PREV_AMT_DOWN_PAYMENT_MEAN"),
    pl.col("AMT_DOWN_PAYMENT").max().alias("PREV_AMT_DOWN_PAYMENT_MAX"),
    pl.col("AMT_DOWN_PAYMENT").sum().alias("PREV_AMT_DOWN_PAYMENT_SUM"),
    
    pl.col("AMT_GOODS_PRICE").mean().alias("PREV_AMT_GOODS_PRICE_MEAN"),
    pl.col("AMT_GOODS_PRICE").max().alias("PREV_AMT_GOODS_PRICE_MAX"),
    pl.col("AMT_GOODS_PRICE").sum().alias("PREV_AMT_GOODS_PRICE_SUM"),
    
    pl.col("RATE_DOWN_PAYMENT").min().alias("PREV_RATE_DOWN_PAYMENT_MIN"),
    pl.col("RATE_DOWN_PAYMENT").max().alias("PREV_RATE_DOWN_PAYMENT_MAX"),
    pl.col("RATE_DOWN_PAYMENT").mean().alias("PREV_RATE_DOWN_PAYMENT_MEAN"),
    
    pl.col("DAYS_DECISION").min().alias("PREV_DAYS_DECISION_MIN"),
    pl.col("DAYS_DECISION").max().alias("PREV_DAYS_DECISION_MAX"),
    pl.col("DAYS_DECISION").mean().alias("PREV_DAYS_DECISION_MEAN"),
    
    pl.col("CNT_PAYMENT").mean().alias("PREV_CNT_PAYMENT_MEAN"),
    pl.col("CNT_PAYMENT").sum().alias("PREV_CNT_PAYMENT_SUM"),
    
    pl.col("PREV_CREDIT_DIFF").mean().alias("PREV_PREV_CREDIT_DIFF_MEAN"),
    pl.col("PREV_CREDIT_DIFF").max().alias("PREV_PREV_CREDIT_DIFF_MAX"),
    pl.col("PREV_CREDIT_DIFF").sum().alias("PREV_PREV_CREDIT_DIFF_SUM"),
    
    pl.col("PREV_CREDIT_APPL_RATIO").mean().alias("PREV_PREV_CREDIT_APPL_RATIO_MEAN"),
    pl.col("PREV_CREDIT_APPL_RATIO").max().alias("PREV_PREV_CREDIT_APPL_RATIO_MAX"),
    
    pl.col("PREV_GOODS_DIFF").mean().alias("PREV_PREV_GOODS_DIFF_MEAN"),
    pl.col("PREV_GOODS_DIFF").max().alias("PREV_PREV_GOODS_DIFF_MAX"),
    pl.col("PREV_GOODS_DIFF").sum().alias("PREV_PREV_GOODS_DIFF_SUM"),
    
    pl.col("PREV_GOODS_APPL_RATIO").mean().alias("PREV_PREV_GOODS_APPL_RATIO_MEAN"),
    pl.col("PREV_GOODS_APPL_RATIO").max().alias("PREV_PREV_GOODS_APPL_RATIO_MAX"),
    
    pl.col("PREV_DAYS_LAST_DUE_DIFF").mean().alias("PREV_PREV_DAYS_LAST_DUE_DIFF_MEAN"),
    pl.col("PREV_DAYS_LAST_DUE_DIFF").max().alias("PREV_PREV_DAYS_LAST_DUE_DIFF_MAX"),
    pl.col("PREV_DAYS_LAST_DUE_DIFF").sum().alias("PREV_PREV_DAYS_LAST_DUE_DIFF_SUM"),
    
    pl.col("PREV_INTERESTS_RATE").mean().alias("PREV_PREV_INTERESTS_RATE_MEAN"),
    pl.col("PREV_INTERESTS_RATE").max().alias("PREV_PREV_INTERESTS_RATE_MAX")
    ]


    prev_amt_agg = prev.group_by("SK_ID_CURR").agg(agg_expressions)

    return prev_amt_agg


def process_prev_refused_appr_agg(prev):

    prev_refused = (prev.filter(
        pl.col("NAME_CONTRACT_STATUS")== "Approved" | pl.col("NAME_CONTRACT_STATUS")== "Refused" )
        ).group_by(['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).agg(
            pl.count('SK_ID_CURR').alias('count')).pivot(
            index='SK_ID_CURR',
            columns='NAME_CONTRACT_STATUS',
            values='count'
        )
    
    result = result.rename({
        'Approved': 'PREV_APPROVED_COUNT',
        'Refused': 'PREV_REFUSED_COUNT'
    })
 
    prev_refused_appr_agg = result.fill_null(0)

    return prev_refused_appr_agg


def process_prev_days365_agg(prev):
    
    cond_days365 = prev.filter(pl.col('DAYS_DECISION') > -365)
    prev_days365_agg = cond_days365.groupby('SK_ID_CURR').agg([
        pl.col('AMT_CREDIT').mean().alias('PREV_D365_AMT_CREDIT_MEAN'),
        pl.col('AMT_CREDIT').max().alias('PREV_D365_AMT_CREDIT_MAX'),
        pl.col('AMT_CREDIT').sum().alias('PREV_D365_AMT_CREDIT_SUM'),
        
        pl.col('AMT_ANNUITY').mean().alias('PREV_D365_AMT_ANNUITY_MEAN'),
        pl.col('AMT_ANNUITY').max().alias('PREV_D365_AMT_ANNUITY_MAX'),
        pl.col('AMT_ANNUITY').sum().alias('PREV_D365_AMT_ANNUITY_SUM'),
        
        pl.col('AMT_APPLICATION').mean().alias('PREV_D365_AMT_APPLICATION_MEAN'),
        pl.col('AMT_APPLICATION').max().alias('PREV_D365_AMT_APPLICATION_MAX'),
        pl.col('AMT_APPLICATION').sum().alias('PREV_D365_AMT_APPLICATION_SUM'),
        
        pl.col('AMT_DOWN_PAYMENT').mean().alias('PREV_D365_AMT_DOWN_PAYMENT_MEAN'),
        pl.col('AMT_DOWN_PAYMENT').max().alias('PREV_D365_AMT_DOWN_PAYMENT_MAX'),
        pl.col('AMT_DOWN_PAYMENT').sum().alias('PREV_D365_AMT_DOWN_PAYMENT_SUM'),
        
        pl.col('AMT_GOODS_PRICE').mean().alias('PREV_D365_AMT_GOODS_PRICE_MEAN'),
        pl.col('AMT_GOODS_PRICE').max().alias('PREV_D365_AMT_GOODS_PRICE_MAX'),
        pl.col('AMT_GOODS_PRICE').sum().alias('PREV_D365_AMT_GOODS_PRICE_SUM'),
        
        pl.col('RATE_DOWN_PAYMENT').min().alias('PREV_D365_RATE_DOWN_PAYMENT_MIN'),
        pl.col('RATE_DOWN_PAYMENT').max().alias('PREV_D365_RATE_DOWN_PAYMENT_MAX'),
        pl.col('RATE_DOWN_PAYMENT').mean().alias('PREV_D365_RATE_DOWN_PAYMENT_MEAN'),
        
        pl.col('DAYS_DECISION').min().alias('PREV_D365_DAYS_DECISION_MIN'),
        pl.col('DAYS_DECISION').max().alias('PREV_D365_DAYS_DECISION_MAX'),
        pl.col('DAYS_DECISION').mean().alias('PREV_D365_DAYS_DECISION_MEAN'),
        
        pl.col('CNT_PAYMENT').mean().alias('PREV_D365_CNT_PAYMENT_MEAN'),
        pl.col('CNT_PAYMENT').sum().alias('PREV_D365_CNT_PAYMENT_SUM'),
        
        pl.col('PREV_CREDIT_DIFF').mean().alias('PREV_D365_CREDIT_DIFF_MEAN'),
        pl.col('PREV_CREDIT_DIFF').max().alias('PREV_D365_CREDIT_DIFF_MAX'),
        pl.col('PREV_CREDIT_DIFF').sum().alias('PREV_D365_CREDIT_DIFF_SUM'),
        
        pl.col('PREV_CREDIT_APPL_RATIO').mean().alias('PREV_D365_CREDIT_APPL_RATIO_MEAN'),
        pl.col('PREV_CREDIT_APPL_RATIO').max().alias('PREV_D365_CREDIT_APPL_RATIO_MAX'),
        
        pl.col('PREV_GOODS_DIFF').mean().alias('PREV_D365_GOODS_DIFF_MEAN'),
        pl.col('PREV_GOODS_DIFF').max().alias('PREV_D365_GOODS_DIFF_MAX'),
        pl.col('PREV_GOODS_DIFF').sum().alias('PREV_D365_GOODS_DIFF_SUM'),
        
        pl.col('PREV_GOODS_APPL_RATIO').mean().alias('PREV_D365_GOODS_APPL_RATIO_MEAN'),
        pl.col('PREV_GOODS_APPL_RATIO').max().alias('PREV_D365_GOODS_APPL_RATIO_MAX'),
        
        pl.col('PREV_DAYS_LAST_DUE_DIFF').mean().alias('PREV_D365_DAYS_LAST_DUE_DIFF_MEAN'),
        pl.col('PREV_DAYS_LAST_DUE_DIFF').max().alias('PREV_D365_DAYS_LAST_DUE_DIFF_MAX'),
        pl.col('PREV_DAYS_LAST_DUE_DIFF').sum().alias('PREV_D365_DAYS_LAST_DUE_DIFF_SUM'),
        
        pl.col('PREV_INTERESTS_RATE').mean().alias('PREV_D365_INTERESTS_RATE_MEAN'),
        pl.col('PREV_INTERESTS_RATE').max().alias('PREV_D365_INTERESTS_RATE_MAX'),
    ])

    return prev_days365_agg


def process_prev_agg(prev):

    prev = process_prev_data(prev)  
    prev_amt_agg = process_prev_amt_agg(prev)
    prev_refused_appr_agg = process_prev_refused_appr_agg(prev)
    prev_days365_agg = process_prev_days365_agg(prev)
    prev_agg = prev_amt_agg.join(prev_refused_appr_agg, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.join(prev_days365_agg, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.with_columns([
        (pl.col('PREV_REFUSED_COUNT') / pl.col('PREV_SK_ID_CURR_COUNT')).alias('PREV_REFUSED_RATIO'),
        (pl.col('PREV_APPROVED_COUNT') / pl.col('PREV_SK_ID_CURR_COUNT')).alias('PREV_APPROVED_RATIO')
    ])
    prev_agg = prev_agg.drop(['PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT'])

    return prev_agg


def process_bureau_data(bureau):

    bureau = bureau.with_columns([
        (pl.col('DAYS_CREDIT_ENDDATE') - pl.col('DAYS_ENDDATE_FACT')).alias('BUREAU_ENDDATE_FACT_DIFF'),
        (pl.col('DAYS_CREDIT') - pl.col('DAYS_ENDDATE_FACT')).alias('BUREAU_CREDIT_FACT_DIFF'),
        (pl.col('DAYS_CREDIT') - pl.col('DAYS_CREDIT_ENDDATE')).alias('BUREAU_CREDIT_ENDDATE_DIFF'),
        (pl.col('AMT_CREDIT_SUM_DEBT') / pl.col('AMT_CREDIT_SUM')).alias('BUREAU_CREDIT_DEBT_RATIO'),
        (pl.col('AMT_CREDIT_SUM_DEBT') - pl.col('AMT_CREDIT_SUM')).alias('BUREAU_CREDIT_DEBT_DIFF')
    ])
    bureau = bureau.with_columns([
        pl.when(pl.col('CREDIT_DAY_OVERDUE') > 0).then(1).otherwise(0).alias('BUREAU_IS_DPD'),
        pl.when(pl.col('CREDIT_DAY_OVERDUE') > 120).then(1).otherwise(0).alias('BUREAU_IS_DPD_OVER120')
    ])

    return bureau

def process_bureau_day_amt_agg(bureau):

    bureau_agg_dict = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
        'BUREAU_ENDDATE_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_IS_DPD': ['mean', 'sum'],
        'BUREAU_IS_DPD_OVER120': ['mean', 'sum']
    }

    bureau_grp = bureau.groupby('SK_ID_CURR').agg([
    
        pl.col(col).min().alias(f'BUREAU_{col.upper()}_MIN') if 'min' in agg else None for col, agg in bureau_agg_dict.items()
    ] + [
        pl.col(col).max().alias(f'BUREAU_{col.upper()}_MAX') if 'max' in agg else None for col, agg in bureau_agg_dict.items()
    ] + [
        pl.col(col).mean().alias(f'BUREAU_{col.upper()}_MEAN') if 'mean' in agg else None for col, agg in bureau_agg_dict.items()
    ] + [
        pl.col(col).sum().alias(f'BUREAU_{col.upper()}_SUM') if 'sum' in agg else None for col, agg in bureau_agg_dict.items()
    ]).drop_none() 

    bureau_grp = bureau_grp.with_columns([pl.col(column).alias(column.upper()) for column in bureau_grp.columns])

    return bureau_grp

#####



def process_bureau_active_agg(bureau):

    bureau_active = bureau.filter(pl.col('CREDIT_ACTIVE') == 'Active')

    bureau_agg_dict = {
        'SK_ID_BUREAU': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
        'BUREAU_ENDDATE_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_IS_DPD': ['mean', 'sum'],
        'BUREAU_IS_DPD_OVER120': ['mean', 'sum']
    }


    bureau_active_agg = bureau_active.groupby('SK_ID_CURR').agg([
        pl.col(col).min().alias(f'BUREAU_ACT_{col.upper()}_MIN') for col in bureau_agg_dict if 'min' in bureau_agg_dict[col]
    ] + [
        pl.col(col).max().alias(f'BUREAU_ACT_{col.upper()}_MAX') for col in bureau_agg_dict if 'max' in bureau_agg_dict[col]
    ] + [
        pl.col(col).mean().alias(f'BUREAU_ACT_{col.upper()}_MEAN') for col in bureau_agg_dict if 'mean' in bureau_agg_dict[col]
    ] + [
        pl.col(col).sum().alias(f'BUREAU_ACT_{col.upper()}_SUM') for col in bureau_agg_dict if 'sum' in bureau_agg_dict[col]
    ])

    return bureau_active_agg


def process_bureau_days750_agg(bureau):

    bureau_days750 = bureau.filter(pl.col('DAYS_CREDIT') > -750)

    bureau_agg_dict = {
        'SK_ID_BUREAU': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
        'BUREAU_ENDDATE_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_IS_DPD': ['mean', 'sum'],
        'BUREAU_IS_DPD_OVER120': ['mean', 'sum']
    }


    bureau_days750_agg = bureau_days750.groupby('SK_ID_CURR').agg([
        pl.col(col).min().alias(f'BUREAU_DAYS750_{col.upper()}_MIN') for col in bureau_agg_dict if 'min' in bureau_agg_dict[col]
    ] + [
        pl.col(col).max().alias(f'BUREAU_DAYS750_{col.upper()}_MAX') for col in bureau_agg_dict if 'max' in bureau_agg_dict[col]
    ] + [
        pl.col(col).mean().alias(f'BUREAU_DAYS750_{col.upper()}_MEAN') for col in bureau_agg_dict if 'mean' in bureau_agg_dict[col]
    ] + [
        pl.col(col).sum().alias(f'BUREAU_DAYS750_{col.upper()}_SUM') for col in bureau_agg_dict if 'sum' in bureau_agg_dict[col]
    ])

    return bureau_days750_agg

def process_bureau_bal_agg(bureau, bureau_bal):

    bureau_bal = bureau_bal.join(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], on='SK_ID_BUREAU', how='left')


    bureau_bal = bureau_bal.with_columns([
        pl.when(pl.col('STATUS').is_in(['1', '2', '3', '4', '5'])).then(1).otherwise(0).alias('BUREAU_BAL_IS_DPD'),
        pl.when(pl.col('STATUS') == '5').then(1).otherwise(0).alias('BUREAU_BAL_IS_DPD_OVER120')
    ])


    bureau_bal_grp = bureau_bal.groupby('SK_ID_CURR').agg([
        pl.col('SK_ID_CURR').count().alias('BUREAU_BAL_SK_ID_CURR_COUNT'),
        pl.col('MONTHS_BALANCE').min().alias('BUREAU_BAL_MONTHS_BALANCE_MIN'),
        pl.col('MONTHS_BALANCE').max().alias('BUREAU_BAL_MONTHS_BALANCE_MAX'),
        pl.col('MONTHS_BALANCE').mean().alias('BUREAU_BAL_MONTHS_BALANCE_MEAN'),
        pl.col('BUREAU_BAL_IS_DPD').mean().alias('BUREAU_BAL_BUREAU_BAL_IS_DPD_MEAN'),
        pl.col('BUREAU_BAL_IS_DPD').sum().alias('BUREAU_BAL_BUREAU_BAL_IS_DPD_SUM'),
        pl.col('BUREAU_BAL_IS_DPD_OVER120').mean().alias('BUREAU_BAL_BUREAU_BAL_IS_DPD_OVER120_MEAN'),
        pl.col('BUREAU_BAL_IS_DPD_OVER120').sum().alias('BUREAU_BAL_BUREAU_BAL_IS_DPD_OVER120_SUM')
    ])

    return bureau_bal_grp


def process_bureau_agg(bureau, bureau_bal):
  
    bureau = process_bureau_data(bureau)
    bureau_day_amt_agg = process_bureau_day_amt_agg(bureau)
    bureau_active_agg = process_bureau_active_agg(bureau)
    bureau_days750_agg = process_bureau_days750_agg(bureau)
    bureau_bal_agg = process_bureau_bal_agg(bureau, bureau_bal)

   
    bureau_agg = bureau_day_amt_agg.join(bureau_active_agg, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.join(bureau_bal_agg, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.join(bureau_days750_agg, on='SK_ID_CURR', how='left')

    bureau_agg = bureau_agg.with_columns([
        (pl.col('BUREAU_ACT_BUREAU_IS_DPD_SUM') / pl.col('BUREAU_SK_ID_BUREAU_COUNT')).alias('BUREAU_ACT_IS_DPD_RATIO'),
        (pl.col('BUREAU_ACT_BUREAU_IS_DPD_OVER120_SUM') / pl.col('BUREAU_SK_ID_BUREAU_COUNT')).alias('BUREAU_ACT_IS_DPD_OVER120_RATIO')
    ])

    return bureau_agg



def process_pos_bal_agg(pos_bal):
   
    pos_bal = pos_bal.with_columns([
        pl.when(pl.col('SK_DPD') > 0).then(1).otherwise(0).alias('POS_IS_DPD'),
        pl.when((pl.col('SK_DPD') > 0) & (pl.col('SK_DPD') < 120)).then(1).otherwise(0).alias('POS_IS_DPD_UNDER_120'),
        pl.when(pl.col('SK_DPD') >= 120).then(1).otherwise(0).alias('POS_IS_DPD_OVER_120')
    ])

    pos_bal_agg = pos_bal.groupby('SK_ID_CURR').agg([
        pl.col('MONTHS_BALANCE').min().alias('POS_MONTHS_BALANCE_MIN'),
        pl.col('MONTHS_BALANCE').mean().alias('POS_MONTHS_BALANCE_MEAN'),
        pl.col('MONTHS_BALANCE').max().alias('POS_MONTHS_BALANCE_MAX'),
        pl.col('SK_DPD').min().alias('POS_SK_DPD_MIN'),
        pl.col('SK_DPD').max().alias('POS_SK_DPD_MAX'),
        pl.col('SK_DPD').mean().alias('POS_SK_DPD_MEAN'),
        pl.col('SK_DPD').sum().alias('POS_SK_DPD_SUM'),
        pl.col('CNT_INSTALMENT').min().alias('POS_CNT_INSTALMENT_MIN'),
        pl.col('CNT_INSTALMENT').max().alias('POS_CNT_INSTALMENT_MAX'),
        pl.col('CNT_INSTALMENT').mean().alias('POS_CNT_INSTALMENT_MEAN'),
        pl.col('CNT_INSTALMENT').sum().alias('POS_CNT_INSTALMENT_SUM'),
        pl.col('CNT_INSTALMENT_FUTURE').min().alias('POS_CNT_INSTALMENT_FUTURE_MIN'),
        pl.col('CNT_INSTALMENT_FUTURE').max().alias('POS_CNT_INSTALMENT_FUTURE_MAX'),
        pl.col('CNT_INSTALMENT_FUTURE').mean().alias('POS_CNT_INSTALMENT_FUTURE_MEAN'),
        pl.col('CNT_INSTALMENT_FUTURE').sum().alias('POS_CNT_INSTALMENT_FUTURE_SUM'),
        pl.col('POS_IS_DPD').mean().alias('POS_POS_IS_DPD_MEAN'),
        pl.col('POS_IS_DPD').sum().alias('POS_POS_IS_DPD_SUM'),
        pl.col('POS_IS_DPD_UNDER_120').mean().alias('POS_POS_IS_DPD_UNDER_120_MEAN'),
        pl.col('POS_IS_DPD_UNDER_120').sum().alias('POS_POS_IS_DPD_UNDER_120_SUM'),
        pl.col('POS_IS_DPD_OVER_120').mean().alias('POS_POS_IS_DPD_OVER_120_MEAN'),
        pl.col('POS_IS_DPD_OVER_120').sum().alias('POS_POS_IS_DPD_OVER_120_SUM')
    ])

    return pos_bal_agg

def process_install_agg(install):

    install = install.with_columns([
        (pl.col('AMT_INSTALMENT') - pl.col('AMT_PAYMENT')).alias('AMT_DIFF'),
        ((pl.col('AMT_PAYMENT') + 1) / (pl.col('AMT_INSTALMENT') + 1)).alias('AMT_RATIO'),
        (pl.col('DAYS_ENTRY_PAYMENT') - pl.col('DAYS_INSTALMENT')).alias('SK_DPD'),
        pl.when(pl.col('SK_DPD') > 0).then(1).otherwise(0).alias('INS_IS_DPD'),
        pl.when((pl.col('SK_DPD') > 0) & (pl.col('SK_DPD') < 120)).then(1).otherwise(0).alias('INS_IS_DPD_UNDER_120'),
        pl.when(pl.col('SK_DPD') >= 120).then(1).otherwise(0).alias('INS_IS_DPD_OVER_120')
    ])

    install_agg = install.groupby('SK_ID_CURR').agg([
        pl.col('NUM_INSTALMENT_VERSION').n_unique().alias('INS_NUM_INSTALMENT_VERSION_NUNIQUE'),
        pl.col('DAYS_ENTRY_PAYMENT').mean().alias('INS_DAYS_ENTRY_PAYMENT_MEAN'),
        pl.col('DAYS_ENTRY_PAYMENT').max().alias('INS_DAYS_ENTRY_PAYMENT_MAX'),
        pl.col('DAYS_ENTRY_PAYMENT').sum().alias('INS_DAYS_ENTRY_PAYMENT_SUM'),
        pl.col('DAYS_INSTALMENT').mean().alias('INS_DAYS_INSTALMENT_MEAN'),
        pl.col('DAYS_INSTALMENT').max().alias('INS_DAYS_INSTALMENT_MAX'),
        pl.col('DAYS_INSTALMENT').sum().alias('INS_DAYS_INSTALMENT_SUM'),
        pl.col('AMT_INSTALMENT').mean().alias('INS_AMT_INSTALMENT_MEAN'),
        pl.col('AMT_INSTALMENT').max().alias('INS_AMT_INSTALMENT_MAX'),
        pl.col('AMT_INSTALMENT').sum().alias('INS_AMT_INSTALMENT_SUM'),
        pl.col('AMT_PAYMENT').mean().alias('INS_AMT_PAYMENT_MEAN'),
        pl.col('AMT_PAYMENT').max().alias('INS_AMT_PAYMENT_MAX'),
        pl.col('AMT_PAYMENT').sum().alias('INS_AMT_PAYMENT_SUM'),
        pl.col('AMT_DIFF').mean().alias('INS_AMT_DIFF_MEAN'),
        pl.col('AMT_DIFF').min().alias('INS_AMT_DIFF_MIN'),
        pl.col('AMT_DIFF').max().alias('INS_AMT_DIFF_MAX'),
        pl.col('AMT_DIFF').sum().alias('INS_AMT_DIFF_SUM'),
        pl.col('AMT_RATIO').mean().alias('INS_AMT_RATIO_MEAN'),
        pl.col('AMT_RATIO').max().alias('INS_AMT_RATIO_MAX'),
        pl.col('SK_DPD').mean().alias('INS_SK_DPD_MEAN'),
        pl.col('SK_DPD').min().alias('INS_SK_DPD_MIN'),
        pl.col('SK_DPD').max().alias('INS_SK_DPD_MAX'),
        pl.col('INS_IS_DPD').mean().alias('INS_INS_IS_DPD_MEAN'),
        pl.col('INS_IS_DPD').sum().alias('INS_INS_IS_DPD_SUM'),
        pl.col('INS_IS_DPD_UNDER_120').mean().alias('INS_INS_IS_DPD_UNDER_120_MEAN'),
        pl.col('INS_IS_DPD_UNDER_120').sum().alias('INS_INS_IS_DPD_UNDER_120_SUM'),
        pl.col('INS_IS_DPD_OVER_120').mean().alias('INS_INS_IS_DPD_OVER_120_MEAN'),
        pl.col('INS_IS_DPD_OVER_120').sum().alias('INS_INS_IS_DPD_OVER_120_SUM')
    ])

    return install_agg

def process_card_bal_agg(card_bal):

    card_bal = card_bal.with_columns([
        (pl.col('AMT_BALANCE') / pl.col('AMT_CREDIT_LIMIT_ACTUAL')).alias('BALANCE_LIMIT_RATIO'),
        (pl.col('AMT_DRAWINGS_CURRENT') / pl.col('AMT_CREDIT_LIMIT_ACTUAL')).alias('DRAWING_LIMIT_RATIO'),
        pl.when(pl.col('SK_DPD') > 0).then(1).otherwise(0).alias('CARD_IS_DPD'),
        pl.when((pl.col('SK_DPD') > 0) & (pl.col('SK_DPD') < 120)).then(1).otherwise(0).alias('CARD_IS_DPD_UNDER_120'),
        pl.when(pl.col('SK_DPD') >= 120).then(1).otherwise(0).alias('CARD_IS_DPD_OVER_120')
    ])
    card_bal_agg = card_bal.groupby('SK_ID_CURR').agg([
        pl.col('AMT_BALANCE').max().alias('CARD_AMT_BALANCE_MAX'),
        pl.col('AMT_CREDIT_LIMIT_ACTUAL').max().alias('CARD_AMT_CREDIT_LIMIT_ACTUAL_MAX'),
        pl.col('AMT_DRAWINGS_ATM_CURRENT').max().alias('CARD_AMT_DRAWINGS_ATM_CURRENT_MAX'),
        pl.col('AMT_DRAWINGS_CURRENT').max().alias('CARD_AMT_DRAWINGS_CURRENT_MAX'),
        pl.col('AMT_DRAWINGS_POS_CURRENT').max().alias('CARD_AMT_DRAWINGS_POS_CURRENT_MAX'),
        pl.col('AMT_INST_MIN_REGULARITY').max().alias('CARD_AMT_INST_MIN_REGULARITY_MAX'),
        pl.col('AMT_PAYMENT_TOTAL_CURRENT').max().alias('CARD_AMT_PAYMENT_TOTAL_CURRENT_MAX'),
        pl.col('AMT_TOTAL_RECEIVABLE').max().alias('CARD_AMT_TOTAL_RECEIVABLE_MAX'),
        pl.col('CNT_DRAWINGS_ATM_CURRENT').max().alias('CARD_CNT_DRAWINGS_ATM_CURRENT_MAX'),
        pl.col('CNT_DRAWINGS_CURRENT').max().alias('CARD_CNT_DRAWINGS_CURRENT_MAX'),
        pl.col('CNT_DRAWINGS_POS_CURRENT').max().alias('CARD_CNT_DRAWINGS_POS_CURRENT_MAX'),
        pl.col('SK_DPD').mean().alias('CARD_SK_DPD_MEAN'),
        pl.col('BALANCE_LIMIT_RATIO').min().alias('CARD_BALANCE_LIMIT_RATIO_MIN'),
        pl.col('BALANCE_LIMIT_RATIO').max().alias('CARD_BALANCE_LIMIT_RATIO_MAX'),
        pl.col('DRAWING_LIMIT_RATIO').min().alias('CARD_DRAWING_LIMIT_RATIO_MIN'),
        pl.col('DRAWING_LIMIT_RATIO').max().alias('CARD_DRAWING_LIMIT_RATIO_MAX'),
        pl.col('CARD_IS_DPD').mean().alias('CARD_CARD_IS_DPD_MEAN'),
        pl.col('CARD_IS_DPD').sum().alias('CARD_CARD_IS_DPD_SUM'),
        pl.col('CARD_IS_DPD_UNDER_120').mean().alias('CARD_CARD_IS_DPD_UNDER_120_MEAN'),
        pl.col('CARD_IS_DPD_UNDER_120').sum().alias('CARD_CARD_IS_DPD_UNDER_120_SUM'),
        pl.col('CARD_IS_DPD_OVER_120').mean().alias('CARD_CARD_IS_DPD_OVER_120_MEAN'),
        pl.col('CARD_IS_DPD_OVER_120').sum().alias('CARD_CARD_IS_DPD_OVER_120_SUM')
    ])

    return card_bal_agg

def process_apps_all_with_all_agg(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal):

    apps_all = process_app_data(apps)
    prev_agg = process_prev_agg(prev)
    bureau_agg = process_bureau_agg(bureau, bureau_bal)
    pos_bal_agg = process_pos_bal_agg(pos_bal)
    install_agg = process_install_agg(install)
    card_bal_agg = process_card_bal_agg(card_bal)

    print('prev_agg shape:', prev_agg.shape, 'bureau_agg shape:', bureau_agg.shape)
    print('pos_bal_agg shape:', pos_bal_agg.shape, 'install_agg shape:', install_agg.shape, 'card_bal_agg shape:', card_bal_agg.shape)
    print('apps_all before join shape:', apps_all.shape)

    apps_all = apps_all.join(prev_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.join(bureau_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.join(pos_bal_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.join(install_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.join(card_bal_agg, on='SK_ID_CURR', how='left')

    print('apps_all after join with all shape:', apps_all.shape)

    return apps_all

def get_apps_all_encoded(apps_all):
    object_columns = [col for col, dtype in apps_all.dtypes.items() if dtype == pl.Object]
    for column in object_columns:
        apps_all = apps_all.with_columns([
            pl.col(column).cast(pl.Categorical).alias(column)
        ])
    
    return apps_all


def feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal):

    apps_all = process_apps_all_with_all_agg(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)

    apps_all = get_apps_all_encoded(apps_all)
    
    return apps_all
def main():
    PATH = 'insert_the_directory_here'
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = load_data(PATH)
    all_data = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    all_data = all_data.collect()


if __name__ == '__main__':
    main()






















    
  










    
