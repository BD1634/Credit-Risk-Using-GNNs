# data_preprocessing_polars.py
import os
import warnings
import numpy as np
import polars as pl

# Suppress specific numpy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")

raw_data_dir = './dataset/home-credit-default-risk/'

def get_balance_data():
    # Define schemas for each dataset
    pos_schema = {
        'SK_ID_PREV': pl.UInt32,
        'SK_ID_CURR': pl.UInt32,
        'MONTHS_BALANCE': pl.Int32,
        'SK_DPD': pl.Int32,
        'SK_DPD_DEF': pl.Int32,
        'CNT_INSTALMENT': pl.Float32,
        'CNT_INSTALMENT_FUTURE': pl.Float32
    }

    install_schema = {
        'SK_ID_PREV': pl.UInt32,
        'SK_ID_CURR': pl.UInt32,
        'NUM_INSTALMENT_NUMBER': pl.Int32,
        'NUM_INSTALMENT_VERSION': pl.Float32,
        'DAYS_INSTALMENT': pl.Float32,
        'DAYS_ENTRY_PAYMENT': pl.Float32,
        'AMT_INSTALMENT': pl.Float32,
        'AMT_PAYMENT': pl.Float32
    }

    card_schema = {
        'SK_ID_PREV': pl.UInt32,
        'SK_ID_CURR': pl.UInt32,
        'MONTHS_BALANCE': pl.Int16,
        'AMT_CREDIT_LIMIT_ACTUAL': pl.Int32,
        'CNT_DRAWINGS_CURRENT': pl.Int32,
        'SK_DPD': pl.Int32,
        'SK_DPD_DEF': pl.Int32,
        'AMT_BALANCE': pl.Float32,
        'AMT_DRAWINGS_ATM_CURRENT': pl.Float32,
        'AMT_DRAWINGS_CURRENT': pl.Float32,
        'AMT_DRAWINGS_OTHER_CURRENT': pl.Float32,
        'AMT_DRAWINGS_POS_CURRENT': pl.Float32,
        'AMT_INST_MIN_REGULARITY': pl.Float32,
        'AMT_PAYMENT_CURRENT': pl.Float32,
        'AMT_PAYMENT_TOTAL_CURRENT': pl.Float32,
        'AMT_RECEIVABLE_PRINCIPAL': pl.Float32,
        'AMT_RECIVABLE': pl.Float32,
        'AMT_TOTAL_RECEIVABLE': pl.Float32,
        'CNT_DRAWINGS_ATM_CURRENT': pl.Float32,
        'CNT_DRAWINGS_OTHER_CURRENT': pl.Float32,
        'CNT_DRAWINGS_POS_CURRENT': pl.Float32,
        'CNT_INSTALMENT_MATURE_CUM': pl.Float32
    }

    pos_bal = pl.read_csv(os.path.join(raw_data_dir, 'POS_CASH_balance.csv'), dtypes=pos_schema)
    install = pl.read_csv(os.path.join(raw_data_dir, 'installments_payments.csv'), dtypes=install_schema)
    card_bal = pl.read_csv(os.path.join(raw_data_dir, 'credit_card_balance.csv'), dtypes=card_schema)

    return pos_bal, install, card_bal


def data_loaded():
    pos_bal, install, card_bal = get_balance_data()
    app_train = pl.read_csv(os.path.join(raw_data_dir, 'application_train.csv'), encoding='utf8')
    apps = app_train
    prev = pl.read_csv(os.path.join(raw_data_dir, 'previous_application.csv'))
    bureau = pl.read_csv(os.path.join(raw_data_dir, 'bureau.csv'))
    bureau_bal = pl.read_csv(os.path.join(raw_data_dir, 'bureau_balance.csv'))
    return apps, prev, bureau, bureau_bal, pos_bal, install, card_bal


def get_apps_processed(apps):
    """
    feature engineering for apps
    """
    # 1. EXT_SOURCE_X FEATURE
    # Create a custom function to calculate row-wise operations in Polars
    def row_stats(df):
        # For mean calculation
        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        values = df.select(ext_cols).to_numpy()
        
        # Calculate mean and std with numpy, handling NaN values
        # Add extra handling for rows with all NaN values
        means = np.nanmean(values, axis=1)
        stds = np.nanstd(values, axis=1)
        
        # Replace NaN results with 0 to avoid warnings
        means = np.nan_to_num(means, nan=0)
        stds = np.nan_to_num(stds, nan=0)
        
        # Add new columns to the dataframe
        return df.with_columns([
            pl.Series("APPS_EXT_SOURCE_MEAN", means),
            pl.Series("APPS_EXT_SOURCE_STD", stds)
        ])
    
    # Apply the function
    apps = row_stats(apps)
    
    # Fill NaN in STD with mean
    std_mean = apps.select(pl.col("APPS_EXT_SOURCE_STD").mean()).item()
    apps = apps.with_columns(
        pl.col("APPS_EXT_SOURCE_STD").fill_null(std_mean)
    )

    # AMT_CREDIT ratios
    apps = apps.with_columns([
        (pl.col("AMT_ANNUITY") / pl.col("AMT_CREDIT")).alias("APPS_ANNUITY_CREDIT_RATIO"),
        (pl.col("AMT_GOODS_PRICE") / pl.col("AMT_CREDIT")).alias("APPS_GOODS_CREDIT_RATIO")
    ])

    # AMT_INCOME_TOTAL ratios
    apps = apps.with_columns([
        (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias("APPS_ANNUITY_INCOME_RATIO"),
        (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL")).alias("APPS_CREDIT_INCOME_RATIO"),
        (pl.col("AMT_GOODS_PRICE") / pl.col("AMT_INCOME_TOTAL")).alias("APPS_GOODS_INCOME_RATIO"),
        (pl.col("AMT_INCOME_TOTAL") / pl.col("CNT_FAM_MEMBERS")).alias("APPS_CNT_FAM_INCOME_RATIO")
    ])

    # DAYS_BIRTH, DAYS_EMPLOYED ratios
    apps = apps.with_columns([
        (pl.col("DAYS_EMPLOYED") / pl.col("DAYS_BIRTH")).alias("APPS_EMPLOYED_BIRTH_RATIO"),
        (pl.col("AMT_INCOME_TOTAL") / pl.col("DAYS_EMPLOYED")).alias("APPS_INCOME_EMPLOYED_RATIO"),
        (pl.col("AMT_INCOME_TOTAL") / pl.col("DAYS_BIRTH")).alias("APPS_INCOME_BIRTH_RATIO"),
        (pl.col("OWN_CAR_AGE") / pl.col("DAYS_BIRTH")).alias("APPS_CAR_BIRTH_RATIO"),
        (pl.col("OWN_CAR_AGE") / pl.col("DAYS_EMPLOYED")).alias("APPS_CAR_EMPLOYED_RATIO")
    ])

    return apps


def get_prev_processed(prev):
    prev = prev.with_columns([
        (pl.col("AMT_APPLICATION") - pl.col("AMT_CREDIT")).alias("PREV_CREDIT_DIFF"),
        (pl.col("AMT_APPLICATION") - pl.col("AMT_GOODS_PRICE")).alias("PREV_GOODS_DIFF"),
        (pl.col("AMT_CREDIT") / pl.col("AMT_APPLICATION")).alias("PREV_CREDIT_APPL_RATIO"),
        (pl.col("AMT_GOODS_PRICE") / pl.col("AMT_APPLICATION")).alias("PREV_GOODS_APPL_RATIO")
    ])

    # Replace 365243 with NaN in date columns
    prev = prev.with_columns([
        pl.col("DAYS_FIRST_DRAWING").replace(365243, None),
        pl.col("DAYS_FIRST_DUE").replace(365243, None),
        pl.col("DAYS_LAST_DUE_1ST_VERSION").replace(365243, None),
        pl.col("DAYS_LAST_DUE").replace(365243, None),
        pl.col("DAYS_TERMINATION").replace(365243, None)
    ])

    prev = prev.with_columns([
        (pl.col("DAYS_LAST_DUE_1ST_VERSION") - pl.col("DAYS_LAST_DUE")).alias("PREV_DAYS_LAST_DUE_DIFF"),
        ((pl.col("AMT_ANNUITY") * pl.col("CNT_PAYMENT") / pl.col("AMT_CREDIT") - 1) / pl.col("CNT_PAYMENT")).alias("PREV_INTERESTS_RATE")
    ])

    return prev


def get_prev_amt_agg(prev):
    """
    feature engineering for the previous credit application
    """
    agg_expr = [
        pl.count("SK_ID_CURR").alias("PREV_SK_ID_CURR_COUNT"),
        
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
        pl.col("PREV_INTERESTS_RATE").max().alias("PREV_PREV_INTERESTS_RATE_MAX"),
    ]
    
    prev_amt_agg = prev.group_by("SK_ID_CURR").agg(agg_expr)
    
    return prev_amt_agg


def get_prev_refused_appr_agg(prev):
    """
    PREV_APPROVED_COUNT : Credit application approved count
    PREV_REFUSED_COUNT :  Credit application refused count
    """
    # Filter for approved and refused applications
    filtered_prev = prev.filter(pl.col("NAME_CONTRACT_STATUS").is_in(["Approved", "Refused"]))
    
    # Count by SK_ID_CURR and NAME_CONTRACT_STATUS
    status_counts = filtered_prev.group_by(["SK_ID_CURR", "NAME_CONTRACT_STATUS"]).agg(
        pl.count().alias("count")
    )
    
    # Pivot to get columns for each status
    pivot_result = status_counts.pivot(
        values="count", 
        index="SK_ID_CURR", 
        columns="NAME_CONTRACT_STATUS"
    ).fill_null(0)
    
    # Rename columns
    prev_refused_appr_agg = pivot_result.rename({
        "Approved": "PREV_APPROVED_COUNT",
        "Refused": "PREV_REFUSED_COUNT"
    })
    
    # Ensure both columns exist
    if "PREV_APPROVED_COUNT" not in prev_refused_appr_agg.columns:
        prev_refused_appr_agg = prev_refused_appr_agg.with_columns(
            pl.lit(0).alias("PREV_APPROVED_COUNT")
        )
    if "PREV_REFUSED_COUNT" not in prev_refused_appr_agg.columns:
        prev_refused_appr_agg = prev_refused_appr_agg.with_columns(
            pl.lit(0).alias("PREV_REFUSED_COUNT")
        )
    
    return prev_refused_appr_agg


def get_prev_days365_agg(prev):
    """
    DAYS_DESCISION means How many days have been take since the previous credit application made.
    Somehow this feature is important.
    """
    # Filter for recent applications (within last year)
    prev_days365 = prev.filter(pl.col("DAYS_DECISION") > -365)
    
    agg_expr = [
        pl.count("SK_ID_CURR").alias("PREV_D365_SK_ID_CURR_COUNT"),
        
        pl.col("AMT_CREDIT").mean().alias("PREV_D365_AMT_CREDIT_MEAN"),
        pl.col("AMT_CREDIT").max().alias("PREV_D365_AMT_CREDIT_MAX"),
        pl.col("AMT_CREDIT").sum().alias("PREV_D365_AMT_CREDIT_SUM"),
        
        pl.col("AMT_ANNUITY").mean().alias("PREV_D365_AMT_ANNUITY_MEAN"),
        pl.col("AMT_ANNUITY").max().alias("PREV_D365_AMT_ANNUITY_MAX"),
        pl.col("AMT_ANNUITY").sum().alias("PREV_D365_AMT_ANNUITY_SUM"),
        
        pl.col("AMT_APPLICATION").mean().alias("PREV_D365_AMT_APPLICATION_MEAN"),
        pl.col("AMT_APPLICATION").max().alias("PREV_D365_AMT_APPLICATION_MAX"),
        pl.col("AMT_APPLICATION").sum().alias("PREV_D365_AMT_APPLICATION_SUM"),
        
        pl.col("AMT_DOWN_PAYMENT").mean().alias("PREV_D365_AMT_DOWN_PAYMENT_MEAN"),
        pl.col("AMT_DOWN_PAYMENT").max().alias("PREV_D365_AMT_DOWN_PAYMENT_MAX"),
        pl.col("AMT_DOWN_PAYMENT").sum().alias("PREV_D365_AMT_DOWN_PAYMENT_SUM"),
        
        pl.col("AMT_GOODS_PRICE").mean().alias("PREV_D365_AMT_GOODS_PRICE_MEAN"),
        pl.col("AMT_GOODS_PRICE").max().alias("PREV_D365_AMT_GOODS_PRICE_MAX"),
        pl.col("AMT_GOODS_PRICE").sum().alias("PREV_D365_AMT_GOODS_PRICE_SUM"),
        
        pl.col("RATE_DOWN_PAYMENT").min().alias("PREV_D365_RATE_DOWN_PAYMENT_MIN"),
        pl.col("RATE_DOWN_PAYMENT").max().alias("PREV_D365_RATE_DOWN_PAYMENT_MAX"),
        pl.col("RATE_DOWN_PAYMENT").mean().alias("PREV_D365_RATE_DOWN_PAYMENT_MEAN"),
        
        pl.col("DAYS_DECISION").min().alias("PREV_D365_DAYS_DECISION_MIN"),
        pl.col("DAYS_DECISION").max().alias("PREV_D365_DAYS_DECISION_MAX"),
        pl.col("DAYS_DECISION").mean().alias("PREV_D365_DAYS_DECISION_MEAN"),
        
        pl.col("CNT_PAYMENT").mean().alias("PREV_D365_CNT_PAYMENT_MEAN"),
        pl.col("CNT_PAYMENT").sum().alias("PREV_D365_CNT_PAYMENT_SUM"),
        
        pl.col("PREV_CREDIT_DIFF").mean().alias("PREV_D365_PREV_CREDIT_DIFF_MEAN"),
        pl.col("PREV_CREDIT_DIFF").max().alias("PREV_D365_PREV_CREDIT_DIFF_MAX"),
        pl.col("PREV_CREDIT_DIFF").sum().alias("PREV_D365_PREV_CREDIT_DIFF_SUM"),
        
        pl.col("PREV_CREDIT_APPL_RATIO").mean().alias("PREV_D365_PREV_CREDIT_APPL_RATIO_MEAN"),
        pl.col("PREV_CREDIT_APPL_RATIO").max().alias("PREV_D365_PREV_CREDIT_APPL_RATIO_MAX"),
        
        pl.col("PREV_GOODS_DIFF").mean().alias("PREV_D365_PREV_GOODS_DIFF_MEAN"),
        pl.col("PREV_GOODS_DIFF").max().alias("PREV_D365_PREV_GOODS_DIFF_MAX"),
        pl.col("PREV_GOODS_DIFF").sum().alias("PREV_D365_PREV_GOODS_DIFF_SUM"),
        
        pl.col("PREV_GOODS_APPL_RATIO").mean().alias("PREV_D365_PREV_GOODS_APPL_RATIO_MEAN"),
        pl.col("PREV_GOODS_APPL_RATIO").max().alias("PREV_D365_PREV_GOODS_APPL_RATIO_MAX"),
        
        pl.col("PREV_DAYS_LAST_DUE_DIFF").mean().alias("PREV_D365_PREV_DAYS_LAST_DUE_DIFF_MEAN"),
        pl.col("PREV_DAYS_LAST_DUE_DIFF").max().alias("PREV_D365_PREV_DAYS_LAST_DUE_DIFF_MAX"),
        pl.col("PREV_DAYS_LAST_DUE_DIFF").sum().alias("PREV_D365_PREV_DAYS_LAST_DUE_DIFF_SUM"),
        
        pl.col("PREV_INTERESTS_RATE").mean().alias("PREV_D365_PREV_INTERESTS_RATE_MEAN"),
        pl.col("PREV_INTERESTS_RATE").max().alias("PREV_D365_PREV_INTERESTS_RATE_MAX"),
    ]
    
    prev_days365_agg = prev_days365.group_by("SK_ID_CURR").agg(agg_expr)
    
    return prev_days365_agg


def get_prev_agg(prev):
    prev = get_prev_processed(prev)
    prev_amt_agg = get_prev_amt_agg(prev)
    prev_refused_appr_agg = get_prev_refused_appr_agg(prev)
    prev_days365_agg = get_prev_days365_agg(prev)

    # Merge aggregations
    prev_agg = prev_amt_agg.join(prev_refused_appr_agg, on="SK_ID_CURR", how="left")
    prev_agg = prev_agg.join(prev_days365_agg, on="SK_ID_CURR", how="left")
    
    # Calculate ratios
    prev_agg = prev_agg.with_columns([
        (pl.col("PREV_REFUSED_COUNT") / pl.col("PREV_SK_ID_CURR_COUNT")).alias("PREV_REFUSED_RATIO"),
        (pl.col("PREV_APPROVED_COUNT") / pl.col("PREV_SK_ID_CURR_COUNT")).alias("PREV_APPROVED_RATIO")
    ])
    
    # Drop original count columns
    prev_agg = prev_agg.drop(["PREV_REFUSED_COUNT", "PREV_APPROVED_COUNT"])
    
    return prev_agg


def get_bureau_processed(bureau):
    bureau = bureau.with_columns([
        (pl.col("DAYS_CREDIT_ENDDATE") - pl.col("DAYS_ENDDATE_FACT")).alias("BUREAU_ENDDATE_FACT_DIFF"),
        (pl.col("DAYS_CREDIT") - pl.col("DAYS_ENDDATE_FACT")).alias("BUREAU_CREDIT_FACT_DIFF"),
        (pl.col("DAYS_CREDIT") - pl.col("DAYS_CREDIT_ENDDATE")).alias("BUREAU_CREDIT_ENDDATE_DIFF"),
        
        (pl.col("AMT_CREDIT_SUM_DEBT") / pl.col("AMT_CREDIT_SUM")).alias("BUREAU_CREDIT_DEBT_RATIO"),
        (pl.col("AMT_CREDIT_SUM_DEBT") - pl.col("AMT_CREDIT_SUM")).alias("BUREAU_CREDIT_DEBT_DIFF"),
        
        (pl.col("CREDIT_DAY_OVERDUE") > 0).cast(pl.Int32).alias("BUREAU_IS_DPD"),
        (pl.col("CREDIT_DAY_OVERDUE") > 120).cast(pl.Int32).alias("BUREAU_IS_DPD_OVER120")
    ])
    
    return bureau


def get_bureau_day_amt_agg(bureau):
    agg_expr = [
        pl.count("SK_ID_BUREAU").alias("BUREAU_SK_ID_BUREAU_COUNT"),
        
        pl.col("DAYS_CREDIT").min().alias("BUREAU_DAYS_CREDIT_MIN"),
        pl.col("DAYS_CREDIT").max().alias("BUREAU_DAYS_CREDIT_MAX"),
        pl.col("DAYS_CREDIT").mean().alias("BUREAU_DAYS_CREDIT_MEAN"),
        
        pl.col("CREDIT_DAY_OVERDUE").min().alias("BUREAU_CREDIT_DAY_OVERDUE_MIN"),
        pl.col("CREDIT_DAY_OVERDUE").max().alias("BUREAU_CREDIT_DAY_OVERDUE_MAX"),
        pl.col("CREDIT_DAY_OVERDUE").mean().alias("BUREAU_CREDIT_DAY_OVERDUE_MEAN"),
        
        pl.col("DAYS_CREDIT_ENDDATE").min().alias("BUREAU_DAYS_CREDIT_ENDDATE_MIN"),
        pl.col("DAYS_CREDIT_ENDDATE").max().alias("BUREAU_DAYS_CREDIT_ENDDATE_MAX"),
        pl.col("DAYS_CREDIT_ENDDATE").mean().alias("BUREAU_DAYS_CREDIT_ENDDATE_MEAN"),
        
        pl.col("DAYS_ENDDATE_FACT").min().alias("BUREAU_DAYS_ENDDATE_FACT_MIN"),
        pl.col("DAYS_ENDDATE_FACT").max().alias("BUREAU_DAYS_ENDDATE_FACT_MAX"),
        pl.col("DAYS_ENDDATE_FACT").mean().alias("BUREAU_DAYS_ENDDATE_FACT_MEAN"),
        
        pl.col("AMT_CREDIT_MAX_OVERDUE").max().alias("BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX"),
        pl.col("AMT_CREDIT_MAX_OVERDUE").mean().alias("BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN"),
        
        pl.col("AMT_CREDIT_SUM").max().alias("BUREAU_AMT_CREDIT_SUM_MAX"),
        pl.col("AMT_CREDIT_SUM").mean().alias("BUREAU_AMT_CREDIT_SUM_MEAN"),
        pl.col("AMT_CREDIT_SUM").sum().alias("BUREAU_AMT_CREDIT_SUM_SUM"),
        
        pl.col("AMT_CREDIT_SUM_DEBT").max().alias("BUREAU_AMT_CREDIT_SUM_DEBT_MAX"),
        pl.col("AMT_CREDIT_SUM_DEBT").mean().alias("BUREAU_AMT_CREDIT_SUM_DEBT_MEAN"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUREAU_AMT_CREDIT_SUM_DEBT_SUM"),
        
        pl.col("AMT_CREDIT_SUM_OVERDUE").max().alias("BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").mean().alias("BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM"),
        
        pl.col("AMT_ANNUITY").max().alias("BUREAU_AMT_ANNUITY_MAX"),
        pl.col("AMT_ANNUITY").mean().alias("BUREAU_AMT_ANNUITY_MEAN"),
        pl.col("AMT_ANNUITY").sum().alias("BUREAU_AMT_ANNUITY_SUM"),
        
        pl.col("BUREAU_ENDDATE_FACT_DIFF").min().alias("BUREAU_BUREAU_ENDDATE_FACT_DIFF_MIN"),
        pl.col("BUREAU_ENDDATE_FACT_DIFF").max().alias("BUREAU_BUREAU_ENDDATE_FACT_DIFF_MAX"),
        pl.col("BUREAU_ENDDATE_FACT_DIFF").mean().alias("BUREAU_BUREAU_ENDDATE_FACT_DIFF_MEAN"),
        
        pl.col("BUREAU_CREDIT_FACT_DIFF").min().alias("BUREAU_BUREAU_CREDIT_FACT_DIFF_MIN"),
        pl.col("BUREAU_CREDIT_FACT_DIFF").max().alias("BUREAU_BUREAU_CREDIT_FACT_DIFF_MAX"),
        pl.col("BUREAU_CREDIT_FACT_DIFF").mean().alias("BUREAU_BUREAU_CREDIT_FACT_DIFF_MEAN"),
        
        pl.col("BUREAU_CREDIT_ENDDATE_DIFF").min().alias("BUREAU_BUREAU_CREDIT_ENDDATE_DIFF_MIN"),
        pl.col("BUREAU_CREDIT_ENDDATE_DIFF").max().alias("BUREAU_BUREAU_CREDIT_ENDDATE_DIFF_MAX"),
        pl.col("BUREAU_CREDIT_ENDDATE_DIFF").mean().alias("BUREAU_BUREAU_CREDIT_ENDDATE_DIFF_MEAN"),
        
        pl.col("BUREAU_CREDIT_DEBT_RATIO").min().alias("BUREAU_BUREAU_CREDIT_DEBT_RATIO_MIN"),
        pl.col("BUREAU_CREDIT_DEBT_RATIO").max().alias("BUREAU_BUREAU_CREDIT_DEBT_RATIO_MAX"),
        pl.col("BUREAU_CREDIT_DEBT_RATIO").mean().alias("BUREAU_BUREAU_CREDIT_DEBT_RATIO_MEAN"),
        
        pl.col("BUREAU_CREDIT_DEBT_DIFF").min().alias("BUREAU_BUREAU_CREDIT_DEBT_DIFF_MIN"),
        pl.col("BUREAU_CREDIT_DEBT_DIFF").max().alias("BUREAU_BUREAU_CREDIT_DEBT_DIFF_MAX"),
        pl.col("BUREAU_CREDIT_DEBT_DIFF").mean().alias("BUREAU_BUREAU_CREDIT_DEBT_DIFF_MEAN"),
        
        pl.col("BUREAU_IS_DPD").mean().alias("BUREAU_BUREAU_IS_DPD_MEAN"),
        pl.col("BUREAU_IS_DPD").sum().alias("BUREAU_BUREAU_IS_DPD_SUM"),
        
        pl.col("BUREAU_IS_DPD_OVER120").mean().alias("BUREAU_BUREAU_IS_DPD_OVER120_MEAN"),
        pl.col("BUREAU_IS_DPD_OVER120").sum().alias("BUREAU_BUREAU_IS_DPD_OVER120_SUM")
    ]
    
    bureau_day_amt_agg = bureau.group_by("SK_ID_CURR").agg(agg_expr)
    
    return bureau_day_amt_agg


def get_bureau_active_agg(bureau):
    '''
    Bureau CREDIT_ACTIVE='Active' filtering
    SK_ID_CURR aggregation
    '''
    # Filter for active credits
    bureau_active = bureau.filter(pl.col("CREDIT_ACTIVE") == "Active")
    
    agg_expr = [
        pl.count("SK_ID_BUREAU").alias("BUREAU_ACT_SK_ID_BUREAU_COUNT"),
        
        pl.col("DAYS_CREDIT").min().alias("BUREAU_ACT_DAYS_CREDIT_MIN"),
        pl.col("DAYS_CREDIT").max().alias("BUREAU_ACT_DAYS_CREDIT_MAX"),
        pl.col("DAYS_CREDIT").mean().alias("BUREAU_ACT_DAYS_CREDIT_MEAN"),
        
        pl.col("CREDIT_DAY_OVERDUE").min().alias("BUREAU_ACT_CREDIT_DAY_OVERDUE_MIN"),
        pl.col("CREDIT_DAY_OVERDUE").max().alias("BUREAU_ACT_CREDIT_DAY_OVERDUE_MAX"),
        pl.col("CREDIT_DAY_OVERDUE").mean().alias("BUREAU_ACT_CREDIT_DAY_OVERDUE_MEAN"),
        
        pl.col("DAYS_CREDIT_ENDDATE").min().alias("BUREAU_ACT_DAYS_CREDIT_ENDDATE_MIN"),
        pl.col("DAYS_CREDIT_ENDDATE").max().alias("BUREAU_ACT_DAYS_CREDIT_ENDDATE_MAX"),
        pl.col("DAYS_CREDIT_ENDDATE").mean().alias("BUREAU_ACT_DAYS_CREDIT_ENDDATE_MEAN"),
        
        pl.col("DAYS_ENDDATE_FACT").min().alias("BUREAU_ACT_DAYS_ENDDATE_FACT_MIN"),
        pl.col("DAYS_ENDDATE_FACT").max().alias("BUREAU_ACT_DAYS_ENDDATE_FACT_MAX"),
        pl.col("DAYS_ENDDATE_FACT").mean().alias("BUREAU_ACT_DAYS_ENDDATE_FACT_MEAN"),
        
        pl.col("AMT_CREDIT_MAX_OVERDUE").max().alias("BUREAU_ACT_AMT_CREDIT_MAX_OVERDUE_MAX"),
        pl.col("AMT_CREDIT_MAX_OVERDUE").mean().alias("BUREAU_ACT_AMT_CREDIT_MAX_OVERDUE_MEAN"),
        
        pl.col("AMT_CREDIT_SUM").max().alias("BUREAU_ACT_AMT_CREDIT_SUM_MAX"),
        pl.col("AMT_CREDIT_SUM").mean().alias("BUREAU_ACT_AMT_CREDIT_SUM_MEAN"),
        pl.col("AMT_CREDIT_SUM").sum().alias("BUREAU_ACT_AMT_CREDIT_SUM_SUM"),
        
        pl.col("AMT_CREDIT_SUM_DEBT").max().alias("BUREAU_ACT_AMT_CREDIT_SUM_DEBT_MAX"),
        pl.col("AMT_CREDIT_SUM_DEBT").mean().alias("BUREAU_ACT_AMT_CREDIT_SUM_DEBT_MEAN"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUREAU_ACT_AMT_CREDIT_SUM_DEBT_SUM"),
        
        pl.col("AMT_CREDIT_SUM_OVERDUE").max().alias("BUREAU_ACT_AMT_CREDIT_SUM_OVERDUE_MAX"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").mean().alias("BUREAU_ACT_AMT_CREDIT_SUM_OVERDUE_MEAN"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("BUREAU_ACT_AMT_CREDIT_SUM_OVERDUE_SUM"),
        
        pl.col("AMT_ANNUITY").max().alias("BUREAU_ACT_AMT_ANNUITY_MAX"),
        pl.col("AMT_ANNUITY").mean().alias("BUREAU_ACT_AMT_ANNUITY_MEAN"),
        pl.col("AMT_ANNUITY").sum().alias("BUREAU_ACT_AMT_ANNUITY_SUM"),
        
        pl.col("BUREAU_ENDDATE_FACT_DIFF").min().alias("BUREAU_ACT_BUREAU_ENDDATE_FACT_DIFF_MIN"),
        pl.col("BUREAU_ENDDATE_FACT_DIFF").max().alias("BUREAU_ACT_BUREAU_ENDDATE_FACT_DIFF_MAX"),
        pl.col("BUREAU_ENDDATE_FACT_DIFF").mean().alias("BUREAU_ACT_BUREAU_ENDDATE_FACT_DIFF_MEAN"),
        
        pl.col("BUREAU_CREDIT_FACT_DIFF").min().alias("BUREAU_ACT_BUREAU_CREDIT_FACT_DIFF_MIN"),
        pl.col("BUREAU_CREDIT_FACT_DIFF").max().alias("BUREAU_ACT_BUREAU_CREDIT_FACT_DIFF_MAX"),
        pl.col("BUREAU_CREDIT_FACT_DIFF").mean().alias("BUREAU_ACT_BUREAU_CREDIT_FACT_DIFF_MEAN"),
        
        pl.col("BUREAU_CREDIT_ENDDATE_DIFF").min().alias("BUREAU_ACT_BUREAU_CREDIT_ENDDATE_DIFF_MIN"),
        pl.col("BUREAU_CREDIT_ENDDATE_DIFF").max().alias("BUREAU_ACT_BUREAU_CREDIT_ENDDATE_DIFF_MAX"),
        pl.col("BUREAU_CREDIT_ENDDATE_DIFF").mean().alias("BUREAU_ACT_BUREAU_CREDIT_ENDDATE_DIFF_MEAN"),
        
        pl.col("BUREAU_CREDIT_DEBT_RATIO").min().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_RATIO_MIN"),
        pl.col("BUREAU_CREDIT_DEBT_RATIO").max().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_RATIO_MAX"),
        pl.col("BUREAU_CREDIT_DEBT_RATIO").mean().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_RATIO_MEAN"),
        
        pl.col("BUREAU_CREDIT_DEBT_DIFF").min().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_DIFF_MIN"),
        pl.col("BUREAU_CREDIT_DEBT_DIFF").max().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_DIFF_MAX"),
        pl.col("BUREAU_CREDIT_DEBT_DIFF").mean().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_DIFF_MEAN"),
        
        pl.col("BUREAU_IS_DPD").mean().alias("BUREAU_ACT_BUREAU_IS_DPD_MEAN"),
        pl.col("BUREAU_IS_DPD").sum().alias("BUREAU_ACT_BUREAU_IS_DPD_SUM"),
        
        pl.col("BUREAU_IS_DPD_OVER120").mean().alias("BUREAU_ACT_BUREAU_IS_DPD_OVER120_MEAN"),
        pl.col("BUREAU_IS_DPD_OVER120").sum().alias("BUREAU_ACT_BUREAU_IS_DPD_OVER120_SUM")
    ]
    
    bureau_active_agg = bureau_active.group_by("SK_ID_CURR").agg(agg_expr)
    
    return bureau_active_agg


def get_bureau_days750_agg(bureau):
    # Filter for recent bureau records
    bureau_days750 = bureau.filter(pl.col("DAYS_CREDIT") > -750)
    
    agg_expr = [
        pl.count("SK_ID_BUREAU").alias("BUREAU_ACT_SK_ID_BUREAU_COUNT"),
        
        pl.col("DAYS_CREDIT").min().alias("BUREAU_ACT_DAYS_CREDIT_MIN"),
        pl.col("DAYS_CREDIT").max().alias("BUREAU_ACT_DAYS_CREDIT_MAX"),
        pl.col("DAYS_CREDIT").mean().alias("BUREAU_ACT_DAYS_CREDIT_MEAN"),
        
        pl.col("CREDIT_DAY_OVERDUE").min().alias("BUREAU_ACT_CREDIT_DAY_OVERDUE_MIN"),
        pl.col("CREDIT_DAY_OVERDUE").max().alias("BUREAU_ACT_CREDIT_DAY_OVERDUE_MAX"),
        pl.col("CREDIT_DAY_OVERDUE").mean().alias("BUREAU_ACT_CREDIT_DAY_OVERDUE_MEAN"),
        
        pl.col("DAYS_CREDIT_ENDDATE").min().alias("BUREAU_ACT_DAYS_CREDIT_ENDDATE_MIN"),
        pl.col("DAYS_CREDIT_ENDDATE").max().alias("BUREAU_ACT_DAYS_CREDIT_ENDDATE_MAX"),
        pl.col("DAYS_CREDIT_ENDDATE").mean().alias("BUREAU_ACT_DAYS_CREDIT_ENDDATE_MEAN"),
        
        pl.col("DAYS_ENDDATE_FACT").min().alias("BUREAU_ACT_DAYS_ENDDATE_FACT_MIN"),
        pl.col("DAYS_ENDDATE_FACT").max().alias("BUREAU_ACT_DAYS_ENDDATE_FACT_MAX"),
        pl.col("DAYS_ENDDATE_FACT").mean().alias("BUREAU_ACT_DAYS_ENDDATE_FACT_MEAN"),
        
        pl.col("AMT_CREDIT_MAX_OVERDUE").max().alias("BUREAU_ACT_AMT_CREDIT_MAX_OVERDUE_MAX"),
        pl.col("AMT_CREDIT_MAX_OVERDUE").mean().alias("BUREAU_ACT_AMT_CREDIT_MAX_OVERDUE_MEAN"),
        
        pl.col("AMT_CREDIT_SUM").max().alias("BUREAU_ACT_AMT_CREDIT_SUM_MAX"),
        pl.col("AMT_CREDIT_SUM").mean().alias("BUREAU_ACT_AMT_CREDIT_SUM_MEAN"),
        pl.col("AMT_CREDIT_SUM").sum().alias("BUREAU_ACT_AMT_CREDIT_SUM_SUM"),
        
        pl.col("AMT_CREDIT_SUM_DEBT").max().alias("BUREAU_ACT_AMT_CREDIT_SUM_DEBT_MAX"),
        pl.col("AMT_CREDIT_SUM_DEBT").mean().alias("BUREAU_ACT_AMT_CREDIT_SUM_DEBT_MEAN"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUREAU_ACT_AMT_CREDIT_SUM_DEBT_SUM"),
        
        pl.col("AMT_CREDIT_SUM_OVERDUE").max().alias("BUREAU_ACT_AMT_CREDIT_SUM_OVERDUE_MAX"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").mean().alias("BUREAU_ACT_AMT_CREDIT_SUM_OVERDUE_MEAN"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("BUREAU_ACT_AMT_CREDIT_SUM_OVERDUE_SUM"),
        
        pl.col("AMT_ANNUITY").max().alias("BUREAU_ACT_AMT_ANNUITY_MAX"),
        pl.col("AMT_ANNUITY").mean().alias("BUREAU_ACT_AMT_ANNUITY_MEAN"),
        pl.col("AMT_ANNUITY").sum().alias("BUREAU_ACT_AMT_ANNUITY_SUM"),
        
        pl.col("BUREAU_ENDDATE_FACT_DIFF").min().alias("BUREAU_ACT_BUREAU_ENDDATE_FACT_DIFF_MIN"),
        pl.col("BUREAU_ENDDATE_FACT_DIFF").max().alias("BUREAU_ACT_BUREAU_ENDDATE_FACT_DIFF_MAX"),
        pl.col("BUREAU_ENDDATE_FACT_DIFF").mean().alias("BUREAU_ACT_BUREAU_ENDDATE_FACT_DIFF_MEAN"),
        
        pl.col("BUREAU_CREDIT_FACT_DIFF").min().alias("BUREAU_ACT_BUREAU_CREDIT_FACT_DIFF_MIN"),
        pl.col("BUREAU_CREDIT_FACT_DIFF").max().alias("BUREAU_ACT_BUREAU_CREDIT_FACT_DIFF_MAX"),
        pl.col("BUREAU_CREDIT_FACT_DIFF").mean().alias("BUREAU_ACT_BUREAU_CREDIT_FACT_DIFF_MEAN"),
        
        pl.col("BUREAU_CREDIT_ENDDATE_DIFF").min().alias("BUREAU_ACT_BUREAU_CREDIT_ENDDATE_DIFF_MIN"),
        pl.col("BUREAU_CREDIT_ENDDATE_DIFF").max().alias("BUREAU_ACT_BUREAU_CREDIT_ENDDATE_DIFF_MAX"),
        pl.col("BUREAU_CREDIT_ENDDATE_DIFF").mean().alias("BUREAU_ACT_BUREAU_CREDIT_ENDDATE_DIFF_MEAN"),
        
        pl.col("BUREAU_CREDIT_DEBT_RATIO").min().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_RATIO_MIN"),
        pl.col("BUREAU_CREDIT_DEBT_RATIO").max().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_RATIO_MAX"),
        pl.col("BUREAU_CREDIT_DEBT_RATIO").mean().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_RATIO_MEAN"),
        
        pl.col("BUREAU_CREDIT_DEBT_DIFF").min().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_DIFF_MIN"),
        pl.col("BUREAU_CREDIT_DEBT_DIFF").max().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_DIFF_MAX"),
        pl.col("BUREAU_CREDIT_DEBT_DIFF").mean().alias("BUREAU_ACT_BUREAU_CREDIT_DEBT_DIFF_MEAN"),
        
        pl.col("BUREAU_IS_DPD").mean().alias("BUREAU_ACT_BUREAU_IS_DPD_MEAN"),
        pl.col("BUREAU_IS_DPD").sum().alias("BUREAU_ACT_BUREAU_IS_DPD_SUM"),
        
        pl.col("BUREAU_IS_DPD_OVER120").mean().alias("BUREAU_ACT_BUREAU_IS_DPD_OVER120_MEAN"),
        pl.col("BUREAU_IS_DPD_OVER120").sum().alias("BUREAU_ACT_BUREAU_IS_DPD_OVER120_SUM")
    ]
    
    bureau_days750_agg = bureau_days750.group_by("SK_ID_CURR").agg(agg_expr)
    
    return bureau_days750_agg


def get_bureau_bal_agg(bureau, bureau_bal):
    # Join bureau_balance with bureau to get SK_ID_CURR
    bureau_subset = bureau.select(["SK_ID_CURR", "SK_ID_BUREAU"])
    bureau_bal = bureau_bal.join(bureau_subset, on="SK_ID_BUREAU", how="left")
    
    # Define DPD indicators based on STATUS
    bureau_bal = bureau_bal.with_columns([
        pl.col("STATUS").is_in(["1", "2", "3", "4", "5"]).cast(pl.Int32).alias("BUREAU_BAL_IS_DPD"),
        (pl.col("STATUS") == "5").cast(pl.Int32).alias("BUREAU_BAL_IS_DPD_OVER120")
    ])
    
    # Aggregate by SK_ID_CURR
    agg_expr = [
        pl.count("SK_ID_CURR").alias("BUREAU_BAL_SK_ID_CURR_COUNT"),
        
        pl.col("MONTHS_BALANCE").min().alias("BUREAU_BAL_MONTHS_BALANCE_MIN"),
        pl.col("MONTHS_BALANCE").max().alias("BUREAU_BAL_MONTHS_BALANCE_MAX"),
        pl.col("MONTHS_BALANCE").mean().alias("BUREAU_BAL_MONTHS_BALANCE_MEAN"),
        
        pl.col("BUREAU_BAL_IS_DPD").mean().alias("BUREAU_BAL_BUREAU_BAL_IS_DPD_MEAN"),
        pl.col("BUREAU_BAL_IS_DPD").sum().alias("BUREAU_BAL_BUREAU_BAL_IS_DPD_SUM"),
        
        pl.col("BUREAU_BAL_IS_DPD_OVER120").mean().alias("BUREAU_BAL_BUREAU_BAL_IS_DPD_OVER120_MEAN"),
        pl.col("BUREAU_BAL_IS_DPD_OVER120").sum().alias("BUREAU_BAL_BUREAU_BAL_IS_DPD_OVER120_SUM")
    ]
    
    bureau_bal_agg = bureau_bal.group_by("SK_ID_CURR").agg(agg_expr)
    
    return bureau_bal_agg


def get_bureau_agg(bureau, bureau_bal):
    bureau = get_bureau_processed(bureau)
    bureau_day_amt_agg = get_bureau_day_amt_agg(bureau)
    bureau_active_agg = get_bureau_active_agg(bureau)
    bureau_days750_agg = get_bureau_days750_agg(bureau)
    bureau_bal_agg = get_bureau_bal_agg(bureau, bureau_bal)

    # Merge all aggregations
    bureau_agg = bureau_day_amt_agg.join(bureau_active_agg, on="SK_ID_CURR", how="left")
    
    # Calculate ratios
    bureau_agg = bureau_agg.with_columns([
        (pl.col("BUREAU_ACT_BUREAU_IS_DPD_SUM") / pl.col("BUREAU_SK_ID_BUREAU_COUNT")).alias("BUREAU_ACT_IS_DPD_RATIO"),
        (pl.col("BUREAU_ACT_BUREAU_IS_DPD_OVER120_SUM") / pl.col("BUREAU_SK_ID_BUREAU_COUNT")).alias("BUREAU_ACT_IS_DPD_OVER120_RATIO")
    ])
    
    # Merge with bureau_bal_agg and bureau_days750_agg
    bureau_agg = bureau_agg.join(bureau_bal_agg, on="SK_ID_CURR", how="left")
    bureau_agg = bureau_agg.join(bureau_days750_agg, on="SK_ID_CURR", how="left")
    
    return bureau_agg


def get_pos_bal_agg(pos_bal):
    # Define DPD indicators
    pos_bal = pos_bal.with_columns([
        (pl.col("SK_DPD") > 0).cast(pl.Int32).alias("POS_IS_DPD"),
        ((pl.col("SK_DPD") > 0) & (pl.col("SK_DPD") < 120)).cast(pl.Int32).alias("POS_IS_DPD_UNDER_120"),
        (pl.col("SK_DPD") >= 120).cast(pl.Int32).alias("POS_IS_DPD_OVER_120")
    ])
    
    # Aggregate by SK_ID_CURR
    agg_expr = [
        pl.count("SK_ID_CURR").alias("POS_SK_ID_CURR_COUNT"),
        
        pl.col("MONTHS_BALANCE").min().alias("POS_MONTHS_BALANCE_MIN"),
        pl.col("MONTHS_BALANCE").mean().alias("POS_MONTHS_BALANCE_MEAN"),
        pl.col("MONTHS_BALANCE").max().alias("POS_MONTHS_BALANCE_MAX"),
        
        pl.col("SK_DPD").min().alias("POS_SK_DPD_MIN"),
        pl.col("SK_DPD").max().alias("POS_SK_DPD_MAX"),
        pl.col("SK_DPD").mean().alias("POS_SK_DPD_MEAN"),
        pl.col("SK_DPD").sum().alias("POS_SK_DPD_SUM"),
        
        pl.col("CNT_INSTALMENT").min().alias("POS_CNT_INSTALMENT_MIN"),
        pl.col("CNT_INSTALMENT").max().alias("POS_CNT_INSTALMENT_MAX"),
        pl.col("CNT_INSTALMENT").mean().alias("POS_CNT_INSTALMENT_MEAN"),
        pl.col("CNT_INSTALMENT").sum().alias("POS_CNT_INSTALMENT_SUM"),
        
        pl.col("CNT_INSTALMENT_FUTURE").min().alias("POS_CNT_INSTALMENT_FUTURE_MIN"),
        pl.col("CNT_INSTALMENT_FUTURE").max().alias("POS_CNT_INSTALMENT_FUTURE_MAX"),
        pl.col("CNT_INSTALMENT_FUTURE").mean().alias("POS_CNT_INSTALMENT_FUTURE_MEAN"),
        pl.col("CNT_INSTALMENT_FUTURE").sum().alias("POS_CNT_INSTALMENT_FUTURE_SUM"),
        
        pl.col("POS_IS_DPD").mean().alias("POS_POS_IS_DPD_MEAN"),
        pl.col("POS_IS_DPD").sum().alias("POS_POS_IS_DPD_SUM"),
        
        pl.col("POS_IS_DPD_UNDER_120").mean().alias("POS_POS_IS_DPD_UNDER_120_MEAN"),
        pl.col("POS_IS_DPD_UNDER_120").sum().alias("POS_POS_IS_DPD_UNDER_120_SUM"),
        
        pl.col("POS_IS_DPD_OVER_120").mean().alias("POS_POS_IS_DPD_OVER_120_MEAN"),
        pl.col("POS_IS_DPD_OVER_120").sum().alias("POS_POS_IS_DPD_OVER_120_SUM")
    ]
    
    pos_bal_agg = pos_bal.group_by("SK_ID_CURR").agg(agg_expr)
    
    # Filter for recent MONTHS_BALANCE (> -20)
    pos_bal_m20 = pos_bal.filter(pl.col("MONTHS_BALANCE") > -20)
    pos_bal_m20_agg = pos_bal_m20.group_by("SK_ID_CURR").agg(agg_expr)
    
    # Rename columns for m20 aggregate
    new_names = {}
    for col in pos_bal_m20_agg.columns:
        if col != "SK_ID_CURR":
            new_names[col] = f"POS_M20{col[3:]}"
    
    pos_bal_m20_agg = pos_bal_m20_agg.rename(new_names)
    
    # Join the two aggregations
    pos_bal_agg = pos_bal_agg.join(pos_bal_m20_agg, on="SK_ID_CURR", how="left")
    
    return pos_bal_agg


def get_install_agg(install):
    # Calculate derived features
    install = install.with_columns([
        (pl.col("AMT_INSTALMENT") - pl.col("AMT_PAYMENT")).alias("AMT_DIFF"),
        ((pl.col("AMT_PAYMENT") + 1) / (pl.col("AMT_INSTALMENT") + 1)).alias("AMT_RATIO"),
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("SK_DPD"),
        
        # DPD indicators
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT") > 0).cast(pl.Int32).alias("INS_IS_DPD"),
        ((pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT") > 0) & 
         (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT") < 120)).cast(pl.Int32).alias("INS_IS_DPD_UNDER_120"),
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT") >= 120).cast(pl.Int32).alias("INS_IS_DPD_OVER_120")
    ])
    
    # Aggregate by SK_ID_CURR
    agg_expr = [
        pl.count("SK_ID_CURR").alias("INS_SK_ID_CURR_COUNT"),
        pl.col("NUM_INSTALMENT_VERSION").n_unique().alias("INS_NUM_INSTALMENT_VERSION_NUNIQUE"),
        
        pl.col("DAYS_ENTRY_PAYMENT").mean().alias("INS_DAYS_ENTRY_PAYMENT_MEAN"),
        pl.col("DAYS_ENTRY_PAYMENT").max().alias("INS_DAYS_ENTRY_PAYMENT_MAX"),
        pl.col("DAYS_ENTRY_PAYMENT").sum().alias("INS_DAYS_ENTRY_PAYMENT_SUM"),
        
        pl.col("DAYS_INSTALMENT").mean().alias("INS_DAYS_INSTALMENT_MEAN"),
        pl.col("DAYS_INSTALMENT").max().alias("INS_DAYS_INSTALMENT_MAX"),
        pl.col("DAYS_INSTALMENT").sum().alias("INS_DAYS_INSTALMENT_SUM"),
        
        pl.col("AMT_INSTALMENT").mean().alias("INS_AMT_INSTALMENT_MEAN"),
        pl.col("AMT_INSTALMENT").max().alias("INS_AMT_INSTALMENT_MAX"),
        pl.col("AMT_INSTALMENT").sum().alias("INS_AMT_INSTALMENT_SUM"),
        
        pl.col("AMT_PAYMENT").mean().alias("INS_AMT_PAYMENT_MEAN"),
        pl.col("AMT_PAYMENT").max().alias("INS_AMT_PAYMENT_MAX"),
        pl.col("AMT_PAYMENT").sum().alias("INS_AMT_PAYMENT_SUM"),
        
        pl.col("AMT_DIFF").mean().alias("INS_AMT_DIFF_MEAN"),
        pl.col("AMT_DIFF").min().alias("INS_AMT_DIFF_MIN"),
        pl.col("AMT_DIFF").max().alias("INS_AMT_DIFF_MAX"),
        pl.col("AMT_DIFF").sum().alias("INS_AMT_DIFF_SUM"),
        
        pl.col("AMT_RATIO").mean().alias("INS_AMT_RATIO_MEAN"),
        pl.col("AMT_RATIO").max().alias("INS_AMT_RATIO_MAX"),
        
        pl.col("SK_DPD").mean().alias("INS_SK_DPD_MEAN"),
        pl.col("SK_DPD").min().alias("INS_SK_DPD_MIN"),
        pl.col("SK_DPD").max().alias("INS_SK_DPD_MAX"),
        
        pl.col("INS_IS_DPD").mean().alias("INS_INS_IS_DPD_MEAN"),
        pl.col("INS_IS_DPD").sum().alias("INS_INS_IS_DPD_SUM"),
        
        pl.col("INS_IS_DPD_UNDER_120").mean().alias("INS_INS_IS_DPD_UNDER_120_MEAN"),
        pl.col("INS_IS_DPD_UNDER_120").sum().alias("INS_INS_IS_DPD_UNDER_120_SUM"),
        
        pl.col("INS_IS_DPD_OVER_120").mean().alias("INS_INS_IS_DPD_OVER_120_MEAN"),
        pl.col("INS_IS_DPD_OVER_120").sum().alias("INS_INS_IS_DPD_OVER_120_SUM")
    ]
    
    install_agg = install.group_by("SK_ID_CURR").agg(agg_expr)
    
    # Filter for recent DAYS_ENTRY_PAYMENT (>= -365)
    install_d365 = install.filter(pl.col("DAYS_ENTRY_PAYMENT") >= -365)
    install_d365_agg = install_d365.group_by("SK_ID_CURR").agg(agg_expr)
    
    # Rename columns for d365 aggregate
    new_names = {}
    for col in install_d365_agg.columns:
        if col != "SK_ID_CURR":
            new_names[col] = f"INS_D365{col[3:]}"
    
    install_d365_agg = install_d365_agg.rename(new_names)
    
    # Join the two aggregations
    install_agg = install_agg.join(install_d365_agg, on="SK_ID_CURR", how="left")
    
    return install_agg


def get_card_bal_agg(card_bal):
    # Calculate derived features
    card_bal = card_bal.with_columns([
        (pl.col("AMT_BALANCE") / pl.col("AMT_CREDIT_LIMIT_ACTUAL")).alias("BALANCE_LIMIT_RATIO"),
        (pl.col("AMT_DRAWINGS_CURRENT") / pl.col("AMT_CREDIT_LIMIT_ACTUAL")).alias("DRAWING_LIMIT_RATIO"),
        
        # DPD indicators
        (pl.col("SK_DPD") > 0).cast(pl.Int32).alias("CARD_IS_DPD"),
        ((pl.col("SK_DPD") > 0) & (pl.col("SK_DPD") < 120)).cast(pl.Int32).alias("CARD_IS_DPD_UNDER_120"),
        (pl.col("SK_DPD") >= 120).cast(pl.Int32).alias("CARD_IS_DPD_OVER_120")
    ])
    
    # Aggregate by SK_ID_CURR
    agg_expr = [
        pl.count("SK_ID_CURR").alias("CARD_SK_ID_CURR_COUNT"),
        
        pl.col("AMT_BALANCE").max().alias("CARD_AMT_BALANCE_MAX"),
        pl.col("AMT_CREDIT_LIMIT_ACTUAL").max().alias("CARD_AMT_CREDIT_LIMIT_ACTUAL_MAX"),
        
        pl.col("AMT_DRAWINGS_ATM_CURRENT").max().alias("CARD_AMT_DRAWINGS_ATM_CURRENT_MAX"),
        pl.col("AMT_DRAWINGS_ATM_CURRENT").sum().alias("CARD_AMT_DRAWINGS_ATM_CURRENT_SUM"),
        
        pl.col("AMT_DRAWINGS_CURRENT").max().alias("CARD_AMT_DRAWINGS_CURRENT_MAX"),
        pl.col("AMT_DRAWINGS_CURRENT").sum().alias("CARD_AMT_DRAWINGS_CURRENT_SUM"),
        
        pl.col("AMT_DRAWINGS_POS_CURRENT").max().alias("CARD_AMT_DRAWINGS_POS_CURRENT_MAX"),
        pl.col("AMT_DRAWINGS_POS_CURRENT").sum().alias("CARD_AMT_DRAWINGS_POS_CURRENT_SUM"),
        
        pl.col("AMT_INST_MIN_REGULARITY").max().alias("CARD_AMT_INST_MIN_REGULARITY_MAX"),
        pl.col("AMT_INST_MIN_REGULARITY").mean().alias("CARD_AMT_INST_MIN_REGULARITY_MEAN"),
        
        pl.col("AMT_PAYMENT_TOTAL_CURRENT").max().alias("CARD_AMT_PAYMENT_TOTAL_CURRENT_MAX"),
        pl.col("AMT_PAYMENT_TOTAL_CURRENT").sum().alias("CARD_AMT_PAYMENT_TOTAL_CURRENT_SUM"),
        
        pl.col("AMT_TOTAL_RECEIVABLE").max().alias("CARD_AMT_TOTAL_RECEIVABLE_MAX"),
        pl.col("AMT_TOTAL_RECEIVABLE").mean().alias("CARD_AMT_TOTAL_RECEIVABLE_MEAN"),
        
        pl.col("CNT_DRAWINGS_ATM_CURRENT").max().alias("CARD_CNT_DRAWINGS_ATM_CURRENT_MAX"),
        pl.col("CNT_DRAWINGS_ATM_CURRENT").sum().alias("CARD_CNT_DRAWINGS_ATM_CURRENT_SUM"),
        
        pl.col("CNT_DRAWINGS_CURRENT").max().alias("CARD_CNT_DRAWINGS_CURRENT_MAX"),
        pl.col("CNT_DRAWINGS_CURRENT").mean().alias("CARD_CNT_DRAWINGS_CURRENT_MEAN"),
        pl.col("CNT_DRAWINGS_CURRENT").sum().alias("CARD_CNT_DRAWINGS_CURRENT_SUM"),
        
        pl.col("CNT_DRAWINGS_POS_CURRENT").mean().alias("CARD_CNT_DRAWINGS_POS_CURRENT_MEAN"),
        
        pl.col("SK_DPD").mean().alias("CARD_SK_DPD_MEAN"),
        pl.col("SK_DPD").max().alias("CARD_SK_DPD_MAX"),
        pl.col("SK_DPD").sum().alias("CARD_SK_DPD_SUM"),
        
        pl.col("BALANCE_LIMIT_RATIO").min().alias("CARD_BALANCE_LIMIT_RATIO_MIN"),
        pl.col("BALANCE_LIMIT_RATIO").max().alias("CARD_BALANCE_LIMIT_RATIO_MAX"),
        
        pl.col("DRAWING_LIMIT_RATIO").min().alias("CARD_DRAWING_LIMIT_RATIO_MIN"),
        pl.col("DRAWING_LIMIT_RATIO").max().alias("CARD_DRAWING_LIMIT_RATIO_MAX"),
        
        pl.col("CARD_IS_DPD").mean().alias("CARD_CARD_IS_DPD_MEAN"),
        pl.col("CARD_IS_DPD").sum().alias("CARD_CARD_IS_DPD_SUM"),
        
        pl.col("CARD_IS_DPD_UNDER_120").mean().alias("CARD_CARD_IS_DPD_UNDER_120_MEAN"),
        pl.col("CARD_IS_DPD_UNDER_120").sum().alias("CARD_CARD_IS_DPD_UNDER_120_SUM"),
        
        pl.col("CARD_IS_DPD_OVER_120").mean().alias("CARD_CARD_IS_DPD_OVER_120_MEAN"),
        pl.col("CARD_IS_DPD_OVER_120").sum().alias("CARD_CARD_IS_DPD_OVER_120_SUM")
    ]
    
    card_bal_agg = card_bal.group_by("SK_ID_CURR").agg(agg_expr)
    
    # Filter for recent MONTHS_BALANCE (>= -3)
    card_bal_m3 = card_bal.filter(pl.col("MONTHS_BALANCE") >= -3)
    card_bal_m3_agg = card_bal_m3.group_by("SK_ID_CURR").agg(agg_expr)
    
    # Rename columns for m3 aggregate
    new_names = {}
    for col in card_bal_m3_agg.columns:
        if col != "SK_ID_CURR":
            new_names[col] = f"CARD_M3{col[4:]}"
    
    card_bal_m3_agg = card_bal_m3_agg.rename(new_names)
    
    # Join the two aggregations
    card_bal_agg = card_bal_agg.join(card_bal_m3_agg, on="SK_ID_CURR", how="left")
    
    return card_bal_agg


def get_apps_all_with_all_agg(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal):
    """
    Description:
    1. Data preparation, aggregation
    2. Produce the finalized result
    """
    apps_all = get_apps_processed(apps)
    prev_agg = get_prev_agg(prev)
    bureau_agg = get_bureau_agg(bureau, bureau_bal)
    pos_bal_agg = get_pos_bal_agg(pos_bal)
    install_agg = get_install_agg(install)
    card_bal_agg = get_card_bal_agg(card_bal)
    
    print('prev_agg shape:', prev_agg.shape, 'bureau_agg shape:', bureau_agg.shape)
    print('pos_bal_agg shape:', pos_bal_agg.shape, 'install_agg shape:', install_agg.shape, 'card_bal_agg shape:', card_bal_agg.shape)
    print('apps_all before merge shape:', apps_all.shape)

    # Join with apps_all
    apps_all = apps_all.join(prev_agg, on="SK_ID_CURR", how="left")
    apps_all = apps_all.join(bureau_agg, on="SK_ID_CURR", how="left")
    apps_all = apps_all.join(pos_bal_agg, on="SK_ID_CURR", how="left")
    apps_all = apps_all.join(install_agg, on="SK_ID_CURR", how="left")
    apps_all = apps_all.join(card_bal_agg, on="SK_ID_CURR", how="left")

    print('apps_all after merge with all shape:', apps_all.shape)

    return apps_all


def get_apps_all_encoded(apps_all):
    # Get list of object/string columns
    object_columns = [col for col in apps_all.columns if apps_all[col].dtype == pl.Utf8]
    
    # Process each categorical column
    for column in object_columns:
        # Get unique values
        distinct_vals = apps_all.select(pl.col(column).unique()).to_series().to_list()
        
        # Create mapping dictionary
        mapping = {val: i for i, val in enumerate(distinct_vals) if val is not None}
        
        # Apply mapping using replace
        apps_all = apps_all.with_columns(
            pl.col(column).replace(mapping, default=None).alias(column)
        )
    
    return apps_all


def feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal):
    apps_all = get_apps_all_with_all_agg(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    # Category Label
    apps_all = get_apps_all_encoded(apps_all)
    
    # Convert back to pandas DataFrame for compatibility with existing code
    import pandas as pd
    return apps_all.to_pandas()



def main():
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = data_loaded()
    all_data = feature_engineering(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)


if __name__ == '__main__':
    main()