import pandas as pd
import numpy as np
import gc
import logging

def bureau(bureau, bureau_balance):
    """
    Process the bureau dataset, merge it with the bureau balance dataset,
    and return a preprocessed dataframe with advanced feature engineering.

    Parameters:
    -----------
    bureau : pandas.DataFrame
        Raw bureau dataset.
    bureau_balance : pandas.DataFrame
        Processed bureau_balance dataset with engineered features.

    Returns:
    --------
    pandas.DataFrame
        Feature-engineered dataset combining bureau and bureau_balance.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Merge bureau_balance into bureau
    bureau = bureau.merge(bureau_balance, how='left', on='SK_ID_BUREAU')
    bureau.fillna(0, inplace=True)

    # Number of Past Loans Per Customer
    grp = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].groupby('SK_ID_CURR')['DAYS_CREDIT'].count().reset_index()
    grp.rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'}, inplace=True)
    bureau = bureau.merge(grp, on='SK_ID_CURR', how='left')

    # Number of Types of Past Loans Per Customer
    grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby('SK_ID_CURR')['CREDIT_TYPE'].nunique().reset_index()
    grp.rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'}, inplace=True)
    bureau = bureau.merge(grp, on='SK_ID_CURR', how='left')

    # Average Number of Loans Per Type Per Customer
    bureau['AVERAGE_LOAN_TYPE'] = bureau['BUREAU_LOAN_COUNT'] / bureau['BUREAU_LOAN_TYPES']
    bureau.drop(['BUREAU_LOAN_COUNT', 'BUREAU_LOAN_TYPES'], axis=1, inplace=True)

    # % of Active Loans From Bureau Data
    bureau['CREDIT_ACTIVE_BINARY'] = bureau['CREDIT_ACTIVE'].apply(lambda x: 0 if x == 'Closed' else 1)
    grp = bureau.groupby('SK_ID_CURR')['CREDIT_ACTIVE_BINARY'].mean().reset_index()
    grp.rename(index=str, columns={'CREDIT_ACTIVE_BINARY': 'ACTIVE_LOANS_PERCENTAGE'}, inplace=True)
    bureau = bureau.merge(grp, on='SK_ID_CURR', how='left')
    bureau.drop('CREDIT_ACTIVE_BINARY', axis=1, inplace=True)

    # Average Days Between Successive Applications
    grp = bureau[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby('SK_ID_CURR')
    grp_sorted = grp.apply(lambda x: x.sort_values('DAYS_CREDIT', ascending=False)).reset_index(drop=True)
    grp_sorted['DAYS_CREDIT_NEG'] = grp_sorted['DAYS_CREDIT'] * -1
    grp_sorted['DAYS_DIFF'] = grp_sorted.groupby('SK_ID_CURR')['DAYS_CREDIT_NEG'].diff().fillna(0).astype('uint32')
    bureau = bureau.merge(grp_sorted[['SK_ID_BUREAU', 'DAYS_DIFF']], on='SK_ID_BUREAU', how='left')

    # % of Loans Where End Date is in the Future
    bureau['CREDIT_ENDDATE_BINARY'] = bureau['DAYS_CREDIT_ENDDATE'].apply(lambda x: 0 if x < 0 else 1)
    grp = bureau.groupby(by=['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(
        index=str, columns={'CREDIT_ENDDATE_BINARY': 'CREDIT_ENDDATE_PERCENTAGE'})
    bureau = bureau.merge(grp, on=['SK_ID_CURR'], how='left')
    del bureau['CREDIT_ENDDATE_BINARY']
    
    # Average Days Credit Expires in the Future
    bureau['CREDIT_ENDDATE_BINARY'] = bureau['DAYS_CREDIT_ENDDATE'].apply(lambda x: 0 if x < 0 else 1)
    bureau_temp = bureau[bureau['CREDIT_ENDDATE_BINARY'] == 1]
    bureau_temp['DAYS_CREDIT_ENDDATE1'] = bureau_temp['DAYS_CREDIT_ENDDATE']
    grp1 = bureau_temp[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE1']].groupby(by=['SK_ID_CURR'])
    grp1 = grp1.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE1'], ascending=True)).reset_index(drop=True)
    grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE1'].diff().fillna(0).astype('uint32')
    bureau = bureau.merge(grp1[['SK_ID_BUREAU', 'DAYS_ENDDATE_DIFF']], on=['SK_ID_BUREAU'], how='left')
    grp = bureau.groupby(by=['SK_ID_CURR'])['DAYS_ENDDATE_DIFF'].mean().reset_index().rename(
        index=str, columns={'DAYS_ENDDATE_DIFF': 'AVG_ENDDATE_FUTURE'})
    bureau = bureau.merge(grp, on=['SK_ID_CURR'], how='left')
    del bureau['DAYS_ENDDATE_DIFF'], bureau['CREDIT_ENDDATE_BINARY'], bureau['DAYS_CREDIT_ENDDATE']
    
    # Debt over Credit Ratio
    bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
    bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)
    grp1 = bureau.groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(
        index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp2 = bureau.groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(
        index=str, columns={'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})
    bureau = bureau.merge(grp1, on=['SK_ID_CURR'], how='left').merge(grp2, on=['SK_ID_CURR'], how='left')
    bureau['DEBT_CREDIT_RATIO'] = bureau['TOTAL_CUSTOMER_DEBT'] / bureau['TOTAL_CUSTOMER_CREDIT']
    del bureau['TOTAL_CUSTOMER_DEBT'], bureau['TOTAL_CUSTOMER_CREDIT']
    
    # Overdue over Debt Ratio
    bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)
    grp1 = bureau.groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(
        index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp2 = bureau.groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(
        index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    bureau = bureau.merge(grp1, on=['SK_ID_CURR'], how='left').merge(grp2, on=['SK_ID_CURR'], how='left')
    bureau['OVERDUE_DEBT_RATIO'] = bureau['TOTAL_CUSTOMER_OVERDUE'] / bureau['TOTAL_CUSTOMER_DEBT']
    del bureau['TOTAL_CUSTOMER_OVERDUE'], bureau['TOTAL_CUSTOMER_DEBT']
    
    # Average Number of Loans Prolonged
    bureau['CNT_CREDIT_PROLONG'] = bureau['CNT_CREDIT_PROLONG'].fillna(0)
    grp = bureau.groupby(by=['SK_ID_CURR'])['CNT_CREDIT_PROLONG'].mean().reset_index().rename(
        index=str, columns={'CNT_CREDIT_PROLONG': 'AVG_CREDITDAYS_PROLONGED'})
    bureau = bureau.merge(grp, on=['SK_ID_CURR'], how='left')
    del grp

    # One-Hot Encoding for Categorical Columns
    categorical_columns = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']
    bureau = pd.get_dummies(bureau, columns=categorical_columns, prefix_sep='_', dummy_na=False)

    # Aggregations
    existing_columns = bureau.columns.tolist()
    
    # Define default aggregation for numerical columns
    default_aggs = ['min', 'max', 'mean', 'sum']
    
    # Create aggregation dictionary with only existing columns
    agg_dict = {}
    
    # Define the columns to potentially aggregate
    columns_to_check = [
        'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 
        'DAYS_ENDDATE_FACT', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 
        'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 
        'DAYS_CREDIT_UPDATE', 'MONTHS_BALANCE_MIN', 'MONTHS_BALANCE_MAX', 
        'MONTHS_BALANCE_MEAN', 'MONTHS_BALANCE_COUNT', 
        'STATUS_0_MEAN', 'STATUS_1_MEAN', 'STATUS_2_MEAN', 
        'STATUS_3_MEAN', 'STATUS_4_MEAN', 'STATUS_5_MEAN', 
        'STATUS_C_MEAN', 'STATUS_X_MEAN', 'LATE_PAYMENT_RATIO'
    ]
    
    # Add columns with available aggregations
    for col in columns_to_check:
        if col in existing_columns:
            agg_dict[col] = default_aggs

    # Add one-hot encoded columns with mean aggregation
    onehot_columns = [col for col in bureau.columns if col.startswith(tuple(categorical_columns))]
    for col in onehot_columns:
        agg_dict[col] = ['mean']

    # Log the columns being aggregated
    logger.info(f"Aggregating columns: {list(agg_dict.keys())}")

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_dict)

    bureau_agg.columns = [f"BUREAU_{col[0]}_{col[1].upper()}" for col in bureau_agg.columns]
    bureau_agg = bureau_agg.reset_index()

    # Add safe checks for column existence
    if all(col in bureau_agg.columns for col in ['BUREAU_AMT_CREDIT_SUM_DEBT_SUM', 'BUREAU_AMT_CREDIT_SUM_SUM']):
        bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1e-8)
    
    if all(col in bureau_agg.columns for col in ['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM', 'BUREAU_AMT_CREDIT_SUM_SUM']):
        bureau_agg['BUREAU_OVERDUE_CREDIT_RATIO'] = bureau_agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM'] / (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1e-8)
    
    if all(col in bureau_agg.columns for col in ['BUREAU_DAYS_CREDIT_ENDDATE_MEAN', 'BUREAU_DAYS_CREDIT_MEAN']):
        bureau_agg['BUREAU_AVERAGE_CREDIT_DURATION'] = bureau_agg['BUREAU_DAYS_CREDIT_ENDDATE_MEAN'] - bureau_agg['BUREAU_DAYS_CREDIT_MEAN']
    
    if 'BUREAU_CNT_CREDIT_PROLONG_SUM' in bureau_agg.columns and 'BUREAU_AMT_CREDIT_SUM_SUM' in bureau_agg.columns:
        bureau_agg['BUREAU_PROLONGATION_RATIO'] = bureau_agg['BUREAU_CNT_CREDIT_PROLONG_SUM'] / (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1e-8)

    # Additional safe checks for other features
    if 'BUREAU_CREDIT_DAY_OVERDUE_MAX' in bureau_agg.columns:
        bureau_agg['BUREAU_OVERDUE_COUNT'] = (bureau_agg['BUREAU_CREDIT_DAY_OVERDUE_MAX'] > 0).astype(int)

    if all(col in bureau_agg.columns for col in ['BUREAU_AMT_CREDIT_SUM_DEBT_SUM', 'BUREAU_AMT_CREDIT_SUM_SUM']):
        bureau_agg['BUREAU_WEIGHTED_DEBT_RATIO'] = (
            bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] * bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM']
        ) / (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1e-8)

    # Safe calculation of credit health score
    if all(col in bureau_agg.columns for col in ['BUREAU_AMT_CREDIT_SUM_DEBT_SUM', 'BUREAU_AMT_CREDIT_SUM_SUM', 'BUREAU_DAYS_CREDIT_ENDDATE_MEAN']):
        bureau_agg['BUREAU_CREDIT_HEALTH_SCORE'] = (
            bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1e-8)
        ) - (bureau_agg['BUREAU_DAYS_CREDIT_ENDDATE_MEAN'] / 365)

    gc.collect() 
    return bureau_agg