import pandas as pd
import numpy as np
import gc

def previous_application(df):
    # Identify categorical columns
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    # Convert categorical variables to dummy variables
    prev = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    
    # Drop unnecessary columns
    prev.drop(['SK_ID_PREV', 'Unnamed: 0'], axis=1, inplace=True)
    
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = np.where(
        prev['AMT_CREDIT'] != 0, 
        prev['AMT_APPLICATION'] / prev['AMT_CREDIT'], 
        0
    )
    
    # Numerical aggregations
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    
    # Categorical aggregations
    cat_aggregations = {col: ['mean'] for col in prev.columns if col not in num_aggregations}
    
    # Combine numerical and categorical aggregations
    aggregations = {**num_aggregations, **cat_aggregations}
    
    # Group by SK_ID_CURR and aggregate
    prev_agg = prev.groupby('SK_ID_CURR').agg(aggregations)
    prev_agg.columns = pd.Index(['PREV_' + col[0] + "_" + col[1].upper() for col in prev_agg.columns.tolist()])
    
    # Approved
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    prev_agg.fillna(0, inplace =True)
    prev_agg = prev_agg.reset_index()
    gc.collect()
    return prev_agg