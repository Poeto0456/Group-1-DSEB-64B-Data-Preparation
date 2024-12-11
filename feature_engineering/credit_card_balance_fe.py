import pandas as pd
import numpy as np
import gc

def credit_card_balance(credit):
    categorical_columns = [col for col in credit.columns if credit[col].dtype == 'object']
    credit = pd.get_dummies(credit, columns= categorical_columns, dtype=int)

    # Calculate Utilization Ratio
    credit['UTILIZATION_RATIO'] = credit['AMT_BALANCE']/credit['AMT_CREDIT_LIMIT_ACTUAL']
    credit.replace([np.inf, -np.inf], np.nan, inplace=True)  
    credit.fillna(0, inplace=True)

    # Calculate Credit Balance Differences
    credit = credit.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[True, False])
    credit.drop(['SK_ID_PREV','Unnamed: 0'], axis= 1, inplace = True)
    credit['AMT_BALANCE_diff'] = credit.groupby('SK_ID_CURR')['AMT_BALANCE'].diff(-1)
    credit['AMT_BALANCE_diff'].fillna(value= 0, inplace= True)

    # Calculate Payment Ratio
    credit['PAYMENT_RATIO'] = credit['AMT_PAYMENT_TOTAL_CURRENT']/credit['AMT_PAYMENT_CURRENT']
    credit['PAYMENT_RATIO'].fillna(value= 1, inplace= True)

    credit['SK_DPD'] = np.log1p(10*credit['SK_DPD'])
    credit['SK_DPD_DEF'] = np.log1p(10*credit['SK_DPD_DEF'])

    aggregations = {
        'MONTHS_BALANCE': ['min'],
        'AMT_BALANCE': ['min', 'max', 'mean'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
        'AMT_DRAWINGS_CURRENT': ['max', 'mean'],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_CURRENT': ['max', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean'],
        'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean'],
        'CNT_INSTALMENT_MATURE_CUM': ['max'],
        'AMT_BALANCE_diff': ['mean'],
        'PAYMENT_RATIO': ['mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'UTILIZATION_RATIO': ['mean']
    }

    onehot_columns = [col for col in credit.columns if any(cat in col for cat in categorical_columns)]
    for col in onehot_columns:
        aggregations[col] = ['mean']

    # General aggregations
    credit_agg = credit.groupby('SK_ID_CURR').agg(aggregations)
   
    credit_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in credit_agg.columns.tolist()])
    
    # Count Credit card lines
    credit_agg['CC_PAYMENT_RATIO_MEAN'] = credit_agg['CC_PAYMENT_RATIO_MEAN'].apply(lambda x: x if x > 0 else -1)
    credit_agg['CREDIT_COUNT'] = credit.groupby('SK_ID_CURR').size()
    credit_agg = credit_agg.reset_index()
    gc.collect()
    return credit_agg