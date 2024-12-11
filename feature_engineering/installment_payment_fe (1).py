import pandas as pd
import numpy as np
import gc


def installments_payments(ins):
    ins = ins.sort_values(by = ['SK_ID_PREV','SK_ID_CURR'])
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['PAYMENT_PERC'] = ins['PAYMENT_PERC'].apply(lambda x: x if x < 1 else 1)
    ins['PAYMENT_DIFF'] = ins['PAYMENT_DIFF'].apply(lambda x: x if x > 0 else 0 )
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    ins['RATE_LATE_PAYMENT'] = np.log1p(ins['DPD']*10)
    ins['PAYMENT_TREND'] = ins['AMT_PAYMENT'].pct_change()
    ins.replace([np.inf, -np.inf], np.nan, inplace=True)
    ins.fillna(0, inplace = True)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'RATE_LATE_PAYMENT':['mean','max'],
        'PAYMENT_PERC': ['mean','max'],
        'PAYMENT_DIFF': ['max', 'mean'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
        'PAYMENT_TREND':['max','mean','sum']
        
    }
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    ins_agg = ins_agg.reset_index()
    gc.collect()
    return ins_agg