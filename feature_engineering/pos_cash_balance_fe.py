import pandas as pd
import numpy as np
import gc

def pos_cash_balance(pos):
    # Feature transformations
    pos['SK_DPD'] = np.log1p(10*pos['SK_DPD'])
    pos['SK_DPD_DEF'] = np.log1p(10*pos['SK_DPD_DEF'])
    pos['INSTALMENT_RATIO'] = pos['CNT_INSTALMENT_FUTURE'] / pos['CNT_INSTALMENT']
    pos['IS_ACTIVE'] = (pos['MONTHS_BALANCE'] == pos.groupby('SK_ID_PREV')['MONTHS_BALANCE'].transform('max')) & (pos['NAME_CONTRACT_STATUS'] == 'Active')
    pos['HAS_DPD'] = (pos['SK_DPD'] > 0).astype(int)
    pos['HAS_DPD_DEF'] = (pos['SK_DPD_DEF'] > 0).astype(int)

    # Calculate Exponential Moving Average (EMA)
    ema_features = ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']
    ema_columns = ['EMA_' + col for col in ema_features]
    pos[ema_columns] = pos.groupby('SK_ID_PREV')[ema_features].transform(lambda x: x.ewm(alpha=0.6).mean())

    # One-hot encode categorical features
    categorical_columns = [col for col in pos.columns if pos[col].dtype == 'object']
    pos = pd.get_dummies(pos, columns=categorical_columns, dummy_na=False)
    categorical_cols = [col for col in pos.columns if col not in pos.columns]

    # Flag late payments
    pos['LATE_PAYMENT'] = pos['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)

    # Aggregate by SK_ID_CURR
    categorical_agg = {key: ['mean'] for key in categorical_cols}
    agg_funcs = {
        'MONTHS_BALANCE': ['min', 'max', 'std'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'std'],
        'CNT_INSTALMENT': ['mean', 'std'],
        'SK_DPD': ['max', 'mean', 'std'],
        'SK_DPD_DEF': ['max', 'mean', 'std'],
        'INSTALMENT_RATIO': ['mean', 'max'],
        'IS_ACTIVE': ['sum'],
        'HAS_DPD': ['sum'],
        'HAS_DPD_DEF': ['sum']
    }
    
    pos_agg = pos.groupby('SK_ID_CURR').agg({**agg_funcs, **categorical_agg})
    pos_agg.columns = pd.Index([f'POS_{e[0]}_{e[1].upper()}' for e in pos_agg.columns.tolist()])
    pos_agg = pos_agg.reset_index()

    # Sort and group by SK_ID_PREV
    sort_pos = pos.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])

    # Create new dataframe with additional features
    df = pd.DataFrame()
    gp = sort_pos.groupby('SK_ID_PREV')
    df['SK_ID_CURR'] = gp['SK_ID_CURR'].first()
    df['MONTHS_BALANCE_MAX'] = gp['MONTHS_BALANCE'].max()
    df['POS_LOAN_COMPLETED_MEAN'] = gp['NAME_CONTRACT_STATUS_Completed'].mean()
    
    df['POS_COMPLETED_BEFORE_MEAN'] = gp['CNT_INSTALMENT'].first() - gp['CNT_INSTALMENT'].last()
    df['POS_COMPLETED_BEFORE_MEAN'] = df.apply(
        lambda x: 1 if x['POS_COMPLETED_BEFORE_MEAN'] > 0 and x['POS_LOAN_COMPLETED_MEAN'] > 0 else 0, 
        axis=1
    )
    df['POS_REMAINING_INSTALMENTS'] = gp['CNT_INSTALMENT_FUTURE'].last()
    df['POS_REMAINING_INSTALMENTS_RATIO'] = gp['CNT_INSTALMENT_FUTURE'].last() / gp['CNT_INSTALMENT'].last()

    # Group by SK_ID_CURR and merge
    df_gp = df.groupby('SK_ID_CURR').sum().reset_index()
    df_gp.drop(['MONTHS_BALANCE_MAX'], axis=1, inplace=True)
    pos_agg = pd.merge(pos_agg, df_gp, on='SK_ID_CURR', how='left')

    # Calculate late payment ratio for last 3 loans
    late_payments = pos.groupby('SK_ID_PREV')['LATE_PAYMENT'].sum().reset_index(name='LATE_PAYMENT_SUM')
    late_payments = late_payments.merge(
        sort_pos[['SK_ID_PREV', 'SK_ID_CURR']].drop_duplicates(), 
        on='SK_ID_PREV'
    )

    # Get last month of each loan
    last_month_df = sort_pos.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()

    # Last 3 loans
    latest_loans = sort_pos.iloc[last_month_df].groupby('SK_ID_CURR').tail(3)

    # Calculate mean for last 3 loans
    gp_mean = latest_loans.groupby('SK_ID_CURR')['LATE_PAYMENT'].mean().reset_index(name='LATE_PAYMENT_MEAN')
    pos_agg = pd.merge(pos_agg, gp_mean, on='SK_ID_CURR', how='left')

    gc.collect()
    return pos_agg