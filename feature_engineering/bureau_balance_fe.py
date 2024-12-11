import pandas as pd
import gc

def bureau_balance(df):
    """
    Process bureau balance dataframe and return a preprocessed pandas dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed bureau balance DataFrame
    
    Returns:
    --------
    pandas.DataFrame
        Processed bureau balance features
    """
    # Xác định các cột categorical
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'SK_ID_BUREAU']
    
    # Thực hiện one-hot encoding
    df_encoded = pd.get_dummies(df, 
                                columns=categorical_columns, 
                                prefix_sep='_', 
                                dummy_na=False)
    
    # Các aggregation cho MONTHS_BALANCE và các feature one-hot
    agg_dict = {
        'MONTHS_BALANCE': [
            'min',     # Giá trị nhỏ nhất 
            'max',     # Giá trị lớn nhất
            'mean',    # Giá trị trung bình
            'count'
        ]
    }
    
    # Thêm aggregation cho các cột one-hot
    onehot_columns = [col for col in df_encoded.columns if any(cat in col for cat in categorical_columns)]
    for col in onehot_columns:
        agg_dict[col] = ['mean']
    
    # Thực hiện groupby và aggregation
    bb_processed = df_encoded.groupby('SK_ID_BUREAU').agg(agg_dict)
    
    # Điều chỉnh tên cột
    bb_processed.columns = [f'{col[0]}_{col[1].upper()}' if col[1] else col[0] 
                             for col in bb_processed.columns]
    
    # Reset index để đưa SK_ID_BUREAU trở lại như một cột
    bb_processed = bb_processed.reset_index()
    
    # Tính tỷ lệ thanh toán trễ
    late_cols = [col for col in bb_processed.columns if any(x in col for x in ['STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5'])]
    if late_cols:
        bb_processed['LATE_PAYMENT_RATIO'] = bb_processed[late_cols].sum(axis=1) / bb_processed['MONTHS_BALANCE_COUNT']
        
    gc.collect()
    return bb_processed

print("Done!")