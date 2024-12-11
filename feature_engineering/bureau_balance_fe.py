import pandas as pd
import gc

def process_bureau_balance(bureau_df):
    """
    Process the bureau balance DataFrame and return preprocessed features.
    
    Parameters:
    -----------
    bureau_df : pandas.DataFrame
        Original bureau balance DataFrame
    
    Returns:
    --------
    pandas.DataFrame
        Processed bureau balance features
    """
    # Identify categorical columns excluding 'SK_ID_BUREAU'
    categorical_cols = [col for col in bureau_df.columns if bureau_df[col].dtype == 'object' and col != 'SK_ID_BUREAU']
    
    # Perform one-hot encoding
    bureau_encoded = pd.get_dummies(bureau_df, 
                                    columns=categorical_cols, 
                                    prefix_sep='_', 
                                    dummy_na=False)
    
    # Define aggregation dictionary
    agg_dict = {
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'count']
    }
    
    # Add aggregations for one-hot encoded columns
    one_hot_features = [col for col in bureau_encoded.columns if any(cat in col for cat in categorical_cols)]
    for col in one_hot_features:
        agg_dict[col] = ['mean']
    
    # Perform groupby and aggregation
    bureau_aggregated = bureau_encoded.groupby('SK_ID_BUREAU').agg(agg_dict)
    
    # Adjust column names
    bureau_aggregated.columns = [f'{col[0]}_{col[1].upper()}' if col[1] else col[0] 
                                 for col in bureau_aggregated.columns]
    
    # Reset index to include 'SK_ID_BUREAU'
    bureau_aggregated = bureau_aggregated.reset_index()
    
    # Calculate late payment ratio if relevant columns exist
    late_payment_cols = [col for col in bureau_aggregated.columns if any(status in col for status in ['STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5'])]
    if late_payment_cols:
        bureau_aggregated['LATE_PAYMENT_RATIO'] = bureau_aggregated[late_payment_cols].sum(axis=1) / bureau_aggregated['MONTHS_BALANCE_COUNT']
    
    gc.collect()
    return bureau_aggregated