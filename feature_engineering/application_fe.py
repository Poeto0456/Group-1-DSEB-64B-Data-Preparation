import pandas as pd
import numpy as np
import gc

def application(application):
    """
    Process the application dataset and return a feature-engineered dataframe.

    Parameters:
    -----------
    application : pandas.DataFrame
        Cleaned application dataset.

    Returns:
    --------
    pandas.DataFrame
        Feature-engineered dataset with aggregations and derived features.
    """
    # Step 1: One-Hot Encoding for Categorical Features
    categorical_columns = application.select_dtypes(include=['object', 'category']).columns.tolist()
    application_encoded = pd.get_dummies(
        application,
        columns=categorical_columns,
        prefix_sep='_',
        dummy_na=False
    )

    # Convert boolean columns to integer (0 and 1)
    bool_columns = application_encoded.select_dtypes(include=['bool']).columns
    application_encoded[bool_columns] = application_encoded[bool_columns].astype(int)

    # Step 2: Advanced Feature Engineering
    application_encoded['INCOME_PER_PERSON'] = (
        application_encoded['AMT_INCOME_TOTAL'] / 
        (application_encoded['CNT_FAM_MEMBERS'] + 1e-8)
    )
    application_encoded['ANNUITY_INCOME_RATIO'] = (
        application_encoded['AMT_ANNUITY'] / 
        (application_encoded['AMT_INCOME_TOTAL'] + 1e-8)
    )
    application_encoded['CREDIT_INCOME_RATIO'] = (
        application_encoded['AMT_CREDIT'] / 
        (application_encoded['AMT_INCOME_TOTAL'] + 1e-8)
    )
    application_encoded['EMPLOYMENT_TO_AGE_RATIO'] = (
        application_encoded['DAYS_EMPLOYED'] / 
        (application_encoded['DAYS_BIRTH'] + 1e-8)
    )

    # Disposable income ratio after deducting loan payments
    application_encoded['DISPOSABLE_INCOME_RATIO'] = (
        (application_encoded['AMT_INCOME_TOTAL'] - application_encoded['AMT_ANNUITY']) /
        (application_encoded['AMT_INCOME_TOTAL'] + 1e-8)
    )

    # Estimated years required to repay the loan in full
    application_encoded['YEARS_TO_REPAY'] = (
        application_encoded['AMT_CREDIT'] /
        (application_encoded['AMT_ANNUITY'] + 1e-8)
    )

    # Total credit cost (loan + annuity) relative to income
    application_encoded['CREDIT_PLUS_ANNUITY_RATIO'] = (
        (application_encoded['AMT_CREDIT'] + application_encoded['AMT_ANNUITY']) /
        (application_encoded['AMT_INCOME_TOTAL'] + 1e-8)
    )

    # Loan-to-asset ratio
    application_encoded['CREDIT_TO_ASSET_RATIO'] = (
        application_encoded['AMT_CREDIT'] /
        (application_encoded['AMT_GOODS_PRICE'] + 1e-8)
    )

    # Average income per family member
    application_encoded['INCOME_PER_FAMILY_MEMBER'] = (
        application_encoded['AMT_INCOME_TOTAL'] /
        (application_encoded['CNT_FAM_MEMBERS'] + 1e-8)
    )

    # Total credit per family member
    application_encoded['CREDIT_PER_FAMILY_MEMBER'] = (
        application_encoded['AMT_CREDIT'] /
        (application_encoded['CNT_FAM_MEMBERS'] + 1e-8)
    )

    # Time since last registration relative to age
    application_encoded['REGISTRATION_TO_AGE_RATIO'] = (
        application_encoded['DAYS_REGISTRATION'] /
        (application_encoded['DAYS_BIRTH'] + 1e-8)
    )

    application_encoded['AGE_YEARS'] = -application_encoded['DAYS_BIRTH'] / 365
    application_encoded['EMPLOYMENT_YEARS'] = (
        -application_encoded['DAYS_EMPLOYED'] / 365
    ).clip(lower=0)
    bins = [21, 30, 40, 50, 60, 69]  
    labels = [1, 2, 3, 4, 5] 
    application_encoded['AGE_GROUP'] = pd.cut(application_encoded['AGE_YEARS'], bins=bins, labels=labels, right=True)
    # Fill NaN with default value before converting to int
    application_encoded['AGE_GROUP'] = application_encoded['AGE_GROUP'].cat.add_categories([0]).fillna(0).astype(int)
    # Convert the AGE_GROUP column to integer
    application_encoded['AGE_GROUP'] = application_encoded['AGE_GROUP'].astype(int)

    # Demographic Features
    # Age ratio compared to expected retirement age (assuming retirement age is 60)
    application_encoded['AGE_RATIO_TO_RETIREMENT'] = (
        application_encoded['DAYS_BIRTH'] / -365 / 60
    )

    # Time from birth to application registration date
    application_encoded['DAYS_BIRTH_TO_APP'] = (
        application_encoded['DAYS_BIRTH'] - application_encoded['DAYS_REGISTRATION']
    )

    # Ratio between income and total loan amount
    application_encoded['INCOME_TO_CREDIT_RATIO'] = (
        application_encoded['AMT_INCOME_TOTAL'] /
        (application_encoded['AMT_CREDIT'] + 1e-8)
    )

    # Ratio between income and number of family members
    application_encoded['INCOME_PER_FAMILY_MEMBER'] = (
        application_encoded['AMT_INCOME_TOTAL'] /
        (application_encoded['CNT_FAM_MEMBERS'] + 1e-8)
    )

    # Family-related features
    # Number of dependents divided by total family members
    application_encoded['DEPENDENTS_TO_FAMILY_RATIO'] = (
        application_encoded['CNT_CHILDREN'] /
        (application_encoded['CNT_FAM_MEMBERS'] + 1e-8)
    )

    # Loan amount per family member
    application_encoded['CREDIT_PER_FAMILY_MEMBER'] = (
        application_encoded['AMT_CREDIT'] /
        (application_encoded['CNT_FAM_MEMBERS'] + 1e-8)
    )

    # Monthly annuity per dependent
    application_encoded['ANNUITY_PER_DEPENDENT'] = (
        application_encoded['AMT_ANNUITY'] /
        (application_encoded['CNT_CHILDREN'] + 1e-8)
    )

    # External Source Features
    # Average credit risk score from external sources
    application_encoded['EXT_SOURCE_MEAN'] = application_encoded[[
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]].mean(axis=1)

    # Standard deviation of the credit risk score
    application_encoded['EXT_SOURCE_STD'] = application_encoded[[
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]].std(axis=1)

    # Weighted integration of credit risk scores
    application_encoded['EXT_SOURCE_WEIGHTED'] = (
        0.5 * application_encoded['EXT_SOURCE_1'] +
        0.3 * application_encoded['EXT_SOURCE_2'] +
        0.2 * application_encoded['EXT_SOURCE_3']
    )

    # Ratio between average credit score and loan amount
    application_encoded['EXT_SOURCE_TO_CREDIT_RATIO'] = (
        application_encoded['EXT_SOURCE_MEAN'] /
        (application_encoded['AMT_CREDIT'] + 1e-8)
    )

    # Normalized credit risk score based on loan amount
    application_encoded['EXT_SOURCE_NORMALIZED_CREDIT'] = (
        application_encoded['EXT_SOURCE_MEAN'] *
        application_encoded['AMT_CREDIT']
    )

    # Ratio between average credit score and annual income
    application_encoded['EXT_SOURCE_TO_INCOME_RATIO'] = (
        application_encoded['EXT_SOURCE_MEAN'] /
        (application_encoded['AMT_INCOME_TOTAL'] + 1e-8)
    )

    # Credit risk score compared to annual annuity payment
    application_encoded['EXT_SOURCE_TO_ANNUITY_RATIO'] = (
        application_encoded['EXT_SOURCE_MEAN'] /
        (application_encoded['AMT_ANNUITY'] + 1e-8)
    )

    # Credit score multiplied by registration days
    application_encoded['EXT_SOURCE_REGISTRATION'] = (
        application_encoded['EXT_SOURCE_MEAN'] *
        abs(application_encoded['DAYS_REGISTRATION'])
    )

    # Credit score divided by days since birth
    application_encoded['EXT_SOURCE_DAYS_BIRTH'] = (
        application_encoded['EXT_SOURCE_MEAN'] /
        abs(application_encoded['DAYS_BIRTH'] + 1e-8)
    )

    # Maximum difference between credit risk scores from different sources
    application_encoded['EXT_SOURCE_RANGE'] = (
        application_encoded[[
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
        ]].max(axis=1) - 
        application_encoded[[
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
        ]].min(axis=1)
    )

    # Correlation coefficient between the two most popular credit risk sources
    application_encoded['EXT_SOURCE_CORRELATION'] = (
        application_encoded['EXT_SOURCE_1'] *
        application_encoded['EXT_SOURCE_2']
    )

    application_encoded['EXT_SOURCE_1^2'] = application_encoded['EXT_SOURCE_1']**2
    application_encoded['EXT_SOURCE_2^2'] = application_encoded['EXT_SOURCE_2']**2
    application_encoded['EXT_SOURCE_3^2'] = application_encoded['EXT_SOURCE_3']**2

    # Ratios involving EXT_SOURCE_1
    application_encoded['SCORE1_TO_FAM_CNT_RATIO'] = application_encoded['EXT_SOURCE_1'] / (application_encoded['CNT_FAM_MEMBERS'] + 1)
    application_encoded['SCORE1_TO_GOODS_RATIO'] = application_encoded['EXT_SOURCE_1'] / (application_encoded['AMT_GOODS_PRICE'] + 1)
    application_encoded['SCORE1_TO_CREDIT_RATIO'] = application_encoded['EXT_SOURCE_1'] / (application_encoded['AMT_CREDIT'] + 1)
    application_encoded['SCORE1_TO_SCORE2_RATIO'] = application_encoded['EXT_SOURCE_1'] / (application_encoded['EXT_SOURCE_2'] + 1e-9)
    application_encoded['SCORE1_TO_SCORE3_RATIO'] = application_encoded['EXT_SOURCE_1'] / (application_encoded['EXT_SOURCE_3'] + 1e-9)

    # Ratios involving EXT_SOURCE_2
    application_encoded['SCORE2_TO_CREDIT_RATIO'] = application_encoded['EXT_SOURCE_2'] / (application_encoded['AMT_CREDIT'] + 1)
    application_encoded['SCORE2_TO_REGION_RATING_RATIO'] = application_encoded['EXT_SOURCE_2'] / (application_encoded['REGION_RATING_CLIENT'] + 1)
    application_encoded['SCORE2_TO_CITY_RATING_RATIO'] = application_encoded['EXT_SOURCE_2'] / (application_encoded['REGION_RATING_CLIENT_W_CITY'] + 1)
    application_encoded['SCORE2_TO_POP_RATIO'] = application_encoded['EXT_SOURCE_2'] / (application_encoded['REGION_POPULATION_RELATIVE'] + 1e-9)
    application_encoded['SCORE2_TO_PHONE_CHANGE_RATIO'] = application_encoded['EXT_SOURCE_2'] / (abs(application_encoded['DAYS_LAST_PHONE_CHANGE']) + 1)

    # Ratios involving EXT_SOURCE_3
    application_encoded['SCORE3_TO_CREDIT_RATIO'] = application_encoded['EXT_SOURCE_3'] / (application_encoded['AMT_CREDIT'] + 1)
    application_encoded['SCORE3_TO_GOODS_RATIO'] = application_encoded['EXT_SOURCE_3'] / (application_encoded['AMT_GOODS_PRICE'] + 1)
    application_encoded['SCORE3_TO_FAM_CNT_RATIO'] = application_encoded['EXT_SOURCE_3'] / (application_encoded['CNT_FAM_MEMBERS'] + 1)
    application_encoded['SCORE3_TO_REGION_RATING_RATIO'] = application_encoded['EXT_SOURCE_3'] / (application_encoded['REGION_RATING_CLIENT'] + 1)
    application_encoded['SCORE3_TO_CITY_RATING_RATIO'] = application_encoded['EXT_SOURCE_3'] / (application_encoded['REGION_RATING_CLIENT_W_CITY'] + 1)

    # Interaction Features
    application_encoded['SCORE1_SCORE2_PRODUCT'] = application_encoded['EXT_SOURCE_1'] * application_encoded['EXT_SOURCE_2']
    application_encoded['SCORE2_SCORE3_PRODUCT'] = application_encoded['EXT_SOURCE_2'] * application_encoded['EXT_SOURCE_3']
    application_encoded['SCORE1_SCORE3_PRODUCT'] = application_encoded['EXT_SOURCE_1'] * application_encoded['EXT_SOURCE_3']
    application_encoded['CREDIT_TO_GOODS_RATIO'] = application_encoded['AMT_CREDIT'] / (application_encoded['AMT_GOODS_PRICE'] + 1)
    application_encoded['CREDIT_TO_INCOME_RATIO'] = application_encoded['AMT_CREDIT'] / (application_encoded['AMT_INCOME_TOTAL'] + 1)

    # Differences
    application_encoded['CREDIT_MINUS_GOODS'] = application_encoded['AMT_CREDIT'] - application_encoded['AMT_GOODS_PRICE']
    application_encoded['INCOME_MINUS_CREDIT'] = application_encoded['AMT_INCOME_TOTAL'] - application_encoded['AMT_CREDIT']
    application_encoded['GOODS_MINUS_SCORE3'] = application_encoded['AMT_GOODS_PRICE'] - application_encoded['EXT_SOURCE_3']

    # Log Transformations
    application_encoded['LOG_CREDIT'] = np.log1p(application_encoded['AMT_CREDIT'])
    application_encoded['LOG_GOODS_PRICE'] = np.log1p(application_encoded['AMT_GOODS_PRICE'])
    application_encoded['LOG_INCOME'] = np.log1p(application_encoded['AMT_INCOME_TOTAL'])
    
    # Step 3: Aggregation of Relevant Columns (if grouping is required)
    # If you need aggregations similar to `bureau`, specify grouping here
    # For example, group by SK_ID_CURR if `application` is linked to multiple loans
    agg_dict = {
        'AMT_CREDIT': ['mean', 'sum', 'max'],
        'AMT_ANNUITY': ['mean', 'sum', 'max'],
        'AMT_INCOME_TOTAL': ['mean', 'sum', 'max'],
        # Add more numerical aggregations as needed
    }
    application_agg = application_encoded.groupby('SK_ID_CURR').agg(agg_dict)
    application_agg.columns = [f"APPLICATION_{col[0]}_{col[1].upper()}" for col in application_agg.columns]
    application_agg = application_agg.reset_index()
    
    # Merge aggregated features with the original feature-engineered dataframe
    final_df = application_encoded.merge(
        application_agg,
        on='SK_ID_CURR',
        how='left'
    )

    gc.collect()
    return final_df
