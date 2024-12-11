import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def merge_all(app, feature_bureau, feature_prev, feature_pos, feature_credit, feature_install):    
    app = app.merge(feature_bureau, how='left', on='SK_ID_CURR')
    print(f"After merging bureau: {app.shape}")
    
    app = app.merge(feature_prev, how='left', on='SK_ID_CURR')
    print(f"After merging previous application: {app.shape}")

    app = app.merge(feature_pos, how='left', on='SK_ID_CURR')
    print(f"After merging POS CASH balance: {app.shape}")

    app = app.merge(feature_credit, how='left', on='SK_ID_CURR')
    print(f"After merging credit card balance: {app.shape}")

    app = app.merge(feature_install, how='left', on='SK_ID_CURR')
    print(f"After merging installments payments: {app.shape}")

   

    # After all merges, fill missing values
    print("Filling missing values...")

    # Using SimpleImputer to fill missing values with 0 for all columns
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    
    # Apply imputer to the entire DataFrame (all columns)
    app_imputed = imputer.fit_transform(app)
    
    # Convert the result back to DataFrame and keep original column names
    app_imputed_df = pd.DataFrame(app_imputed, columns=app.columns)

    print("After filling missing values:")
    print(app_imputed_df.isnull().sum())  # Check if any missing values are left
    print(app_imputed_df.shape)

    return app_imputed_df

# Feature Importance Function
def get_important_features(X, y, threshold=0.0001):
    # Train a Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    # Get feature importances
    feature_importances = rf.feature_importances_

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    # Filter features with importance above the threshold
    important_features = importance_df[importance_df["Importance"] > threshold]
    selected_features = important_features["Feature"].tolist()

    print(f"Selected {len(selected_features)} important features out of {X.shape[1]}")
    
    return selected_features

def process_and_select_features(X_train, y_train, X_test=None, threshold=0.0001):
    """
    Selects important features based on a feature importance threshold.

    Parameters:
    - X_train (pd.DataFrame): Training feature set.
    - y_train (pd.Series): Training target column.
    - X_test (pd.DataFrame, optional): Test feature set. Default is None.
    - threshold (float): Minimum importance score to keep a feature.

    Returns:
    - X_train_reduced (pd.DataFrame): Training set with reduced features.
    - X_test_reduced (pd.DataFrame or None): Test set with reduced features (if provided).
    """
    if y_train is not None:
        # Get important features
        important_features = get_important_features(X_train, y_train, threshold=threshold)

        # Filter training dataset
        X_train_reduced = X_train[important_features]
        print(f"Reduced training dataset shape: {X_train_reduced.shape}")

        # If test set is provided, filter it using the same important features
        if X_test is not None:
            X_test_reduced = X_test[important_features]
            print(f"Reduced test dataset shape: {X_test_reduced.shape}")
            return X_train_reduced, X_test_reduced, y_train

        return X_train_reduced, None, y_train
    else:
        print("Target column 'y_train' not found or is None.")
        return X_train, X_test, None