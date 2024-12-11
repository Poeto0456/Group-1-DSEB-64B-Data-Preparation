import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.stats import chi2_contingency
from scipy.stats import zscore
from scipy.stats import ttest_ind
from sklearn.feature_selection import VarianceThreshold
from IPython.display import display

def display_dataframe_info(df):
    """
    Display basic information about the DataFrame and plot column classification.

    Parameters:
    - df: The DataFrame to inspect.

    Outputs:
    - Basic information table.
    - Column classification plot.
    """

    # Identify column types
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Create an information DataFrame
    info_summary = {
        "Attribute": ["Shape", "Numerical Columns", "Categorical Columns", "Data Types"],
        "Value": [
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}",
            "\n".join(numerical_cols) if numerical_cols else "None",
            "\n".join(categorical_cols) if categorical_cols else "None",
            "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()]),
        ],
    }
    info_df = pd.DataFrame(info_summary)

    # Column classification plot
    col_types = {
        "Numerical": len(numerical_cols),
        "Categorical": len(categorical_cols),
        "Time-series": 0,  # Can be extended if there are time columns
    }
    
    # Set color palette
    sns.set_palette("coolwarm")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(col_types.keys(), col_types.values(), color=sns.color_palette("coolwarm"))
    ax.set_title("Column Classification in DataFrame", fontsize=14)
    ax.set_ylabel("Number of Columns", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Display the information with a nice format
    display(info_df.style.set_properties(**{
        'background-color': '#f9f9f9', 
        'color': '#333', 
        'border-color': '#ccc',
        'text-align': 'left',
        'padding': '10px'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f2f2f2'), 
                                     ('font-weight', 'bold'), 
                                     ('text-align', 'center')]}]))
    
    plt.show()


def analyze_missing_values(df, row_threshold=0.5):
    """
    Analyze missing values in the DataFrame.

    Parameters:
    - df: The DataFrame to inspect.
    - row_threshold: Threshold to identify rows with many missing values (default: 50%).

    Outputs:
    - Table of missing values count and percentage for each column.
    - Rows with high missing values.
    - Visualization plot.
    """

    # Calculate missing values count and percentage
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100
    
    # Create a summary DataFrame
    missing_df = pd.DataFrame({
        "Column": missing_counts.index,
        "Missing Count": missing_counts.values,
        "Missing Percentage (%)": missing_percentage.values.round(2)
    }).sort_values(by="Missing Percentage (%)", ascending=False)

    # Display the summary table
    display(missing_df.style.background_gradient(cmap="YlGnBu", subset=["Missing Percentage (%)"])
            .format({"Missing Percentage (%)": "{:.2f}%"}))

    # Check rows with high missing values
    rows_missing = df.isnull().mean(axis=1)
    high_missing_rows = df[rows_missing >= row_threshold]

    if not high_missing_rows.empty:
        print(f"There are {len(high_missing_rows)} rows with more than {row_threshold * 100:.0f}% missing values.")
        display(high_missing_rows.head())

    # Plot missing values
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=missing_df, 
        x="Missing Percentage (%)", 
        y="Column", 
        palette="coolwarm"
    )
    plt.title("Missing Value Analysis by Column", fontsize=14)
    plt.xlabel("Missing Percentage (%)", fontsize=12)
    plt.ylabel("Column", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()


def basic_statistics_summary(df):
    """
    Display basic descriptive statistics of the DataFrame.

    Parameters:
    - df: The DataFrame to analyze.

    Outputs:
    - Summary statistics table from describe().
    """
    # Generate descriptive statistics using describe()
    stats_df = df.describe().transpose()

    # Style the statistics table
    styled_stats = stats_df.style.set_properties(**{
        'background-color': '#f9f9f9',  # Light background
        'color': '#333',                # Dark text color
        'border-color': '#ccc',         # Table border
        'text-align': 'center',         # Center text
        'padding': '10px'               # Content padding
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#f2f2f2'),  # Header background
            ('font-weight', 'bold'),          # Bold header
            ('text-align', 'center'),         # Center header text
        ]},
    ]).format(precision=2)  # Round to 2 decimal places

    # Display the table
    display(styled_stats)


# PHASE 2
def visualize_cate_with_rare_and_binary(df, threshold):
    # Identify categorical columns (object, category) and binary columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    binary_columns = [col for col in df.columns if df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1})]

    # Combine categorical and binary columns
    all_categorical_columns = list(categorical_columns) + binary_columns

    for column in all_categorical_columns:
        plt.figure(figsize=(10, 6))

        # Calculate frequency and percentages
        value_counts = df[column].value_counts()
        percentages = (value_counts / len(df) * 100).round(2)
        rare_categories = percentages[percentages < threshold]

        # Plot the bar chart
        value_counts.plot(kind='bar', color='darkblue', edgecolor='black')

        # Add title and labels
        plt.title(f'Count of {column}', fontsize=16)
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Format y-axis as integers
        def format_func(value, tick_number):
            return f"{int(value):,}" 
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

        # Display rare categories as a note
        if not rare_categories.empty:
            rare_note = f"Rare Categories (< {threshold}%):\n" + \
                        "\n".join([f"{cat}: {freq:.2f}%" for cat, freq in rare_categories.items()])
            
            plt.text(
                1.05, 0.5,
                rare_note,
                transform=plt.gca().transAxes,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
            )
        
        plt.show()

# PHASE 3
def merge_target(data, app, key_col='SK_ID_CURR', target_col='TARGET'):
    """
    Merges the TARGET column from app_train into the data dataframe.

    Parameters:
        data (DataFrame): The dataframe to which the TARGET column will be added.
        app_train (DataFrame): The dataframe containing the TARGET column.
        key_col (str): The key column used for the merge (default: 'SK_ID_CURR').
        target_col (str): The target column name in app_train (default: 'TARGET').

    Returns:
        DataFrame: The data dataframe with the TARGET column merged.
    """
    if key_col not in data.columns:
        raise KeyError(f"'{key_col}' not found in data columns.")
    if key_col not in app.columns or target_col not in app.columns:
        raise KeyError(f"'{key_col}' or '{target_col}' not found in app_train columns.")
    
    merged_data = data.merge(app[[key_col, target_col]], on=key_col, how='left')
    print(f"Successfully merged '{target_col}' column into the data.")
    return merged_data


# PHASE 4
def checking_outlier(df, target=None, outlier_threshold=3):    
    """
    Detects and handles outliers using Z-score method, excluding 'SK_ID_' columns.

    Parameters:
        df (DataFrame): The dataset.
        target (str): Target variable to exclude from outlier detection.
        outlier_threshold (float): Z-score threshold for detecting outliers.

    Returns:
        None: Displays boxplots and outlier counts.
    """
    print("\n=== Outlier Detection and Handling ===")

    numeric_cols = [col for col in df.select_dtypes(include='number').columns if not col.startswith('SK_ID_')]
    
    for col in numeric_cols:
        if col != target: 
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

            z_scores = zscore(df[col].dropna())
            outliers = np.abs(z_scores) > outlier_threshold
            
            if outliers.sum() > 0:
                print(f"Outliers detected in column {col}: {outliers.sum()} rows")

                lower_limit = np.percentile(df[col].dropna(), 1)
                upper_limit = np.percentile(df[col].dropna(), 99)
                df[col] = np.clip(df[col], lower_limit, upper_limit)
            else:
                print(f"No outliers detected in column {col}.")


def checking_imbalance(df, target='TARGET', imbalance_threshold=0.3, top_n=5):
    """
    Kiểm tra các cột phân loại bị mất cân bằng trong dataframe (trừ cột target).

    Args:
        df (pd.DataFrame): Dataframe cần kiểm tra.
        target (str, optional): Tên cột mục tiêu, nếu có, sẽ bỏ qua cột này. Mặc định là 'TARGET'.
        imbalance_threshold (float): Ngưỡng để xác định mất cân bằng. Mặc định là 0.1 (10%).
        top_n (int): Số lượng feature tối đa hiển thị. Mặc định là 5.

    Returns:
        None
    """
    print("\n=== Checking for Imbalanced Columns ===")
    
    # Chuyển các cột int có giá trị duy nhất là 0 và 1 thành category
    binary_cols = [col for col in df.select_dtypes(include=['int', 'float']).columns
                   if df[col].dropna().nunique() == 2]
    df[binary_cols] = df[binary_cols].astype('category')
    
    # Lấy danh sách các cột phân loại
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['SK_ID_CURR', 'SK_ID_PREV', 'TARGET']]
    if target in categorical_cols:
        categorical_cols.remove(target)
    
    # Duyệt qua từng cột để kiểm tra sự mất cân bằng
    for col in categorical_cols:
        class_distribution = df[col].value_counts(normalize=True)
        max_class_ratio = class_distribution.max()
        
        # Xác định cột mất cân bằng
        if max_class_ratio > 1 - imbalance_threshold:
            print(f"\nColumn '{col}' is IMBALANCED (Max class proportion: {max_class_ratio:.2%}):")
            print(class_distribution)
            
            # Vẽ biểu đồ thanh thể hiện phân phối
            top_features = class_distribution.nlargest(top_n)
            plt.figure(figsize=(8, 4))
            sns.barplot(x=top_features.index, y=top_features.values, palette="coolwarm")
            plt.title(f"Class Distribution in '{col}' (Top {top_n})")
            plt.xlabel("Classes")
            plt.ylabel("Proportion")
            plt.show()
        else:
            print(f"Column '{col}' is BALANCED (Max class proportion: {max_class_ratio:.2%}).")
            

def low_variance(df, variance_threshold=0.01):
    """
    Detects and visualizes low-variance features in a DataFrame.

    Parameters:
        df (DataFrame): The dataset to analyze.
        variance_threshold (float): Minimum variance to keep a feature.

    Returns:
        list: Names of low-variance columns.
    """
    print("\n=== Low-Variance Feature Detection ===")

    numeric_cols = [col for col in df.select_dtypes(include='number').columns if not col.startswith('SK_ID_')]
    
    if numeric_cols: 
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(df[numeric_cols])
        variances = selector.variances_

        low_variance_cols = [col for col, var in zip(numeric_cols, variances) if var < variance_threshold]

        # Biểu đồ bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x=numeric_cols, y=variances, palette="viridis")
        plt.axhline(variance_threshold, color='red', linestyle='--', label='Threshold')
        plt.xticks(rotation=90)
        plt.title("Feature Variances")
        plt.xlabel("Features")
        plt.ylabel("Variance")
        plt.legend()
        plt.show()

        if low_variance_cols:
            print(f"Low-variance columns to be removed: {low_variance_cols}")
        else:
            print("No low-variance columns detected.")
    else:
        print("No numeric columns detected for low-variance check.")
    
    return low_variance_cols
        

def correlation_matrix(data, method='pearson', threshold=0.5, figsize=(12, 8)):
    """
    Generates a correlation matrix heatmap for numerical variables, excluding columns starting with 'SK_ID_'.

    Parameters:
        data (DataFrame): The dataset.
        method (str): Correlation method ('pearson', 'spearman', or 'kendall').
        threshold (float): Highlight correlations above this absolute value.
        figsize (tuple): Size of the heatmap.

    Returns:
        None: Displays the heatmap and prints strong correlations.
    """
    # Select only numerical columns excluding SK_ID_ prefix
    numeric_data = data.select_dtypes(include=['number']).drop(
        columns=[col for col in data.columns if col.startswith('SK_ID_')], 
        errors='ignore'
    )

    # Compute the correlation matrix
    corr = numeric_data.corr(method=method)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, vmax=1,
        cbar_kws={'label': f'{method.capitalize()} Correlation'},
        mask=np.triu(np.ones_like(corr, dtype=bool))
    )
    plt.title("Correlation Matrix")
    plt.show()

    # Extract strong correlations above threshold
    strong_corrs = (
        corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))  # Upper triangle
        .stack()
        .reset_index()
    )
    strong_corrs.columns = ['Variable 1', 'Variable 2', 'Correlation']
    strong_corrs = strong_corrs[strong_corrs['Correlation'].abs() > threshold]

    print("\nStrong correlations above threshold:")
    if not strong_corrs.empty:
        print(strong_corrs.sort_values(by='Correlation', ascending=False))
    else:
        print("No correlations found above the threshold.")


def analyze_numeric_relationships(data, sample_frac=1, figsize=(12, 6), min_unique=10):
    """
    Visualizes relationships between numerical variables with sufficient unique values using pairplots and distribution plots.

    Parameters:
        data (DataFrame): The dataset.
        sample_frac (float): Fraction of the dataset to sample for visualization.
        figsize (tuple): Size of each plot.
        min_unique (int): Minimum number of unique values required for a column to be considered.

    Returns:
        None: Displays visualizations.
    """
    # Sample the data
    sample_data = data.sample(frac=sample_frac, random_state=42) if len(data) > 10000 else data
    
    # Filter numeric columns with sufficient unique values
    numeric_cols = sample_data.select_dtypes(include=['number']).columns
    diverse_cols = [col for col in numeric_cols if sample_data[col].nunique() >= min_unique]

    if diverse_cols:
        print(f"\nAnalyzing the following numerical variables with at least {min_unique} unique values:")
        print(diverse_cols)
        
        # Pairplot for selected numerical variables
        if len(diverse_cols) > 1:
            print("\nGenerating pairplot for numerical variables...")
            sns.pairplot(sample_data[diverse_cols], diag_kind='kde', corner=True, plot_kws={'alpha': 0.6}, height=2.5)
            plt.show()

        # Distribution plot for each numerical variable
        for col in diverse_cols:
            plt.figure(figsize=figsize)
            sns.histplot(sample_data[col], kde=True, color="skyblue", bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
    else:
        print(f"No numerical columns with at least {min_unique} unique values were found.")


def analyze_target_relationship(data, target_col, group_cols, sample_frac=1, figsize=(8, 6)):
    # Sample more than 10,000
    sample_data = data.sample(frac=sample_frac, random_state=42) if len(data) > 10000 else data

    numeric_cols = [col for col in group_cols if sample_data[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in group_cols if col not in numeric_cols]

    # Numerical columns
    for col in numeric_cols:
        plt.figure(figsize=figsize)
        sns.scatterplot(x=col, y=target_col, data=sample_data, alpha=0.6, color="blue")
        plt.title(f"Scatterplot of {target_col} vs {col}")
        plt.xticks(rotation=45)
        plt.show()

        # Check for binary target variable
        if sample_data[target_col].nunique() == 2:
            group1 = sample_data[sample_data[target_col] == sample_data[target_col].unique()[0]][col].dropna()
            group2 = sample_data[sample_data[target_col] == sample_data[target_col].unique()[1]][col].dropna()
            
            # Skip if insufficient data
            if len(group1) == 0 or len(group2) == 0:
                print(f"Insufficient data in one of the groups for {col}. Skipping...")
                continue
            
            # Skip if variance is zero
            if group1.var() == 0 or group2.var() == 0:
                print(f"Zero variance detected in one of the groups for {col}. Skipping...")
                continue

            # Perform t-test
            stat, p = ttest_ind(group1, group2, equal_var=False)
            print(f"T-test Results for {col} vs {target_col}: t-stat: {stat:.4f}, p-value: {p:.4f}")
            if p < 0.05:
                print("There is a statistically significant relationship.\n")
            else:
                print("No statistically significant relationship.\n")

    # Categorical columns
    for col in categorical_cols:
        plt.figure(figsize=figsize)
        sns.barplot(x=col, y=target_col, data=sample_data, ci=None, palette="muted")
        plt.title(f"Barplot of {target_col} by {col}")
        plt.xticks(rotation=45)
        plt.show()

        # Apply Chi-square test
        contingency = pd.crosstab(sample_data[col], sample_data[target_col])
        stat, p, _, _ = chi2_contingency(contingency)
        print(f"Chi-square Test Results for {col} vs {target_col}: Chi2: {stat:.4f}, p-value: {p:.4f}")
        if p < 0.05:
            print("There is a statistically significant relationship.\n")
        else:
            print("No statistically significant relationship.\n")