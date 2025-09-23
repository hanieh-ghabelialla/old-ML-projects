# IBM Transaction Dataset Preprocessing Functions
# 
# This file contains preprocessing functions specifically designed for IBM transaction datasets
# These functions address the unique characteristics of transaction data including:
# - Mixed data types (numerical, categorical, temporal)
# - Imbalanced classes (fraud detection)
# - Missing values
# - High cardinality categorical features
# - Time-series dependencies

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings

def explore_transaction_dataset(df, target_column='is_fraud'):
    """
    Explore the structure and characteristics of an IBM transaction dataset
    
    Parameters:
    df (pandas.DataFrame): The transaction dataset
    target_column (str): Name of the target variable column
    
    Returns:
    dict: Summary statistics and insights about the dataset
    """
    print("=" * 60)
    print("IBM TRANSACTION DATASET EXPLORATION")
    print("=" * 60)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "="*40)
    print("COLUMN INFORMATION")
    print("="*40)
    print(f"Total columns: {len(df.columns)}")
    print(f"Numerical columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"DateTime columns: {len(df.select_dtypes(include=['datetime64']).columns)}")
    
    print("\nColumn types:")
    print(df.dtypes)
    
    print("\n" + "="*40)
    print("MISSING VALUES")
    print("="*40)
    missing_stats = df.isnull().sum()
    missing_percent = (missing_stats / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_stats,
        'Missing Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if target_column in df.columns:
        print("\n" + "="*40)
        print("TARGET VARIABLE ANALYSIS")
        print("="*40)
        print(f"Target variable: {target_column}")
        target_counts = df[target_column].value_counts()
        print(f"Class distribution:")
        for class_val, count in target_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {class_val}: {count} ({percentage:.2f}%)")
        
        # Calculate imbalance ratio
        if len(target_counts) == 2:
            minority_class = target_counts.min()
            majority_class = target_counts.max()
            imbalance_ratio = majority_class / minority_class
            print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 10:
                print("⚠️  WARNING: Highly imbalanced dataset detected!")
                print("   Consider using techniques like SMOTE, class weights, or stratified sampling")
    
    print("\n" + "="*40)
    print("RECOMMENDATIONS")
    print("="*40)
    
    recommendations = []
    
    # Check for high cardinality categorical features
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_column:
            unique_count = df[col].nunique()
            if unique_count > 100:
                recommendations.append(f"• {col} has {unique_count} unique values - consider using Label Encoding or dimensionality reduction")
            elif unique_count > 10:
                recommendations.append(f"• {col} has {unique_count} unique values - consider using Target Encoding")
    
    # Check for potential datetime columns
    for col in df.select_dtypes(include=['object']).columns:
        if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp']):
            recommendations.append(f"• {col} appears to be a datetime column - consider parsing and extracting time features")
    
    # Check for potential ID columns
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['id', 'user', 'transaction', 'account']):
            if df[col].nunique() / len(df) > 0.8:
                recommendations.append(f"• {col} appears to be an ID column with high cardinality - consider dropping or using for grouping")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("• Dataset structure looks good for preprocessing")
    
    return {
        'shape': df.shape,
        'missing_values': missing_stats.sum(),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'numerical_cols': len(df.select_dtypes(include=[np.number]).columns),
        'target_distribution': df[target_column].value_counts().to_dict() if target_column in df.columns else None
    }


def preprocess_transaction_features(df, timestamp_column=None, exclude_columns=None):
    """
    Extract features from transaction data including temporal features
    
    Parameters:
    df (pandas.DataFrame): The transaction dataset
    timestamp_column (str): Name of the timestamp column
    exclude_columns (list): Columns to exclude from feature engineering
    
    Returns:
    pandas.DataFrame: DataFrame with engineered features
    """
    df_processed = df.copy()
    
    if exclude_columns is None:
        exclude_columns = []
    
    print("Extracting transaction features...")
    
    # Handle timestamp features if provided
    if timestamp_column and timestamp_column in df.columns:
        print(f"Extracting time features from {timestamp_column}")
        
        # Convert to datetime if not already
        if df_processed[timestamp_column].dtype == 'object':
            df_processed[timestamp_column] = pd.to_datetime(df_processed[timestamp_column])
        
        # Extract time-based features
        df_processed['hour'] = df_processed[timestamp_column].dt.hour
        df_processed['day_of_week'] = df_processed[timestamp_column].dt.dayofweek
        df_processed['month'] = df_processed[timestamp_column].dt.month
        df_processed['quarter'] = df_processed[timestamp_column].dt.quarter
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        
        # Business hours feature
        df_processed['is_business_hours'] = ((df_processed['hour'] >= 9) & 
                                           (df_processed['hour'] <= 17) & 
                                           (df_processed['day_of_week'] < 5)).astype(int)
        
        # Time since epoch (for trend analysis)
        df_processed['timestamp_epoch'] = df_processed[timestamp_column].astype(np.int64) // 10**9
        
        # Optional: You might want to keep the original timestamp for sorting
        # exclude_columns.append(timestamp_column)
    
    # Handle amount features (common in transaction data)
    amount_columns = [col for col in df_processed.columns 
                     if any(keyword in col.lower() for keyword in ['amount', 'value', 'price', 'cost'])
                     and col not in exclude_columns]
    
    for col in amount_columns:
        if df_processed[col].dtype in ['int64', 'float64']:
            print(f"Engineering amount features for {col}")
            
            # Log transformation for skewed amount data
            df_processed[f'{col}_log'] = np.log1p(df_processed[col])
            
            # Amount categories
            df_processed[f'{col}_category'] = pd.cut(df_processed[col], 
                                                   bins=[0, 10, 50, 100, 500, float('inf')],
                                                   labels=['micro', 'small', 'medium', 'large', 'very_large'])
            
            # Is round number (psychological pricing effects)
            df_processed[f'{col}_is_round'] = (df_processed[col] % 1 == 0).astype(int)
    
    return df_processed


def encode_categorical_features(df, target_column=None, high_cardinality_threshold=50):
    """
    Encode categorical features in transaction dataset
    
    Parameters:
    df (pandas.DataFrame): Dataset with categorical features
    target_column (str): Name of target variable
    high_cardinality_threshold (int): Threshold for determining high cardinality features
    
    Returns:
    pandas.DataFrame: Dataset with encoded categorical features
    """
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    print(f"Encoding {len(categorical_columns)} categorical features...")
    
    label_encoders = {}
    
    for col in categorical_columns:
        unique_count = df_encoded[col].nunique()
        print(f"Processing {col}: {unique_count} unique values")
        
        # Handle missing values
        if df_encoded[col].isnull().any():
            df_encoded[col] = df_encoded[col].fillna('MISSING')
        
        if unique_count > high_cardinality_threshold:
            # High cardinality: use label encoding
            print(f"  Using Label Encoding (high cardinality)")
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            # Drop original column
            df_encoded = df_encoded.drop(columns=[col])
            
        elif unique_count > 10:
            # Medium cardinality: use label encoding (you might prefer target encoding here)
            print(f"  Using Label Encoding (medium cardinality)")
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            df_encoded = df_encoded.drop(columns=[col])
            
        else:
            # Low cardinality: use one-hot encoding
            print(f"  Using One-Hot Encoding (low cardinality)")
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=False)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
    
    return df_encoded, label_encoders


def handle_missing_values(df, strategy_numerical='median', strategy_categorical='most_frequent'):
    """
    Handle missing values in transaction dataset
    
    Parameters:
    df (pandas.DataFrame): Dataset with missing values
    strategy_numerical (str): Strategy for numerical features ('mean', 'median', 'constant')
    strategy_categorical (str): Strategy for categorical features ('most_frequent', 'constant')
    
    Returns:
    pandas.DataFrame: Dataset with imputed missing values
    """
    df_imputed = df.copy()
    
    print("Handling missing values...")
    
    # Separate numerical and categorical columns
    numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()
    
    # Impute numerical features
    if numerical_cols:
        print(f"Imputing {len(numerical_cols)} numerical features using {strategy_numerical}")
        num_imputer = SimpleImputer(strategy=strategy_numerical)
        df_imputed[numerical_cols] = num_imputer.fit_transform(df_imputed[numerical_cols])
    
    # Impute categorical features
    if categorical_cols:
        print(f"Imputing {len(categorical_cols)} categorical features using {strategy_categorical}")
        cat_imputer = SimpleImputer(strategy=strategy_categorical)
        df_imputed[categorical_cols] = cat_imputer.fit_transform(df_imputed[categorical_cols])
    
    return df_imputed


def scale_features(X_train, X_test, scaling_method='standard'):
    """
    Scale numerical features for transaction data
    
    Parameters:
    X_train (pandas.DataFrame): Training features
    X_test (pandas.DataFrame): Test features  
    scaling_method (str): 'standard', 'minmax', or 'robust'
    
    Returns:
    tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print(f"Scaling features using {scaling_method} scaling...")
    
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Scaling method must be 'standard' or 'minmax'")
    
    # Identify numerical columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fit scaler on training data only
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if numerical_cols:
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train_scaled, X_test_scaled, scaler


def handle_imbalanced_data(X, y, method='smote', random_state=42):
    """
    Handle imbalanced classes in transaction data (e.g., fraud detection)
    
    Parameters:
    X (pandas.DataFrame): Features
    y (pandas.Series): Target variable
    method (str): Method to handle imbalance ('smote', 'undersample', 'oversample')
    random_state (int): Random state for reproducibility
    
    Returns:
    tuple: (X_resampled, y_resampled)
    """
    print(f"Handling imbalanced data using {method}...")
    
    # Check class distribution
    class_counts = pd.Series(y).value_counts()
    print(f"Original class distribution: {dict(class_counts)}")
    
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    if imbalance_ratio < 3:
        print("Dataset is relatively balanced. No resampling needed.")
        return X, y
    
    if method == 'smote':
        # Use SMOTE for oversampling minority class
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
    else:
        raise ValueError("Only 'smote' method is implemented")
    
    # Check new class distribution
    new_class_counts = pd.Series(y_resampled).value_counts()
    print(f"Resampled class distribution: {dict(new_class_counts)}")
    
    return X_resampled, y_resampled


def preprocess_ibm_transaction_dataset(df, 
                                     target_column='is_fraud',
                                     timestamp_column=None,
                                     test_size=0.2,
                                     handle_imbalance=True,
                                     scaling_method='standard',
                                     random_state=42):
    """
    Complete preprocessing pipeline for IBM transaction dataset
    
    Parameters:
    df (pandas.DataFrame): Raw transaction dataset
    target_column (str): Name of target variable column
    timestamp_column (str): Name of timestamp column (if any)
    test_size (float): Proportion of data for testing
    handle_imbalance (bool): Whether to handle class imbalance
    scaling_method (str): Method for feature scaling
    random_state (int): Random state for reproducibility
    
    Returns:
    dict: Dictionary containing processed data and metadata
    """
    print("=" * 60)
    print("IBM TRANSACTION DATASET PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Initial exploration
    print("\n1. DATASET EXPLORATION")
    exploration_results = explore_transaction_dataset(df, target_column)
    
    # Step 2: Feature engineering
    print("\n2. FEATURE ENGINEERING")
    df_processed = preprocess_transaction_features(df, timestamp_column)
    
    # Step 3: Handle missing values
    print("\n3. HANDLING MISSING VALUES")
    df_processed = handle_missing_values(df_processed)
    
    # Step 4: Separate features and target
    print("\n4. SEPARATING FEATURES AND TARGET")
    if target_column not in df_processed.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Step 5: Encode categorical features
    print("\n5. ENCODING CATEGORICAL FEATURES")
    X_encoded, label_encoders = encode_categorical_features(X, target_column)
    
    # Step 6: Train-test split
    print("\n6. TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 7: Handle imbalanced data (on training set only)
    if handle_imbalance:
        print("\n7. HANDLING IMBALANCED DATA")
        X_train_resampled, y_train_resampled = handle_imbalanced_data(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Step 8: Feature scaling
    print("\n8. FEATURE SCALING")
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_resampled, X_test, scaling_method
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Final training set shape: {X_train_scaled.shape}")
    print(f"Final test set shape: {X_test_scaled.shape}")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_resampled,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': X_train_scaled.columns.tolist(),
        'exploration_results': exploration_results
    }


# Example usage:
if __name__ == "__main__":
    # Example of how to use these functions with an IBM transaction dataset
    print("IBM Transaction Dataset Preprocessing Functions")
    print("=" * 60)
    print("This module provides comprehensive preprocessing for IBM transaction datasets.")
    print("\nKey features:")
    print("• Handles mixed data types (numerical, categorical, temporal)")
    print("• Addresses class imbalance (fraud detection)")
    print("• Proper feature engineering for transaction data")
    print("• Missing value imputation")
    print("• Categorical encoding strategies")
    print("• Feature scaling options")
    print("\nExample usage:")
    print("""
# Load your IBM transaction dataset
df = pd.read_csv('ibm_transaction_dataset.csv')

# Run complete preprocessing pipeline
results = preprocess_ibm_transaction_dataset(
    df=df,
    target_column='is_fraud',  # or your target column name
    timestamp_column='timestamp',  # if you have timestamp data
    test_size=0.2,
    handle_imbalance=True,
    scaling_method='standard'
)

# Access processed data
X_train = results['X_train']
X_test = results['X_test'] 
y_train = results['y_train']
y_test = results['y_test']
    """)