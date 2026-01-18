"""
Data loader module for real credit risk dataset.
Handles loading, preprocessing, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_credit_risk_data(
    data_path="data/credit_risk_dataset.csv",
    test_size=0.2,
    random_state=42,
    return_raw=False
):
    """
    Load and preprocess the real credit risk dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the credit risk CSV file
    test_size : float
        Fraction of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    return_raw : bool
        If True, also return the raw dataframe before preprocessing
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy arrays
        Preprocessed and scaled train/test splits
    df_raw : pandas DataFrame (optional)
        Raw dataframe if return_raw=True
    """
    
    # Load the dataset
    if not Path(data_path).exists():
        # Try relative to repo root
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / data_path
    
    df = pd.read_csv(data_path)
    
    print(f"Loaded credit risk dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Store raw dataframe
    df_raw = df.copy()
    
    # Identify target column (usually the last column or 'loan_status')
    target_col = None
    if 'loan_status' in df.columns:
        target_col = 'loan_status'
    else:
        # Assume last column is target
        target_col = df.columns[-1]
    
    print(f"Target column: {target_col}")
    print(f"Target distribution:\n{df[target_col].value_counts()}")
    
    # Separate features and target
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    
    # Handle missing values
    print(f"\nMissing values before imputation:")
    missing = X.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
        
        # Impute numerical columns with median
        if numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
        
        # Impute categorical columns with most frequent
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        
        print(f"Missing values after imputation: {X.isnull().sum().sum()}")
    else:
        print("No missing values found")
    
    # Encode categorical variables
    if categorical_cols:
        print(f"\nEncoding categorical variables...")
        for col in categorical_cols:
            # Use label encoding for binary categories, one-hot for multi-class
            n_unique = X[col].nunique()
            if n_unique == 2:
                # Binary encoding
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                print(f"  {col}: Label encoded (2 classes)")
            else:
                # One-hot encoding
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                print(f"  {col}: One-hot encoded ({n_unique} classes -> {len(dummies.columns)} features)")
    
    # Feature Engineering: Create domain-specific features
    print(f"\nCreating domain-specific features...")
    
    # Debt-to-income ratio (if loan_amnt and person_income exist)
    if 'loan_amnt' in X.columns and 'person_income' in X.columns:
        X['debt_to_income'] = X['loan_amnt'] / (X['person_income'] + 1e-8)
        print(f"  Created: debt_to_income")
    
    # Loan-to-income ratio (if loan_percent_income doesn't exist)
    if 'loan_percent_income' not in X.columns and 'loan_amnt' in X.columns and 'person_income' in X.columns:
        X['loan_to_income'] = X['loan_amnt'] / (X['person_income'] + 1e-8)
        print(f"  Created: loan_to_income")
    
    # Credit history per age (credit maturity)
    if 'cb_person_cred_hist_length' in X.columns and 'person_age' in X.columns:
        X['credit_maturity'] = X['cb_person_cred_hist_length'] / (X['person_age'] + 1e-8)
        print(f"  Created: credit_maturity")
    
    # Employment stability
    if 'person_emp_length' in X.columns and 'person_age' in X.columns:
        X['employment_stability'] = X['person_emp_length'] / (X['person_age'] + 1e-8)
        print(f"  Created: employment_stability")
    
    # Interest rate risk (if loan_int_rate exists)
    if 'loan_int_rate' in X.columns and 'loan_amnt' in X.columns:
        X['interest_burden'] = X['loan_int_rate'] * X['loan_amnt'] / 100.0
        print(f"  Created: interest_burden")
    
    # Income per age (earning power)
    if 'person_income' in X.columns and 'person_age' in X.columns:
        X['earning_power'] = X['person_income'] / (X['person_age'] + 1e-8)
        print(f"  Created: earning_power")
    
    # Convert to numpy array
    X = X.values
    
    print(f"\nFinal feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train target distribution: {np.bincount(y_train)}")
    print(f"Test target distribution: {np.bincount(y_test)}")
    
    # Standardization (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nData preprocessing complete!")
    
    if return_raw:
        return X_train_scaled, X_test_scaled, y_train, y_test, df_raw
    else:
        return X_train_scaled, X_test_scaled, y_train, y_test


def get_portfolio_metrics(data_path="data/credit_risk_dataset.csv", n_assets=6):
    """
    Extract portfolio metrics (mean returns and covariance) from real credit risk data.
    Uses loan intents as 'assets'.
    """
    if not Path(data_path).exists():
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / data_path
    
    df = pd.read_csv(data_path)
    
    # Fill missing interest rates with median
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    # Group by loan_intent
    intents = df['loan_intent'].unique()[:n_assets]
    
    # Calculate mean interest rate per intent as 'expected return'
    # Scale interest rates to a reasonable range for optimization (e.g., 0.05 to 0.15)
    mu_raw = df.groupby('loan_intent')['loan_int_rate'].mean()
    mu = (mu_raw[intents].values / 100.0) # convert to decimal
    
    # To get a covariance matrix, we need 'observations' for each intent.
    # We can use segments of the data (e.g., by age deciles) as observations.
    df['age_bin'] = pd.qcut(df['person_age'], q=10, duplicates='drop')
    
    # Create a pivot table: age_bin vs loan_intent, values = mean loan_int_rate
    pivot = df.pivot_table(
        values='loan_int_rate', 
        index='age_bin', 
        columns='loan_intent', 
        aggfunc='mean'
    ).fillna(method='ffill').fillna(method='bfill')
    
    # Use the selected intents
    pivot = pivot[intents]
    
    # Calculate covariance matrix
    # Scale covariance slightly to match the magnitude of returns
    cov = pivot.cov().values / (100.0 * 100.0)
    
    # Ensure cov is positive definite (add small diagonal)
    cov += np.eye(len(cov)) * 1e-6
    
    return mu, cov, list(intents)


def get_dataset_info(data_path="data/credit_risk_dataset.csv"):
    """
    Get basic information about the credit risk dataset without full preprocessing.
    Useful for dashboard display.
    """
    if not Path(data_path).exists():
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / data_path
    
    df = pd.read_csv(data_path)
    
    # Identify target column
    target_col = None
    if 'loan_status' in df.columns:
        target_col = 'loan_status'
    else:
        target_col = df.columns[-1]
    
    info = {
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,  # Exclude target
        'target_col': target_col,
        'target_distribution': df[target_col].value_counts().to_dict(),
        'columns': df.columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'numerical_cols': df.select_dtypes(include=['int64', 'float64']).columns.drop(target_col).tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'sample_data': df.head(5)
    }
    
    return info


if __name__ == "__main__":
    # Test the data loader
    print("Testing credit risk data loader...")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = load_credit_risk_data(
        test_size=0.2,
        random_state=42
    )
    
    print("\n" + "=" * 80)
    print("Data loader test successful!")
    print(f"Ready for model training with {X_train.shape[0]} training samples")
