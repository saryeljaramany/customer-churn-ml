# src/preprocessing.py
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
CARDINALITY_THRESHOLD = 30

def clean_data(df, save_path='../data/processed/cleaned_telco_churn.csv'):
    """Clean and encode the Telco dataset"""
    
    # Drop customerID
    df = df.drop(columns=['customerID'], errors='ignore')
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Encode target
    df['Churn_Yes'] = (df['Churn'] == 'Yes').astype(int)
    df = df.drop(columns=['Churn'], errors='ignore')
    
    # Detect categorical columns
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # One-hot encode low cardinality
    cols_onehot = [c for c in cat_cols if df[c].nunique() <= CARDINALITY_THRESHOLD]
    cols_freq = [c for c in cat_cols if df[c].nunique() > CARDINALITY_THRESHOLD]
    
    if cols_onehot:
        df = pd.get_dummies(df, columns=cols_onehot, drop_first=True)
    
    # Frequency encode high cardinality
    for col in cols_freq:
        freqs = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freqs)
        df = df.rename(columns={col: f"{col}_freq"})
    
    # Save cleaned data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def split_and_scale(df, test_size=0.2, save_scaler_path='../model/scaler.pkl'):
    """Split dataset and scale numerical features"""
    X = df.drop('Churn_Yes', axis=1)
    y = df['Churn_Yes']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Save scaler
    with open(save_scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train, X_test, y_train, y_test, scaler