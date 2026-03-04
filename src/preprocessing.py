import pandas as pd
from sklearn.preprocessing import StandardScaler


def convert_total_charges(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    return df


def encode_target(df):
    df['Churn_Yes'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df = df.drop('Churn', axis=1)
    return df


def encode_categoricals(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def scale_numeric(df):
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler