# src/train_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')
RANDOM_STATE = 42

def train_models(X_train, X_test, y_train, y_test, save_model_path='../model/churn_model.pkl', save_features_path='../model/feature_names.pkl'):
    """Train models, evaluate, and save best model"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'model': model,
            'Accuracy': acc,
            'ROC_AUC': roc_auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    # Select best model by ROC_AUC
    best_name = max(results.keys(), key=lambda k: results[k]['ROC_AUC'])
    best_model = results[best_name]['model']
    
    # Save best model
    with open(save_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save feature names
    with open(save_features_path, 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)
    
    return results, best_name, best_model