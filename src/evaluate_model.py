# src/evaluate_model.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

sns.set(style='whitegrid')

def plot_roc(results, y_test):
    """Plot ROC curves for multiple models"""
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['ROC_AUC']:.3f})")
    plt.plot([0,1], [0,1], 'k--', label='Random baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Churn Prediction Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(best_model, X_train, best_name):
    """Plot top 20 features importance"""
    feature_names = X_train.columns
    if best_name == "Random Forest":
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(20))
        plt.title("Top 20 Feature Importances (Random Forest)")
        plt.show()
    elif best_name == "Logistic Regression":
        coefficients = best_model.coef_[0]
        feature_coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients
        }).sort_values("Coefficient", key=abs, ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Coefficient", y="Feature", data=feature_coef_df.head(20))
        plt.title("Top 20 Feature Effects (Logistic Regression)")
        plt.show()