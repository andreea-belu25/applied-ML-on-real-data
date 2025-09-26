import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score,
)
import warnings
warnings.filterwarnings('ignore')

def plot_confusion_matrices(rf_results, mlp_results, dt_results):
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
    
    # Decision Tree
    heart_pred_dt, heart_true_dt = dt_results['heart_predictions']
    news_pred_dt, news_true_dt = dt_results['news_predictions']
    
    ConfusionMatrixDisplay.from_predictions(
        heart_true_dt, heart_pred_dt,
        ax=axes[0, 0],
        cmap='Reds',
        values_format='d',
        display_labels=['No Risk', 'Risk']
    )
    axes[0, 0].set_title('Decision Tree - Heart Disease')
    
    ConfusionMatrixDisplay.from_predictions(
        news_true_dt, news_pred_dt,
        ax=axes[0, 1],
        cmap='Reds',
        values_format='d'
    )
    axes[0, 1].set_title('Decision Tree - News Popularity')
    
    # Random Forest
    heart_pred_rf, heart_true_rf = rf_results['heart_predictions']
    news_pred_rf, news_true_rf = rf_results['news_predictions']
    
    ConfusionMatrixDisplay.from_predictions(
        heart_true_rf, heart_pred_rf,
        ax=axes[1, 0],
        cmap='Blues',
        values_format='d',
        display_labels=['No Risk', 'Risk']
    )
    axes[1, 0].set_title('Random Forest - Heart Disease')
    
    ConfusionMatrixDisplay.from_predictions(
        news_true_rf, news_pred_rf,
        ax=axes[1, 1],
        cmap='Greens',
        values_format='d'
    )
    axes[1, 1].set_title('Random Forest - News Popularity')
    
    # MLP
    heart_pred_mlp, heart_true_mlp = mlp_results['heart_predictions']
    news_pred_mlp, news_true_mlp = mlp_results['news_predictions']
    
    ConfusionMatrixDisplay.from_predictions(
        heart_true_mlp, heart_pred_mlp,
        ax=axes[2, 0],
        cmap='Oranges',
        values_format='d',
        display_labels=['No Risk', 'Risk']
    )
    axes[2, 0].set_title('MLP - Heart Disease')
    
    ConfusionMatrixDisplay.from_predictions(
        news_true_mlp, news_pred_mlp,
        ax=axes[2, 1],
        cmap='Purples',
        values_format='d'
    )
    axes[2, 1].set_title('MLP - News Popularity')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comparative_table(rf_results, mlp_results, dt_results):
    results_data = []
    
    algorithms = [
        ('Decision Tree', dt_results),
        ('Random Forest', rf_results), 
        ('MLP', mlp_results)
    ]
    
    for dataset_name, target_key in [('Heart Disease', 'heart_predictions'), ('News Popularity', 'news_predictions')]:
        for algorithm_name, results in algorithms:
            pred, true = results[target_key]
            
            # Calculate metrics
            accuracy = accuracy_score(true, pred)
            precision = precision_score(true, pred, average='weighted', zero_division=0)
            recall = recall_score(true, pred, average='weighted', zero_division=0)
            f1 = f1_score(true, pred, average='weighted', zero_division=0)
            
            # One row per algorithm per dataset
            results_data.append({
                'Dataset': dataset_name,
                'Algorithm': algorithm_name,
                'Accuracy': f"{accuracy:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'F1-Score': f"{f1:.4f}"
            })
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('algorithm_comparison.csv', index=False)
    
    return results_df

def run_complete_comparison(rf_results, mlp_results, dt_results):
    # Generate confusion matrices
    plot_confusion_matrices(rf_results, mlp_results, dt_results)
    
    # Create comparative table
    comparison_df = create_comparative_table(rf_results, mlp_results, dt_results)
    
    return comparison_df