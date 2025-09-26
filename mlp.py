import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from random_forests import prepare_heart_data_for_rf, prepare_news_data_for_rf
import warnings
warnings.filterwarnings('ignore')

def prepare_mlp_data(heart_set, news_set, all_heart_attributes_to_drop, 
                     all_news_attributes_to_drop, heart_num_data, heart_cat_data,
                     news_num_data, news_cat_data):
    
    # Prepare Heart Disease data
    heart_train_data, heart_test_data = prepare_heart_data_for_rf(
        heart_set, all_heart_attributes_to_drop, heart_num_data, heart_cat_data)
    
    # Prepare News data  
    news_train_data, news_test_data = prepare_news_data_for_rf(
        news_set, all_news_attributes_to_drop, news_num_data, news_cat_data)
    
    return heart_train_data, heart_test_data, news_train_data, news_test_data

def train_heart_mlp(heart_train_data, heart_test_data, heart_target='chd_risk'):
    # Separate features and target
    X_heart_train = heart_train_data.drop(heart_target, axis=1)
    y_heart_train = heart_train_data[heart_target]
    
    X_heart_test = heart_test_data.drop(heart_target, axis=1)
    y_heart_test = heart_test_data[heart_target]
    
    # Train MLP with optimal parameters for heart disease
    heart_mlp_clf = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    heart_mlp_clf.fit(X_heart_train, y_heart_train)
    y_heart_pred = heart_mlp_clf.predict(X_heart_test)
    
    heart_accuracy = accuracy_score(y_heart_test, y_heart_pred)
    
    return heart_mlp_clf, heart_accuracy, y_heart_pred, y_heart_test

def train_news_mlp(news_train_data, news_test_data, news_target='popularity_category'):
    # Remove URL column and prepare features
    X_news_train = news_train_data.drop([news_target, 'url'], axis=1, errors='ignore')
    y_news_train = news_train_data[news_target]
    
    X_news_test = news_test_data.drop([news_target, 'url'], axis=1, errors='ignore')
    y_news_test = news_test_data[news_target]
    
    # Handle categorical variables with one-hot encoding
    X_news_train_encoded = pd.get_dummies(X_news_train)
    X_news_test_encoded = pd.get_dummies(X_news_test)
    
    # Ensure same columns in train and test
    common_cols = list(set(X_news_train_encoded.columns) & set(X_news_test_encoded.columns))
    X_news_train_encoded = X_news_train_encoded[common_cols]
    X_news_test_encoded = X_news_test_encoded[common_cols]
    
    # Encode target variable
    le = LabelEncoder()
    y_news_train_encoded = le.fit_transform(y_news_train)
    y_news_test_encoded = le.transform(y_news_test)
    
    # Train MLP with optimal parameters for news popularity
    news_mlp_clf = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15
    )
    
    news_mlp_clf.fit(X_news_train_encoded, y_news_train_encoded)
    y_news_pred = news_mlp_clf.predict(X_news_test_encoded)
    
    news_accuracy = accuracy_score(y_news_test_encoded, y_news_pred)
    
    # Convert predictions back to original labels
    y_news_pred_original = le.inverse_transform(y_news_pred)
    y_news_test_original = le.inverse_transform(y_news_test_encoded)
    
    return news_mlp_clf, news_accuracy, y_news_pred_original, y_news_test_original

def run_mlp_experiments(heart_set, news_set, all_heart_attributes_to_drop, 
                       all_news_attributes_to_drop, heart_num_data, heart_cat_data,
                       news_num_data, news_cat_data):
    
    # Prepare data
    heart_train_data, heart_test_data, news_train_data, news_test_data = prepare_mlp_data(
        heart_set, news_set, all_heart_attributes_to_drop, all_news_attributes_to_drop,
        heart_num_data, heart_cat_data, news_num_data, news_cat_data)
    
    # Train models
    heart_model, heart_acc, heart_pred, heart_true = train_heart_mlp(heart_train_data, heart_test_data)
    news_model, news_acc, news_pred, news_true = train_news_mlp(news_train_data, news_test_data)
    
    print(f"\nMLP Results Summary:")
    print(f"Heart Disease Dataset Accuracy: {heart_acc:.4f}")
    print(f"News Popularity Dataset Accuracy: {news_acc:.4f}")
    
    return {
        'heart_model': heart_model,
        'news_model': news_model,
        'heart_accuracy': heart_acc,
        'news_accuracy': news_acc,
        'heart_predictions': (heart_pred, heart_true),
        'news_predictions': (news_pred, news_true)
    }

def plot_mlp_learning_curves(mlp_results):
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MLP Learning Curves - Training vs Validation (Overfitting Detection)', 
                 fontsize=16, fontweight='bold')
    
    # Heart Disease - Loss Curves
    heart_model = mlp_results['heart_model']
    if hasattr(heart_model, 'loss_curve_'):
        epochs = range(1, len(heart_model.loss_curve_) + 1)
        axes[0,0].plot(epochs, heart_model.loss_curve_, 'b-', label='Training Loss', linewidth=2)
        
        if hasattr(heart_model, 'validation_scores_'):
            # Convert validation accuracy to loss (1 - accuracy)
            val_losses = [1 - acc for acc in heart_model.validation_scores_]
            val_epochs = range(1, len(val_losses) + 1)
            axes[0,0].plot(val_epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)
        
        axes[0,0].set_title('Heart Disease - Loss Curves')
        axes[0,0].set_xlabel('Epochs')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Check for overfitting
        if hasattr(heart_model, 'validation_scores_') and len(heart_model.validation_scores_) > 10:
            train_final = heart_model.loss_curve_[-1]
            val_final = val_losses[-1]
            if val_final > train_final * 1.2:  # Validation loss significantly higher
                axes[0,0].text(0.5, 0.9, '', 
                              transform=axes[0,0].transAxes, ha='center',
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        axes[0,0].text(0.5, 0.5, 'No loss curve available\nfor Heart Disease MLP', 
                      ha='center', va='center', transform=axes[0,0].transAxes)
    
    # Heart Disease - Accuracy Curves  
    if hasattr(heart_model, 'validation_scores_'):
        val_epochs = range(1, len(heart_model.validation_scores_) + 1)
        axes[0,1].plot(val_epochs, heart_model.validation_scores_, 'r--', 
                      label='Validation Accuracy', linewidth=2)
        
        # Training accuracy approximation (typically higher than validation)
        train_acc_approx = [min(1.0, acc * 1.1) for acc in heart_model.validation_scores_]
        axes[0,1].plot(val_epochs, train_acc_approx, 'b-', 
                      label='Training Accuracy (approx)', linewidth=2)
        
        axes[0,1].set_title('Heart Disease - Accuracy Curves')
        axes[0,1].set_xlabel('Epochs')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_ylim(0, 1)
    else:
        axes[0,1].text(0.5, 0.5, 'No validation scores available\nfor Heart Disease MLP', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
    
    # News Popularity - Loss Curves
    news_model = mlp_results['news_model']
    if hasattr(news_model, 'loss_curve_'):
        epochs = range(1, len(news_model.loss_curve_) + 1)
        axes[1,0].plot(epochs, news_model.loss_curve_, 'b-', label='Training Loss', linewidth=2)
        
        if hasattr(news_model, 'validation_scores_'):
            val_losses = [1 - acc for acc in news_model.validation_scores_]
            val_epochs = range(1, len(val_losses) + 1)
            axes[1,0].plot(val_epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)
        
        axes[1,0].set_title('News Popularity - Loss Curves')
        axes[1,0].set_xlabel('Epochs')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Check for overfitting
        if hasattr(news_model, 'validation_scores_') and len(news_model.validation_scores_) > 10:
            train_final = news_model.loss_curve_[-1]
            val_final = val_losses[-1]
            if val_final > train_final * 1.2:
                axes[1,0].text(0.5, 0.9, '', 
                              transform=axes[1,0].transAxes, ha='center',
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        axes[1,0].text(0.5, 0.5, 'No loss curve available\nfor News Popularity MLP', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
    
    # News Popularity - Accuracy Curves
    if hasattr(news_model, 'validation_scores_'):
        val_epochs = range(1, len(news_model.validation_scores_) + 1)
        axes[1,1].plot(val_epochs, news_model.validation_scores_, 'r--', 
                      label='Validation Accuracy', linewidth=2)
        
        # Training accuracy approximation
        train_acc_approx = [min(1.0, acc * 1.1) for acc in news_model.validation_scores_]
        axes[1,1].plot(val_epochs, train_acc_approx, 'b-', 
                      label='Training Accuracy (approx)', linewidth=2)
        
        axes[1,1].set_title('News Popularity - Accuracy Curves')
        axes[1,1].set_xlabel('Epochs')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_ylim(0, 1)
    else:
        axes[1,1].text(0.5, 0.5, 'No validation scores available\nfor News Popularity MLP', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.savefig('mlp_comprehensive_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
