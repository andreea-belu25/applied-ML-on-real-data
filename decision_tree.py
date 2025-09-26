import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from random_forests import prepare_heart_data_for_rf, prepare_news_data_for_rf
import warnings
warnings.filterwarnings('ignore')

def train_heart_decision_tree(heart_train_data, heart_test_data, heart_target='chd_risk'):
    # Separate features and target
    X_heart_train = heart_train_data.drop(heart_target, axis=1)
    y_heart_train = heart_train_data[heart_target]
    
    X_heart_test = heart_test_data.drop(heart_target, axis=1)
    y_heart_test = heart_test_data[heart_target]
    
    # Train Decision Tree with optimal parameters for heart disease
    heart_dt_clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )
    
    heart_dt_clf.fit(X_heart_train, y_heart_train)
    y_heart_pred = heart_dt_clf.predict(X_heart_test)
    
    heart_accuracy = accuracy_score(y_heart_test, y_heart_pred)
    
    return heart_dt_clf, heart_accuracy, y_heart_pred, y_heart_test

def train_news_decision_tree(news_train_data, news_test_data, news_target='popularity_category'):
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
    
    # Train Decision Tree with optimal parameters for news popularity
    news_dt_clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        class_weight='balanced'
    )
    
    news_dt_clf.fit(X_news_train_encoded, y_news_train_encoded)
    y_news_pred = news_dt_clf.predict(X_news_test_encoded)
    
    news_accuracy = accuracy_score(y_news_test_encoded, y_news_pred)
    
    # Convert predictions back to original labels
    y_news_pred_original = le.inverse_transform(y_news_pred)
    y_news_test_original = le.inverse_transform(y_news_test_encoded)
    
    return news_dt_clf, news_accuracy, y_news_pred_original, y_news_test_original

def run_decision_tree_experiments(heart_set, news_set, all_heart_attributes_to_drop, 
                                 all_news_attributes_to_drop, heart_num_data, heart_cat_data,
                                 news_num_data, news_cat_data):
    
    # Prepare data using the same functions as Random Forest
    heart_train_data, heart_test_data = prepare_heart_data_for_rf(
        heart_set, all_heart_attributes_to_drop, heart_num_data, heart_cat_data)
    news_train_data, news_test_data = prepare_news_data_for_rf(
        news_set, all_news_attributes_to_drop, news_num_data, news_cat_data)
    
    # Train models
    heart_model, heart_acc, heart_pred, heart_true = train_heart_decision_tree(
        heart_train_data, heart_test_data)
    news_model, news_acc, news_pred, news_true = train_news_decision_tree(
        news_train_data, news_test_data)
    
    print(f"\nDecision Tree Results Summary:")
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