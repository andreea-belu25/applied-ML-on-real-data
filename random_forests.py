import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from data_preprocessing import (
    fill_missing_values_data, replace_outliers_data, standardize_numerical_data
)
import warnings
warnings.filterwarnings('ignore')

def prepare_heart_data_for_rf(heart_set, all_heart_attributes_to_drop, heart_num_data, heart_cat_data):
    # Process training data
    heart_train_data = heart_set['heart_train_filled'].copy()
    heart_train_data = heart_train_data.drop(columns=list(all_heart_attributes_to_drop))
    
    # Filter numerical columns to only include those still present after dropping attributes
    remaining_num_cols = [col for col in heart_num_data.columns if col in heart_train_data.columns]
    remaining_num_data = pd.DataFrame(columns=remaining_num_cols)

    heart_train_data = replace_outliers_data(heart_train_data, remaining_num_data)
    heart_train_data = standardize_numerical_data(heart_train_data, remaining_num_data)
    
    # Process test data
    heart_test_data = heart_set['heart_test'].copy()
    heart_test_data = heart_test_data.drop(columns=list(all_heart_attributes_to_drop))

    # Filter columns for test data processing too
    remaining_num_cols_test = [col for col in heart_num_data.columns if col in heart_test_data.columns]
    remaining_cat_cols_test = [col for col in heart_cat_data.columns if col in heart_test_data.columns]
    
    # Fill missing values in test data
    heart_test_data_filled = fill_missing_values_data(heart_test_data, remaining_num_cols_test, remaining_cat_cols_test)
    heart_test_data_filled = replace_outliers_data(heart_test_data_filled, remaining_num_data)
    heart_test_data_filled = standardize_numerical_data(heart_test_data_filled, remaining_num_data)
    
    return heart_train_data, heart_test_data_filled

def prepare_news_data_for_rf(news_set, all_news_attributes_to_drop, news_num_data, news_cat_data):
    # Process training data
    news_train_data = news_set['news_train_filled'].copy()
    news_train_data = news_train_data.drop(columns=list(all_news_attributes_to_drop))
    
    # Filter numerical columns to only include those still present after dropping
    remaining_num_cols = [col for col in news_num_data.columns if col in news_train_data.columns]
    remaining_num_data = pd.DataFrame(columns=remaining_num_cols)  # Create dummy DataFrame with filtered columns
    
    news_train_data = replace_outliers_data(news_train_data, remaining_num_data)
    news_train_data = standardize_numerical_data(news_train_data, remaining_num_data)
    
    # Process test data
    news_test_data = news_set['news_test'].copy()
    news_test_data = news_test_data.drop(columns=list(all_news_attributes_to_drop))
    
    # Filter columns for test data processing too
    remaining_num_cols_test = [col for col in news_num_data.columns if col in news_test_data.columns]
    remaining_cat_cols_test = [col for col in news_cat_data.columns if col in news_test_data.columns]
    
    # Fill missing values in test data
    news_test_data_filled = fill_missing_values_data(news_test_data, remaining_num_cols_test, remaining_cat_cols_test)
    news_test_data_filled = replace_outliers_data(news_test_data_filled, remaining_num_data)
    news_test_data_filled = standardize_numerical_data(news_test_data_filled, remaining_num_data)

    return news_train_data, news_test_data_filled

def train_heart_random_forest(heart_train_data, heart_test_data, heart_target='chd_risk'):
    # Separate features and target
    X_heart_train = heart_train_data.drop(heart_target, axis=1)
    y_heart_train = heart_train_data[heart_target]
    
    X_heart_test = heart_test_data.drop(heart_target, axis=1)
    y_heart_test = heart_test_data[heart_target]
    
    # Train Random Forest with optimal parameters
    heart_forest_clf = RandomForestClassifier(
        n_estimators=1000,
        criterion='gini',
        min_samples_leaf=1,
        min_samples_split=5,
        random_state=0,
        max_depth=None,
        class_weight='balanced',
        bootstrap=True,
        max_samples=1500
    )
    
    heart_forest_clf.fit(X_heart_train, y_heart_train)
    y_heart_pred = heart_forest_clf.predict(X_heart_test)
    
    heart_accuracy = accuracy_score(y_heart_test, y_heart_pred)
    print(f"Random Forest Heart Disease accuracy: {heart_accuracy:.4f}")
    
    return heart_forest_clf, heart_accuracy, y_heart_pred, y_heart_test

def train_news_random_forest(news_train_data, news_test_data, news_target='popularity_category'):
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
    
    # Train Random Forest
    news_forest_clf = RandomForestClassifier(
        n_estimators=1000,
        criterion='entropy',
        min_samples_leaf=2,
        min_samples_split=2,
        random_state=0,
        max_depth=30,
        class_weight='balanced',
        max_samples=1500
    )
    
    news_forest_clf.fit(X_news_train_encoded, y_news_train_encoded)
    y_news_pred = news_forest_clf.predict(X_news_test_encoded)
    
    news_accuracy = accuracy_score(y_news_test_encoded, y_news_pred)
    
    # Convert predictions back to original labels for interpretation
    y_news_pred_original = le.inverse_transform(y_news_pred)
    y_news_test_original = le.inverse_transform(y_news_test_encoded)
    
    return news_forest_clf, news_accuracy, y_news_pred_original, y_news_test_original

def run_random_forest_experiments(heart_set, news_set, all_heart_attributes_to_drop, 
                                 all_news_attributes_to_drop, heart_num_data, heart_cat_data,
                                 news_num_data, news_cat_data):
    
    # Get data for analysis
    heart_train_data, heart_test_data = prepare_heart_data_for_rf(heart_set, all_heart_attributes_to_drop, heart_num_data, heart_cat_data)
    news_train_data, news_test_data = prepare_news_data_for_rf(news_set, all_news_attributes_to_drop, news_num_data, news_cat_data)
    
    # Train models
    heart_model, heart_acc, heart_pred, heart_true = train_heart_random_forest(heart_train_data, heart_test_data)
    news_model, news_acc, news_pred, news_true = train_news_random_forest(news_train_data, news_test_data)
    
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