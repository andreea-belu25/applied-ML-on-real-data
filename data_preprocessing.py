import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

def combine_imputed_data(original_data, imputed_numerical_data, imputed_categorical_data, num_data, cat_data):
    # Combine the imputed numerical and categorical data with the original data
    if len(num_data) == 0:
        return imputed_categorical_data
    
    if len(cat_data) == 0:
        return imputed_numerical_data
    
    combined_df = pd.concat([imputed_numerical_data, imputed_categorical_data], axis=1)
    
    return combined_df[original_data.columns.tolist()]
    
def impute_categorical_data(original_data, cat_data):
    # Impute missing values in categorical data
    if len(cat_data) == 0:
        return pd.DataFrame()
    
    imputer = SimpleImputer(strategy='most_frequent')
    transformed_data = imputer.fit_transform(original_data[cat_data])
    result_df = pd.DataFrame(transformed_data, columns=cat_data)
    
    return result_df

def impute_numerical_data(original_data, num_data):
    # Impute missing values in numerical data
    if len (num_data) == 0:
        return pd.DataFrame()
    
    imputer = IterativeImputer(max_iter=300, random_state=0, initial_strategy='mean', n_nearest_features=4, tol=1e-1)
    transformed_data = imputer.fit_transform(original_data[num_data])
    result_df = pd.DataFrame(transformed_data, columns=num_data)
    
    return result_df

def fill_missing_values_data(original_data, num_data, cat_data):
    numerical_imputed = impute_numerical_data(original_data, num_data)
    categorical_imputed = impute_categorical_data(original_data, cat_data)
    result_data = combine_imputed_data(original_data, numerical_imputed, categorical_imputed, num_data, cat_data)

    return result_data

def process_heart_disease_missing_data(heart_set, heart_num_data, heart_cat_data):
    original_heart_data = heart_set['heart_train'].copy()
    total_missing_values = original_heart_data.isnull().sum().sum()
    if total_missing_values == 0:
        return original_heart_data
    else:
        heart_num_attributs = heart_num_data.columns.tolist()
        heart_cat_attributs = heart_cat_data.columns.tolist()
        filled_heart_data = fill_missing_values_data(original_heart_data, heart_num_attributs, heart_cat_attributs)

    return filled_heart_data

def process_news_popularity_missing_data(news_set, news_num_data, news_cat_data):
    original_news_data = news_set['news_train'].copy()
    total_missing_values = original_news_data.isnull().sum().sum()
    if total_missing_values == 0:
        return original_news_data
    else:
        news_num_attributs = news_num_data.columns.tolist()
        news_cat_attributs = news_cat_data.columns.tolist()
        filled_news_data = fill_missing_values_data(original_news_data, news_num_attributs, news_cat_attributs)

    return filled_news_data

def print_filled_data_info(filled_data):
    print(f"Shape: {filled_data.shape}")
    print(f"Missing values: {filled_data.isnull().sum().sum()}")
    print("First 5 rows:")
    print(filled_data.head())
    print("\n" + "="*50 + "\n")

def replace_outliers_data(data, num_data):
    if len(num_data) == 0:
        return data
    
    data_copy = data.copy() # to preserve the original data
    cols = num_data.columns.tolist()
    bounds = {}
    total_outliers = 0
    
    for column in cols:
        col_q1 = data[column].quantile(0.25)
        col_q3 = data[column].quantile(0.75)
        col_iqr = col_q3 - col_q1
        threshold = 1.5

        bounds[column] = {
            'lower': col_q1 - threshold * col_iqr,
            'upper': col_q3 + threshold * col_iqr
        }

    # create temporary data with outliers replaced with NaN
    temp_data = data_copy.copy()
    for column in cols:
        if column not in bounds:
            continue

        lower_bound = bounds[column]['lower']
        upper_bound = bounds[column]['upper']
        outlier_mask = (temp_data[column] < lower_bound) | (temp_data[column] > upper_bound)
        column_outliers = outlier_mask.sum()

        if column_outliers > 0:
            total_outliers += column_outliers
            temp_data.loc[outlier_mask, column] = np.nan

    if total_outliers == 0:
        print("No outliers found in the numerical data.")
        return data_copy

    imputed_numerical_data = impute_numerical_data(temp_data, cols)
    for column in cols:
        if column not in bounds:
            continue

        lower_bound = bounds[column]['lower']
        upper_bound = bounds[column]['upper']
        outlier_mask = (data_copy[column] < lower_bound) | (data_copy[column] > upper_bound)
        
        data_copy.loc[outlier_mask, column] = imputed_numerical_data.loc[outlier_mask, column]

    return data_copy

def remove_redundant_news_attributes(attributes_to_drop, replaced_extreme_news_num_data):
    if not attributes_to_drop:
        cleaned_data = replaced_extreme_news_num_data.copy()
        print("No redundant attributes found in News Popularity dataset.")
    else:
        print(f"Removing {len(attributes_to_drop)} redundant attributes from News Popularity:")
        for attr in attributes_to_drop:
            print(f"  - {attr}")
        cleaned_data = replaced_extreme_news_num_data.drop(columns=list(attributes_to_drop))
        print(f"News Popularity: {replaced_extreme_news_num_data.shape} → {cleaned_data.shape}")

    return cleaned_data

def remove_redundant_heart_attributes(attributes_to_drop, replaced_extreme_heart_num_data):
    if not attributes_to_drop:
        cleaned_data = replaced_extreme_heart_num_data.copy()
        print("No redundant attributes found in Heart Disease dataset.")
    else:
        print(f"Removing {len(attributes_to_drop)} redundant attributes from Heart Disease:")
        for attr in attributes_to_drop:
            print(f"  - {attr}")
        cleaned_data = replaced_extreme_heart_num_data.drop(columns=list(attributes_to_drop))
        print(f"Heart Disease: {replaced_extreme_heart_num_data.shape} → {cleaned_data.shape}")

    return cleaned_data

def standardize_numerical_data(data, num_data):
    standardized_data = data.copy()

    existing_num_cols = [col for col in num_data if col in data.columns]
    
    if existing_num_cols:
        scaler = preprocessing.StandardScaler()
        standardized_data[existing_num_cols] = scaler.fit_transform(data[existing_num_cols])
    
    return standardized_data