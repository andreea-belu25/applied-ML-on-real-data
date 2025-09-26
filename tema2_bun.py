import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_heart_set():
    heart_set = {}

    heart_set['heart_train'] = pd.read_csv('heart_1_train.csv');
    heart_set['heart_test'] = pd.read_csv('heart_1_test.csv');

    heart_set['heart_train'].columns = heart_set['heart_train'].columns.str.strip();
    heart_set['heart_test'].columns = heart_set['heart_test'].columns.str.strip();

    return heart_set;

def load_news_set():
    news_set = {}

    news_set['news_train'] = pd.read_csv('news_popularity_train.csv');
    news_set['news_test'] = pd.read_csv('news_popularity_test.csv');

    news_set['news_train'].columns = news_set['news_train'].columns.str.strip();
    news_set['news_test'].columns = news_set['news_test'].columns.str.strip();

    return news_set;

def separate_data_heart_set(heart_set):
    # Separate data between numerical and categorical   
    heart_categorical_cols = [
        'gender', 'education_level', 'stroke_history', 'hypertension_history', 
        'diabetes_history', 'high_blood_sugar', 'smoking_status', 
        'blood_pressure_medication', 'chd_risk'  # chd_risk is the target
    ]

    heart_numerical_cols = [
        'age', 'systolic_pressure', 'diastolic_pressure', 'daily_cigarettes',
        'heart_rate', 'mass_index', 'blood_sugar_level', 'glucose', 
        'cholesterol_level', 'total_cigarettes'
    ]

    heart_num_data = heart_set['heart_train'][heart_numerical_cols]
    heart_cat_data = heart_set['heart_train'][heart_categorical_cols]

    return heart_num_data, heart_cat_data

def separate_data_news_set(news_set):
    # Separate data between numerical and categorical   
    news_categorical_cols = [
        'url', 'channel_lifestyle', 'channel_entertainment', 'channel_business', 
        'channel_social_media', 'channel_tech', 'channel_world',
        'day_monday', 'day_tuesday', 'day_wednesday', 'day_thursday', 
        'day_friday', 'day_saturday', 'day_sunday',
        'is_weekend', 'publication_period', 'popularity_category'
    ]

    news_numerical_cols = [
        'days_since_published', 'title_word_count', 'content_word_count',
        'unique_word_ratio', 'non_stop_word_ratio', 'unique_non_stop_ratio',
        'external_links', 'internal_links', 'image_count', 'video_count',
        'avg_word_length', 'keyword_count',
        'keyword_worst_min_shares', 'keyword_worst_max_shares', 'keyword_worst_avg_shares',
        'keyword_best_min_shares', 'keyword_best_max_shares', 'keyword_best_avg_shares',
        'keyword_avg_min_shares', 'keyword_avg_max_shares', 'keyword_avg_avg_shares',
        'ref_min_shares', 'ref_max_shares', 'ref_avg_shares',
        'topic_0_relevance', 'topic_1_relevance', 'topic_2_relevance', 
        'topic_3_relevance', 'topic_4_relevance',
        'content_subjectivity', 'content_sentiment',
        'positive_word_rate', 'negative_word_rate',
        'non_neutral_positive_rate', 'non_neutral_negative_rate',
        'avg_positive_sentiment', 'min_positive_sentiment', 'max_positive_sentiment',
        'avg_negative_sentiment', 'min_negative_sentiment', 'max_negative_sentiment',
        'title_subjectivity', 'title_sentiment',
        'title_subjectivity_magnitude', 'title_sentiment_magnitude',
        'engagement_ratio', 'content_density'
    ]

    news_num_data = news_set['news_train'][news_numerical_cols]
    news_cat_data = news_set['news_train'][news_categorical_cols]
    
    return news_num_data, news_cat_data

def analyze_num_data(num_data, dataset_name):
    # Generate statistics table for numerical data
    stats_table = []

    for col in num_data.columns:
        stats_element = {
            'attribute': col,
            'non_null_count': num_data[col].count(),
            'mean': num_data[col].mean(),
            'std': num_data[col].std(),
            'min': num_data[col].min(),
            'max': num_data[col].max(),
            '25%': num_data[col].quantile(0.25),
            '50%': num_data[col].quantile(0.5),
            '75%': num_data[col].quantile(0.75)
        }
        stats_table.append(stats_element)
    
    stats_table_df = pd.DataFrame(stats_table)
    
    # For each numerical attribute, generate a bloxpot
    for _, row in stats_table_df.iterrows():
        attribute_name = row['attribute']
        stats = row[['non_null_count', 'mean', 'std', 'min', 'max', '25%', '50%', '75%']]
        stats_df = pd.DataFrame([stats])
        boxplot = stats_df.boxplot(column=['non_null_count', 'mean', 'std', 'min', 'max', '25%', '50%', '75%'], 
                                figsize=(12, 6), rot=45)
        plt.suptitle(f'Num_Statistics for {attribute_name} - {dataset_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_{attribute_name}_stats_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

    return stats_table_df

def analyze_cat_data(cat_data, dataset_name):
    # Generate statistics table for categorical data 
    stats_table = []

    for col in cat_data.columns:
        stats_element = {
            'attribute': col,
            'non_null_count': cat_data[col].count(),
            'unique_count': cat_data[col].nunique()
        }
        stats_table.append(stats_element)

    stats_table_df = pd.DataFrame(stats_table)

    # For each categorical attribute, generate a histogram
    for _, row in stats_table_df.iterrows():
        attribute_name = row['attribute']
        plt.figure(figsize=(10, 6))
        stats_values = [row['non_null_count'], row['unique_count']]
        labels = ['Non-null Count', 'Unique Values']
        bars = plt.bar(labels, stats_values, color=['skyblue', 'lightgreen'], width=0.6)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Cat_Statistics for {attribute_name} - {dataset_name}', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_{attribute_name}_cat_stats.png', dpi=200, bbox_inches='tight')
        plt.close()
        
    return stats_table_df  

def analyze_class_balance(data, dataset_name, columns=None):
    # Analyze only the specified columns because the dataset is too large => it got killed

    if isinstance(columns, str):
        # If a single column is provided    
        columns_to_analyze = [columns]
    else:
        # If a list of columns is provided
        columns_to_analyze = [col for col in columns if col in data.columns]
    
    # Analyze each specified column
    for col in columns_to_analyze:
        normalized_data = data[col].value_counts().sort_index()
        percentages = data[col].value_counts(normalize=True).sort_index() * 100
        
        # Calculate imbalance ratio
        max_val_cnt = normalized_data.max()
        min_val_cnt = normalized_data.min()
        imbalance = max_val_cnt / min_val_cnt
        
        # Create figure
        plt.figure(figsize=(max(10, len(normalized_data) * 0.8), 6))
        bars = sns.barplot(x=normalized_data.index.astype(str), y=normalized_data.values)

        # Add count and percentage labels to bars
        for i, (bar, pct) in enumerate(zip(bars.patches, percentages.values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + (max_val_cnt * 0.01),
                    f'{int(height)}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        # Add title and labels
        plt.title(f'Class Distribution for {col} - {dataset_name}', fontsize=14)
        plt.xlabel('Values')
        plt.ylabel('Count')
        
        # Add imbalance ratio
        plt.annotate(f'Imbalance ratio: {imbalance:.2f}', 
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', va='top', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        
        # Handle x-axis label rotation for better readability
        if len(normalized_data) > 5 or max(len(str(x)) for x in normalized_data.index) > 10:
            plt.xticks(rotation=45, ha='right')
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{dataset_name}_{col}_class_balance.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_numerical_correlation(num_data, dataset_name, target):
    # Check if there are any numerical attributes to analyze
    if len(num_data.columns) == 0:
        return set()
    
    correlations = num_data.corr(method='pearson')
    
    # Create plot correlation matrix with increased size and rotated labels
    plt.figure(figsize=(16, 14))
    
    # Create the heatmap with adjusted parameters
    heatmap = sns.heatmap(correlations, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 8})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    
    # Adjust margins to make room for labels
    plt.subplots_adjust(left=0.2, bottom=0.2)
    
    plt.title(f'Correlation Matrix - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_num_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    correlations_list = []
    abandoned_attributs = set()

    for i in range(len(correlations.columns)):
        for j in range(0, i):
            correlation_value = correlations.iloc[i, j]
            # High correlation threshold
            if abs(correlation_value) > 0.8:
                attribute1 = correlations.columns[i]
                attribute2 = correlations.columns[j]
                correlations_list.append((attribute1, attribute2))

    # Process the correlated pairs to decide which attributes to drop
    for (attribute1, attribute2) in correlations_list:
        # Skip this pair if either attribute is the target
        if target is not None and (attribute1 == target or attribute2 == target):
            continue

        # Only consider the pair if neither attribute has been marked for dropping yet
        if attribute1 not in abandoned_attributs and attribute2 not in abandoned_attributs:
            # Drop the attribute with more missing values
            if num_data[attribute1].count() < num_data[attribute2].count():
                abandoned_attributs.add(attribute1)
            else:
                abandoned_attributs.add(attribute2)
    

    return abandoned_attributs

def analyze_categorical_correlation(cat_data, dataset_name, target):
    # Check if there are any categorical attributes to analyze
    if len(cat_data.columns) == 0:
        return set()
    
    # Initialize p-value matrix
    p_value_matrix = pd.DataFrame(index=cat_data.columns, columns=cat_data.columns, dtype=float)
    
    # Fill all values with 1.0 initially (no correlation)
    p_value_matrix = p_value_matrix.fillna(1.0)

    # Fill diagonal with zeros (initially perfect self-correlation)
    for col in cat_data.columns:
        p_value_matrix.loc[col, col] = 0
    
    # Track correlated pairs
    correlations_list = []
    
    # Compute Chi-Square tests for all pairs of attributes
    for i in range(len(cat_data.columns)):
        for j in range(0, i):  # Only compute lower triangle to avoid redundant calculations
            col1 = cat_data.columns[i]
            col2 = cat_data.columns[j]
            
            # Create contingency table
            contingency_table = pd.crosstab(cat_data[col1], cat_data[col2])
            
            # Check if the table is valid for Chi-Square test
            # (should have at least 2 rows and 2 columns with non-zero values)
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                # Calculate Chi-Square test
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                
                # Store p-value symmetrically
                p_value_matrix.loc[col1, col2] = p_value
                p_value_matrix.loc[col2, col1] = p_value
                
                # If p-value is significant (≤ 0.05), record correlation
                if p_value <= 0.05:
                    correlations_list.append((col1, col2))
            else:
                # For invalid tables, set p-value to NaN
                p_value_matrix.loc[col1, col2] = np.nan
                p_value_matrix.loc[col2, col1] = np.nan
    
    # Create visualization of p-value matrix
    plt.figure(figsize=(16, 14))
    
    # Create the heatmap with adjusted parameters
    sns.heatmap(p_value_matrix, annot=True, fmt=".3f", cmap='viridis_r', square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 8},
        vmin=0, vmax=0.1)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    
    # Adjust margins to make room for labels
    plt.subplots_adjust(left=0.2, bottom=0.2)
    
    plt.title(f'Chi-Square Test P-Values - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_categorical_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Initialize set of attributes to drop
    abandoned_attributs = set()
    
    # Process the correlated pairs to decide which attributes to drop
    for (attribute1, attribute2) in correlations_list:
        # Skip this pair if either attribute is the target
        if target is not None and (attribute1 == target or attribute2 == target):
            continue

        # Only consider the pair if neither attribute has been marked for dropping yet
        if attribute1 not in abandoned_attributs and attribute2 not in abandoned_attributs:
            # Drop the attribute with more missing values
            if cat_data[attribute1].count() < cat_data[attribute2].count():
                abandoned_attributs.add(attribute1)
            else:
                abandoned_attributs.add(attribute2)
    
    return abandoned_attributs

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

def main():
    heart_set = load_heart_set()
    news_set = load_news_set()

    # 3.1.1
    heart_num_data, heart_cat_data = separate_data_heart_set(heart_set)
    news_num_data, news_cat_data = separate_data_news_set(news_set)

    heart_num_stats = analyze_num_data(heart_num_data, "Heart Disease")
    news_num_stats = analyze_num_data(news_num_data, "News Popularity")

    heart_cat_data_stats = analyze_cat_data(heart_cat_data, "Heart Disease")
    news_cat_data_stats = analyze_cat_data(news_cat_data, "News Popularity")

    # 3.1.2
    # analyze_class_balance(heart_set['heart_train'], "Heart_Disease", "chd_risk")
    
    # news_cols_to_analyze = ['popularity_category', 'is_weekend', 'publication_period']
    # analyze_class_balance(news_set['news_train'], "News_Popularity", news_cols_to_analyze)

    # 3.1.3
    # abandoned_heart_num_attributs = analyze_numerical_correlation(heart_num_data, "Heart Disease", "chd_risk")
    # abandoned_heart_cat_attributs = analyze_categorical_correlation(heart_cat_data, "Heart Disease", "chd_risk")
    # all_heart_attributes_to_drop = abandoned_heart_num_attributs.union(abandoned_heart_cat_attributs)
    
    # abandoned_news_num_attributs = analyze_numerical_correlation(news_num_data, "News Popularity", "popularity_category")
    # abandoned_news_cat_attributs = analyze_categorical_correlation(news_cat_data, "News Popularity", "popularity_category")
    # all_news_attributes_to_drop = abandoned_news_num_attributs.union(abandoned_news_cat_attributs)

    # # print(f"Heart Disease - Attributes to drop: {list(all_heart_attributes_to_drop)}")
    # # print(f"News Popularity - Attributes to drop: {list(all_news_attributes_to_drop)}")

    # # 3.2.1
    # # Impute missing values in numerical data
    # filled_heart_data = process_heart_disease_missing_data(heart_set, heart_num_data, heart_cat_data)
    # filled_news_data = process_news_popularity_missing_data(news_set, news_num_data, news_cat_data)

    # heart_set['heart_train_filled'] = filled_heart_data
    # news_set['news_train_filled'] = filled_news_data

    # Print filled data info
    # print("Heart Disease Data after filling missing values:")
    # print_filled_data_info(filled_heart_data)

    # print("News Popularity Data after filling missing values:")
    # print_filled_data_info(filled_news_data)

    # filled_heart_data.to_csv('filled_heart_data.csv', index=False)
    # filled_news_data.to_csv('filled_news_data.csv', index=False)
    
    # 3.2.2
    # Replace extreme numerical values with an imputed value
    # replaced_extreme_heart_num_data = replace_outliers_data(heart_set['heart_train_filled'], heart_num_data)
    # replaced_extreme_news_num_data = replace_outliers_data(news_set['news_train_filled'], news_num_data)

    # replaced_extreme_heart_num_data.to_csv('replaced_extreme_heart_num_data.csv', index=False)
    # replaced_extreme_news_num_data.to_csv('replaced_extreme_news_num_data.csv', index=False)

    # 3.2.3
    # + uncomment 3.1.3
    # Remove redundant attributes from Heart Disease dataset
    # heart_final = remove_redundant_heart_attributes(all_heart_attributes_to_drop, replaced_extreme_heart_num_data)

    # Remove redundant attributes from News dataset  
    # news_final = remove_redundant_news_attributes(all_news_attributes_to_drop, replaced_extreme_news_num_data)

    # Save final cleaned datasets
    # heart_final.to_csv('heart_final_cleaned.csv', index=False)
    # news_final.to_csv('news_final_cleaned.csv', index=False)

    # 3.2.4
    # Standardize numerical data
    # standardized_heart_data = standardize_numerical_data(heart_final, heart_num_data)
    # standardized_news_data = standardize_numerical_data(news_final, news_num_data)

    # standardized_heart_data.to_csv('standardized_heart_data.csv', index=False)
    # standardized_news_data.to_csv('standardized_news_data.csv', index=False)

    # 3.3.2 - Random Forest Classifier
    # Prepare data for Random Forest Classifier
    # heart_train_data, heart_test_data = prepare_heart_data_for_rf(heart_set, all_heart_attributes_to_drop, heart_num_data, heart_cat_data)
    # news_train_data, news_test_data = prepare_news_data_for_rf(news_set, all_news_attributes_to_drop, news_num_data, news_cat_data)

if __name__ == "__main__":
    main()