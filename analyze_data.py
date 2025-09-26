import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency

def load_heart_set():
    heart_set = {}

    heart_set['heart_train'] = pd.read_csv('heart_1_train.csv')
    heart_set['heart_test'] = pd.read_csv('heart_1_test.csv')

    heart_set['heart_train'].columns = heart_set['heart_train'].columns.str.strip()
    heart_set['heart_test'].columns = heart_set['heart_test'].columns.str.strip()

    return heart_set

def load_news_set():
    news_set = {}

    news_set['news_train'] = pd.read_csv('news_popularity_train.csv')
    news_set['news_test'] = pd.read_csv('news_popularity_test.csv')

    news_set['news_train'].columns = news_set['news_train'].columns.str.strip()
    news_set['news_test'].columns = news_set['news_test'].columns.str.strip()

    return news_set

def separate_data_heart_set(heart_set):
    # Separate data between numerical and categorical   
    heart_categorical_cols = [
        'gender', 'education_level', 'stroke_history', 'hypertension_history', 
        'diabetes_history', 'high_blood_sugar', 'smoking_status', 
        'blood_pressure_medication', 'chd_risk'
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
    stats_table = []

    # Generate statistics table
    for col in num_data.columns:
        stats_element = {
            'attribute': col,
            'non_null_count': num_data[col].count(),
            'mean': num_data[col].mean(),
            'std': num_data[col].std(),
            'min': num_data[col].min(),
            '25%': num_data[col].quantile(0.25),
            '50%': num_data[col].quantile(0.5),
            '75%': num_data[col].quantile(0.75),
            'max': num_data[col].max()
        }
        stats_table.append(stats_element)
    
    stats_table_df = pd.DataFrame(stats_table)
    
    # Generate individual boxplots
    for col in num_data.columns:
        data_clean = num_data[col].dropna()
        
        if len(data_clean) > 0:
            plt.figure(figsize=(8, 6))
            
            box_plot = plt.boxplot(data_clean, patch_artist=True, 
                                  boxprops=dict(facecolor='orange', alpha=0.7),
                                  medianprops=dict(color='red', linewidth=2),
                                  whiskerprops=dict(color='black', linewidth=1.5),
                                  capprops=dict(color='black', linewidth=1.5))
            
            plt.title(f'{col}', fontsize=14, fontweight='bold')
            plt.ylabel('Values', fontsize=12)
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add statistics as text
            stats_text = f'Count: {len(data_clean)}\nMean: {data_clean.mean():.2f}\nStd: {data_clean.std():.2f}\nMin: {data_clean.min():.2f}\nMax: {data_clean.max():.2f}'
            plt.text(1.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                    fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f'{dataset_name}_{col}_boxplot.png', dpi=300, bbox_inches='tight')
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

    # For each categorical attribute, generate a distribution histogram
    for col in cat_data.columns:
        
        # Skip URL column as it has too many unique values
        if 'url' in col.lower():
            continue
            
        # Get value counts for the categorical attribute
        value_counts = cat_data[col].value_counts().sort_index()
        
        # Skip columns with too many unique values (more than 50)
        if len(value_counts) > 50:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Convert index values to integers if they are numeric, otherwise keep as strings
        try:
            # Try to convert to int first, then to string (handles 1.0 -> 1)
            x_labels = [str(int(float(x))) for x in value_counts.index]
        except (ValueError, TypeError):
            # If conversion fails, keep original values as strings
            x_labels = [str(x) for x in value_counts.index]
        
        # Create histogram showing distribution of unique values
        bars = plt.bar(x_labels, value_counts.values, 
                      color='steelblue', width=0.6)
        
        # Add value labels on top of bars
        for bar in bars:
            height = int(bar.get_height())  # Convert to integer
            plt.text(bar.get_x() + bar.get_width()/2., height + max(value_counts.values) * 0.01,
                    f'{height:,}',  # Format with comma separators for large numbers
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Distribution: {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'{dataset_name}_{col}_distribution.png', dpi=200, bbox_inches='tight')
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
        percentages = data[col].value_counts(normalize=True).sort_index() * 100  # normalize => [0,1]
        
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