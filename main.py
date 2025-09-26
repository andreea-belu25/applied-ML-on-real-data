
from sklearn.experimental import enable_iterative_imputer

from analyze_data import (
    load_heart_set, load_news_set,
    separate_data_heart_set, separate_data_news_set,
    analyze_num_data, analyze_cat_data, analyze_class_balance,
    analyze_numerical_correlation, analyze_categorical_correlation
)

from data_preprocessing import (
    process_heart_disease_missing_data,
    process_news_popularity_missing_data,
    standardize_numerical_data,
    replace_outliers_data,
    remove_redundant_heart_attributes,
    remove_redundant_news_attributes
)

from random_forests import run_random_forest_experiments
from mlp import run_mlp_experiments, plot_mlp_learning_curves
from decision_tree import run_decision_tree_experiments
from comparisons import run_complete_comparison

def main():
    heart_set = load_heart_set()
    news_set = load_news_set()

    # 3.1.1
    heart_num_data, heart_cat_data = separate_data_heart_set(heart_set)
    news_num_data, news_cat_data = separate_data_news_set(news_set) 

    print("\nGenerating numerical data statistics...")
    heart_num_stats = analyze_num_data(heart_num_data, "Heart_Disease")
    news_num_stats = analyze_num_data(news_num_data, "News_Popularity")

    print("\nGenerating categorical data statistics...")
    heart_cat_data_stats = analyze_cat_data(heart_cat_data, "Heart_Disease")
    news_cat_data_stats = analyze_cat_data(news_cat_data, "News_Popularity")

    # 3.1.2
    print("Analyzing Heart Disease class balance...")
    heart_cols_to_analyze = ['chd_risk', 'heart_rate', 'daily_cigarettes']
    analyze_class_balance(heart_set['heart_train'], "Heart_Disease", heart_cols_to_analyze)
    
    print("Analyzing News Popularity class balance...")
    news_cols_to_analyze = ['popularity_category', 'is_weekend', 'publication_period']
    analyze_class_balance(news_set['news_train'], "News_Popularity", news_cols_to_analyze)

    # 3.1.3
    print("Analyzing correlations in Heart Disease dataset...")
    abandoned_heart_num_attributs = analyze_numerical_correlation(heart_num_data, "Heart_Disease", "chd_risk")
    abandoned_heart_cat_attributs = analyze_categorical_correlation(heart_cat_data, "Heart_Disease", "chd_risk")
    all_heart_attributes_to_drop = abandoned_heart_num_attributs.union(abandoned_heart_cat_attributs)
    
    print("Analyzing correlations in News Popularity dataset...")
    abandoned_news_num_attributs = analyze_numerical_correlation(news_num_data, "News_Popularity", "popularity_category")
    abandoned_news_cat_attributs = analyze_categorical_correlation(news_cat_data, "News_Popularity", "popularity_category")
    all_news_attributes_to_drop = abandoned_news_num_attributs.union(abandoned_news_cat_attributs)

    print(f"Heart Disease - Attributes to drop: {len(all_heart_attributes_to_drop)} ({list(all_heart_attributes_to_drop)})")
    print(f"News Popularity - Attributes to drop: {len(all_news_attributes_to_drop)} ({list(all_news_attributes_to_drop)})")

    # 3.2.1
    print("Processing missing values in Heart Disease dataset...")
    filled_heart_data = process_heart_disease_missing_data(heart_set, heart_num_data, heart_cat_data)
    
    print("Processing missing values in News Popularity dataset...")
    filled_news_data = process_news_popularity_missing_data(news_set, news_num_data, news_cat_data)

    heart_set['heart_train_filled'] = filled_heart_data
    news_set['news_train_filled'] = filled_news_data

    # 3.2.2
    print("Replacing extreme values in Heart Disease dataset...")
    replaced_extreme_heart_data = replace_outliers_data(heart_set['heart_train_filled'], heart_num_data)
    
    print("Replacing extreme values in News Popularity dataset...")
    replaced_extreme_news_data = replace_outliers_data(news_set['news_train_filled'], news_num_data)

    # 3.2.3
    print("Removing redundant attributes from Heart Disease dataset...")
    heart_final = remove_redundant_heart_attributes(all_heart_attributes_to_drop, replaced_extreme_heart_data)
    
    print("Removing redundant attributes from News Popularity dataset...")
    news_final = remove_redundant_news_attributes(all_news_attributes_to_drop, replaced_extreme_news_data)
    news_final.to_csv("news_final_cleaned.csv", index=False)

    # 3.2.4
    print("Standardizing numerical features in Heart Disease dataset...")
    standardized_heart_data = standardize_numerical_data(heart_final, heart_num_data)
    
    print("Standardizing numerical features in News Popularity dataset...")
    standardized_news_data = standardize_numerical_data(news_final, news_num_data)

    4.1
    print("Running Decision Tree experiments on both datasets...")
    dt_results = run_decision_tree_experiments(
        heart_set, news_set, all_heart_attributes_to_drop, all_news_attributes_to_drop,
        heart_num_data, heart_cat_data, news_num_data, news_cat_data)
    
    # # 4.2
    print("Running Random Forest experiments on both datasets...")
    rf_results = run_random_forest_experiments(
            heart_set, news_set, all_heart_attributes_to_drop, all_news_attributes_to_drop,
            heart_num_data, heart_cat_data, news_num_data, news_cat_data)
        
    # 4.3
    print("Running MLP experiments on both datasets...")
    mlp_results = run_mlp_experiments(
            heart_set, news_set, all_heart_attributes_to_drop, all_news_attributes_to_drop,
            heart_num_data, heart_cat_data, news_num_data, news_cat_data)
    
    # Plot MLP learning curves
    plot_mlp_learning_curves(mlp_results)
    
    comparison_df = run_complete_comparison(dt_results, rf_results, mlp_results)
    
if __name__ == "__main__":
    main()