# ML Classification: News Popularity & Heart Disease Risk

Machine learning classification pipeline comparing Decision Trees, Random Forests, and Multi-Layer Perceptrons on two real-world datasets with comprehensive exploratory data analysis and visual diagnostics.

A more comprehensive documentation can be found in `multi-domain-data-analysis.pdf`.
  
## Datasets

**Online News Popularity** (39,644 articles, 64 features)
- Multi-class classification: 5 popularity categories
- Features: content metrics, sentiment, multimedia, temporal patterns, keyword performance

**Coronary Heart Disease Risk** (4,000+ patients, 19 features)
- Binary classification: 10-year CHD risk prediction
- Features: demographics, vital signs, metabolic indicators, lifestyle, medical history

## Analysis Pipeline

### 1. Exploratory Data Analysis
- Statistical summaries with **boxplots** for continuous variables
- **Histograms** for categorical distributions
- **Correlation heatmaps** (Pearson for numerical, Chi-square for categorical)
- Class imbalance visualization with bar charts

### 2. Data Preprocessing
- Missing value imputation (SimpleImputer, IterativeImputer)
- Outlier detection via IQR method and replacement
- Feature elimination: removed 8 redundant features (Heart) and 22 (News) based on correlation >0.8
- Standardization (StandardScaler) and encoding (Label/OneHot)

### 3. Model Training & Evaluation

| Algorithm | Heart Disease | News Popularity |
|-----------|---------------|-----------------|
| **Decision Tree** | Acc: 84.79%, F1: 0.78 | Acc: 56.98%, F1: 0.62 |
| **Random Forest** | Acc: 64.98%, F1: 0.70 | Acc: 58.92%, F1: 0.67 |
| **MLP Neural Net** | Acc: 84.79%, F1: 0.78 | Acc: 68.44%, F1: 0.65 |

## Key Visualizations

**Confusion Matrices** - Comparative 6-panel grid showing classification performance across algorithms
- Reveals Decision Tree/MLP miss 97% of heart disease risk cases despite high accuracy
- Random Forest achieves 52% recall on risk detection vs 2.3% for others

**MLP Learning Curves** - Training vs validation loss and accuracy
- Heart Disease: stable convergence, minimal overfitting
- News Popularity: training accuracy ~100%, validation ~90% (overfitting detected)

**Correlation Matrices** - Feature relationship heatmaps
- Identified redundant pairs: `glucose`↔`blood_sugar_level`, `daily_cigarettes`↔`total_cigarettes`
- Guided dimensionality reduction strategy

**Distribution Analysis** - Boxplots revealing outlier patterns
- Heart: extreme values in smoking habits, blood pressure
- News: highly skewed engagement metrics, keyword performance
