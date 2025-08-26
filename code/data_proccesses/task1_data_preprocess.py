import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_combine_data():
    """
    Load all CSV files and combine them into a single dataset
    """
    # Define file names - ALL 9 categories as required by assignment
    categories = ['accessories', 'bags', 'beauty', 'house', 'jewelry', 'kids', 'men', 'shoes', 'women']
    
    # List to store dataframes
    dataframes = []
    
    # Load each CSV file
    for category in categories:
        try:
            df = pd.read_csv(f'data/{category}.csv')
            print(f"Loaded {category}.csv: {df.shape[0]} rows, {df.shape[1]} columns")
            dataframes.append(df)
        except FileNotFoundError:
            print(f"Warning: {category}.csv not found")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nCombined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    print(f"Categories included: {categories}")
    
    return combined_df

def analyze_data_structure(df):
    """
    Analyze the structure of the combined dataset
    """
    print("=== DATA STRUCTURE ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names and types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    print(f"\nUnique values per column:")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: {df[col].nunique()} unique values")
        else:
            print(f"{col}: {df[col].nunique()} unique values (range: {df[col].min()} - {df[col].max()})")

def select_analysis_columns(df):
    """
    Select columns for analysis based on assignment requirements
    Focus on integer and decimal columns (excluding id)
    """
    print("\n=== COLUMN SELECTION ===")
    
    # Identify numeric columns (integer and decimal types) - as required by assignment
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove 'id' column as specified in assignment
    if 'id' in numeric_columns:
        numeric_columns.remove('id')
        print(f"Excluded 'id' column as per assignment requirements")
    
    print(f"Selected numeric columns for analysis: {numeric_columns}")
    
    # Keep category for target variable and other useful string columns for context
    categorical_columns = ['category', 'subcategory', 'name', 'brand', 'currency']
    
    # Select final columns for analysis
    analysis_columns = numeric_columns + categorical_columns
    analysis_df = df[analysis_columns].copy()
    
    print(f"Final analysis dataset shape: {analysis_df.shape}")
    print(f"Categorical columns kept for context: {categorical_columns}")
    
    return analysis_df, numeric_columns

def handle_missing_values(df, numeric_columns):
    """
    Handle missing values in the dataset
    """
    print("\n=== MISSING VALUE HANDLING ===")
    
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before cleaning: {missing_before}")
    
    # Handle missing values for numeric columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            if col in ['current_price', 'raw_price']:
                # Use median for price columns to avoid outlier impact
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val, inplace=True)
                print(f"Filled missing {col} with median: {median_val:.2f}")
            elif col in ['discount', 'likes_count']:
                # Use 0 for discount and likes_count (reasonable defaults)
                df[col] = df[col].fillna(0, inplace=True)
                print(f"Filled missing {col} with 0")
            else:
                # Use mean for other numeric columns
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val, inplace=True)
                print(f"Filled missing {col} with mean: {mean_val:.2f}")
    
    # Handle missing values for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown', inplace=True)
            print(f"Filled missing {col} with 'Unknown'")
    
    missing_after = df.isnull().sum().sum()
    print(f"Total missing values after cleaning: {missing_after}")
    
    return df

def remove_outliers(df, numeric_columns):
    """
    Remove outliers using IQR method
    """
    print("\n=== OUTLIER REMOVAL ===")
    
    initial_rows = len(df)
    
    for col in numeric_columns:
        if col != 'is_new':  # Skip boolean column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                print(f"{col}: {outliers} outliers removed (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
                # Remove outliers
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    final_rows = len(df)
    print(f"Rows removed: {initial_rows - final_rows} ({((initial_rows - final_rows)/initial_rows)*100:.1f}%)")
    print(f"Final dataset shape: {df.shape}")
    
    return df

def normalize_features(df, numeric_columns):
    """
    Normalize numeric features for clustering and classification
    """
    print("\n=== FEATURE NORMALIZATION ===")
    
    # Create a copy for normalized data
    df_normalized = df.copy()
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Select columns for normalization (exclude boolean is_new)
    norm_columns = [col for col in numeric_columns if col != 'is_new']
    
    # Normalize selected columns
    df_normalized[norm_columns] = scaler.fit_transform(df[norm_columns])
    
    print(f"Normalized columns: {norm_columns}")
    print("Normalization statistics:")
    for col in norm_columns:
        print(f"  {col}: mean={df_normalized[col].mean():.3f}, std={df_normalized[col].std():.3f}")
    
    return df_normalized, scaler

def create_composite_score(df):
    """
    Create composite score for ranking products - DEFINES "TOP" CRITERIA
    """
    print("\n=== COMPOSITE SCORE CREATION ===")
    print("DEFINING 'TOP 10 PRODUCTS' CRITERIA:")
    
    # Normalize individual components to 0-1 scale
    df['price_score'] = 1 - (df['current_price'] - df['current_price'].min()) / (df['current_price'].max() - df['current_price'].min())
    df['discount_score'] = (df['discount'] - df['discount'].min()) / (df['discount'].max() - df['discount'].min())
    df['likes_score'] = (df['likes_count'] - df['likes_count'].min()) / (df['likes_count'].max() - df['likes_count'].min())

    # Calculate weighted composite score
    df['composite_score'] = (
        0.50 * df['likes_score'] +      # Customer engagement (50%)
        0.30 * df['discount_score'] +    # Value proposition (30%)
        0.20 * df['price_score']        # Affordability (20%)
    )
    
    print("Composite score formula:")
    print("  composite_score = 0.50 * likes_score + 0.30 * discount_score + 0.20 * price_score")
    print("Where:")
    print("  - likes_score: Higher customer engagement = higher score")
    print("  - discount_score: Higher discount = higher score") 
    print("  - price_score: Lower price = higher score (affordability)")
    print("\nThis composite score defines our 'TOP' criteria - products with:")
    print("  1. High customer engagement (likes)")
    print("  2. Good value (high discounts)")
    print("  3. Reasonable prices")
    
    return df

def extract_top_products(df):
    """
    Extract TOP 10 PRODUCTS as required by assignment
    """
    print("\n=== EXTRACTING TOP 10 PRODUCTS ===")
    
    # Sort by composite score to get top 10
    top_10_products = df.nlargest(10, 'composite_score')
    
    print("TOP 10 PRODUCTS (Based on composite score):")
    print("-" * 80)
    
    for i, (idx, product) in enumerate(top_10_products.iterrows(), 1):
        print(f"{i:2d}. {product['name'][:50]:<50} | Score: {product['composite_score']:.3f}")
        print(f"    Category: {product['category']:<12} | Price: {product['current_price']:.2f} {product['currency']}")
        print(f"    Discount: {product['discount']:<3.0f}% | Likes: {product['likes_count']:<6.0f} | Brand: {product['brand']}")
        print()
    
    return top_10_products

def determine_best_category(df):
    """
    Determine BEST CATEGORY as required by assignment
    """
    print("\n=== DETERMINING BEST CATEGORY ===")
    print("DEFINING 'BEST CATEGORY' CRITERIA:")
    
    # Calculate category metrics
    category_metrics = df.groupby('category').agg({
        'composite_score': ['mean', 'max', 'count'],
        'likes_count': 'mean',
        'discount': 'mean',
        'current_price': 'mean'
    }).round(3)
    
    # Flatten column names
    category_metrics.columns = ['avg_composite_score', 'max_composite_score', 'product_count', 
                               'avg_likes', 'avg_discount', 'avg_price']
    
    # Create overall category score
    # Normalize metrics to 0-1 scale
    category_metrics['score_norm'] = (category_metrics['avg_composite_score'] - category_metrics['avg_composite_score'].min()) / \
                                    (category_metrics['avg_composite_score'].max() - category_metrics['avg_composite_score'].min())
    
    category_metrics['likes_norm'] = (category_metrics['avg_likes'] - category_metrics['avg_likes'].min()) / \
                                    (category_metrics['avg_likes'].max() - category_metrics['avg_likes'].min())
    
    category_metrics['count_norm'] = (category_metrics['product_count'] - category_metrics['product_count'].min()) / \
                                    (category_metrics['product_count'].max() - category_metrics['product_count'].min())
    
    # Calculate final category score
    category_metrics['final_category_score'] = (
        0.60 * category_metrics['score_norm'] +    # Average product quality
        0.25 * category_metrics['likes_norm'] +    # Customer engagement
        0.15 * category_metrics['count_norm']      # Product variety
    )
    
    # Sort by final score
    category_ranking = category_metrics.sort_values('final_category_score', ascending=False)
    
    print("CATEGORY RANKING (Best to Worst):")
    print("-" * 100)
    print(f"{'Rank':<4} {'Category':<12} {'Final Score':<12} {'Avg Composite':<14} {'Avg Likes':<10} {'Products':<8}")
    print("-" * 100)
    
    for i, (category, metrics) in enumerate(category_ranking.iterrows(), 1):
        print(f"{i:<4} {category:<12} {metrics['final_category_score']:.3f}        "
              f"{metrics['avg_composite_score']:.3f}          {metrics['avg_likes']:.0f}        {metrics['product_count']:.0f}")
    
    best_category = category_ranking.index[0]
    print(f"\n🏆 BEST CATEGORY: {best_category}")
    print(f"   Final Score: {category_ranking.loc[best_category, 'final_category_score']:.3f}")
    print(f"   Average Composite Score: {category_ranking.loc[best_category, 'avg_composite_score']:.3f}")
    print(f"   Average Likes: {category_ranking.loc[best_category, 'avg_likes']:.0f}")
    print(f"   Number of Products: {category_ranking.loc[best_category, 'product_count']:.0f}")
    
    return category_ranking, best_category

def visualize_data_distribution(df, numeric_columns):
    """
    Create visualizations for data distribution
    """
    print("\n=== DATA VISUALIZATION ===")
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create subplots for numeric columns
    n_cols = len([col for col in numeric_columns if col != 'is_new'])
    fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(15, 10))
    axes = axes.ravel()
    
    plot_idx = 0
    for col in numeric_columns:
        if col != 'is_new':  # Skip boolean column for histogram
            axes[plot_idx].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[plot_idx].set_title(f'Distribution of {col}')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel('Frequency')
            plot_idx += 1
    
    # Remove empty subplots
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Category distribution
    plt.figure(figsize=(12, 6))
    category_counts = df['category'].value_counts()
    plt.bar(category_counts.index, category_counts.values)
    plt.title('Distribution of Products by Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main preprocessing pipeline - COMPLETE TASK 1
    """
    print("=" * 60)
    print("CSCI446/946 Assignment 1 - Task 1: Data Preprocessing")
    print("=" * 60)
    
    # Step 1: Load and combine data
    df = load_and_combine_data()
    
    # Step 2: Analyze data structure
    analyze_data_structure(df)
    
    # Step 3: Select columns for analysis
    df_analysis, numeric_columns = select_analysis_columns(df)
    
    # Step 4: Handle missing values
    df_clean = handle_missing_values(df_analysis, numeric_columns)
    
    # Step 5: Remove outliers
    df_no_outliers = remove_outliers(df_clean, numeric_columns)
    
    # Step 6: Create composite score (defines "top" criteria)
    df_with_score = create_composite_score(df_no_outliers)
    
    # Step 7: Extract TOP 10 PRODUCTS (Assignment requirement)
    top_10_products = extract_top_products(df_with_score)
    
    # Step 8: Determine BEST CATEGORY (Assignment requirement)
    category_ranking, best_category = determine_best_category(df_with_score)
    
    # Step 9: Normalize features for clustering/classification
    df_normalized, scaler = normalize_features(df_with_score, numeric_columns)

    # Step 10: Visualize data
    visualize_data_distribution(df_with_score, numeric_columns)
    
    # Step 11: Save processed data
    df_with_score.to_csv('preprocessed_data.csv', index=False)
    df_normalized.to_csv('normalized_data.csv', index=False)
    top_10_products.to_csv('top_10_products.csv', index=False)
    category_ranking.to_csv('category_ranking.csv')

    
    return df_with_score, df_normalized, scaler, numeric_columns, top_10_products, category_ranking, best_category

if __name__ == "__main__":
    processed_data, normalized_data, scaler, numeric_cols, top_10, cat_ranking, best_cat = main()