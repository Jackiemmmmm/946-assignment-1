import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def load_and_combine_data():
    """
    Enhanced version of data loading with better error handling
    """
    print("=== ENHANCED DATA LOADING ===")
    
    categories = [
        "accessories", "bags", "beauty", "house", "jewelry", 
        "kids", "men", "shoes", "women"
    ]
    
    dataframes = []
    
    for category in categories:
        try:
            path = f"../assignment/A1_2025_Released/{category}.csv"
            df = pd.read_csv(path)
            print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
            dataframes.append(df)
        except FileNotFoundError:
            print(f"Warning: {path} not found")
    
    if not dataframes:
        print("Error: No data files found!")
        return None
        
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nCombined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    
    return combined_df


def advanced_feature_engineering(df):
    """
    Create advanced engineered features
    """
    print("\n=== ADVANCED FEATURE ENGINEERING ===")
    
    df_enhanced = df.copy()
    
    # 1. Price-based features
    if 'current_price' in df.columns and 'raw_price' in df.columns:
        # Price efficiency ratio
        df_enhanced['price_efficiency'] = df_enhanced['current_price'] / (df_enhanced['raw_price'] + 1e-6)
        
        # Price tier (categorical)
        price_quantiles = df_enhanced['current_price'].quantile([0.25, 0.5, 0.75])
        df_enhanced['price_tier'] = pd.cut(
            df_enhanced['current_price'], 
            bins=[-np.inf, price_quantiles[0.25], price_quantiles[0.5], price_quantiles[0.75], np.inf],
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        # Absolute discount amount
        df_enhanced['discount_amount'] = df_enhanced['raw_price'] - df_enhanced['current_price']
        print("Created price-based features: price_efficiency, price_tier, discount_amount")
    
    # 2. Popularity features
    if 'likes_count' in df.columns:
        # Popularity tier
        likes_quantiles = df_enhanced['likes_count'].quantile([0.33, 0.67])
        df_enhanced['popularity_tier'] = pd.cut(
            df_enhanced['likes_count'],
            bins=[-np.inf, likes_quantiles[0.33], likes_quantiles[0.67], np.inf],
            labels=['Low', 'Medium', 'High']
        )
        
        # Log-normalized likes (for clustering)
        df_enhanced['likes_log'] = np.log1p(df_enhanced['likes_count'])
        print("Created popularity features: popularity_tier, likes_log")
    
    # 3. Value proposition features
    if all(col in df.columns for col in ['current_price', 'discount', 'likes_count']):
        # Value score: high likes, low price, high discount
        df_enhanced['value_score'] = (
            (df_enhanced['likes_count'] / (df_enhanced['likes_count'].max() + 1)) * 0.4 +
            (df_enhanced['discount'] / (df_enhanced['discount'].max() + 1)) * 0.4 +
            (1 - df_enhanced['current_price'] / (df_enhanced['current_price'].max() + 1)) * 0.2
        )
        
        # Price per like ratio
        df_enhanced['price_per_like'] = df_enhanced['current_price'] / (df_enhanced['likes_count'] + 1)
        print("Created value features: value_score, price_per_like")
    
    # 4. Category-based features
    if 'category' in df.columns:
        # Category size (number of products in each category)
        category_counts = df_enhanced['category'].value_counts()
        df_enhanced['category_size'] = df_enhanced['category'].map(category_counts)
        
        # Category average price
        category_avg_price = df_enhanced.groupby('category')['current_price'].transform('mean')
        df_enhanced['category_price_ratio'] = df_enhanced['current_price'] / category_avg_price
        print("Created category features: category_size, category_price_ratio")
    
    # 5. Brand features
    if 'brand' in df.columns:
        # Brand frequency
        brand_counts = df_enhanced['brand'].value_counts()
        df_enhanced['brand_frequency'] = df_enhanced['brand'].map(brand_counts).fillna(1)
        
        # Brand average price
        brand_avg_price = df_enhanced.groupby('brand')['current_price'].transform('mean')
        df_enhanced['brand_price_premium'] = df_enhanced['current_price'] / brand_avg_price
        df_enhanced['brand_price_premium'].fillna(1, inplace=True)
        print("Created brand features: brand_frequency, brand_price_premium")
    
    # 6. Statistical features
    numeric_cols = ['current_price', 'discount', 'likes_count']
    available_numeric = [col for col in numeric_cols if col in df_enhanced.columns]
    
    if len(available_numeric) >= 2:
        # Feature interactions
        df_enhanced['price_discount_interaction'] = df_enhanced['current_price'] * df_enhanced['discount']
        df_enhanced['discount_likes_interaction'] = df_enhanced['discount'] * df_enhanced['likes_count']
        print("Created interaction features")
    
    return df_enhanced


def intelligent_outlier_handling(df, method='iqr_adaptive'):
    """
    Intelligent outlier detection and handling
    """
    print(f"\n=== INTELLIGENT OUTLIER HANDLING ({method}) ===")
    
    df_clean = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Exclude ID and boolean columns
    numeric_columns = [col for col in numeric_columns if col not in ['id', 'is_new', 'new_score']]
    
    outlier_stats = {}
    
    for col in numeric_columns:
        initial_count = len(df_clean)
        
        if method == 'iqr_adaptive':
            # Adaptive IQR based on data distribution
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Adjust multiplier based on skewness
            skewness = abs(stats.skew(df_clean[col]))
            if skewness > 2:  # Highly skewed
                multiplier = 2.5
            elif skewness > 1:  # Moderately skewed
                multiplier = 2.0
            else:  # Normal-ish
                multiplier = 1.5
                
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(df_clean[col]))
            threshold = 3
            outlier_mask = z_scores > threshold
            df_clean = df_clean[~outlier_mask]
            
        elif method == 'isolation_forest':
            # Using isolation forest for multivariate outliers
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(df_clean[[col]])
            df_clean = df_clean[outlier_labels == 1]
        
        if method == 'iqr_adaptive':
            outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            df_clean = df_clean[~outlier_mask]
        
        final_count = len(df_clean)
        removed_count = initial_count - final_count
        
        outlier_stats[col] = {
            'removed': removed_count,
            'percentage': (removed_count / initial_count) * 100
        }
        
        if removed_count > 0:
            print(f"{col}: removed {removed_count} outliers ({removed_count/initial_count*100:.1f}%)")
    
    print(f"Total rows after outlier removal: {len(df_clean)}")
    return df_clean, outlier_stats


def advanced_feature_scaling(df, method='robust'):
    """
    Advanced feature scaling with multiple methods
    """
    print(f"\n=== ADVANCED FEATURE SCALING ({method}) ===")
    
    df_scaled = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Exclude categorical encodings and ID columns
    scale_columns = [col for col in numeric_columns 
                    if col not in ['id', 'is_new', 'new_score'] and 
                    not col.endswith('_tier') and
                    not col.endswith('_frequency')]
    
    if method == 'robust':
        scaler = RobustScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    else:
        scaler = StandardScaler()
    
    # Apply scaling
    df_scaled[scale_columns] = scaler.fit_transform(df_scaled[scale_columns])
    
    print(f"Scaled {len(scale_columns)} columns using {method} scaling")
    print("Scaling statistics:")
    for col in scale_columns[:5]:  # Show first 5 columns
        print(f"  {col}: mean={df_scaled[col].mean():.3f}, std={df_scaled[col].std():.3f}")
    
    return df_scaled, scaler


def intelligent_feature_selection(X, y, method='mutual_info', k=10):
    """
    Intelligent feature selection using multiple methods
    """
    print(f"\n=== INTELLIGENT FEATURE SELECTION ({method}) ===")
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    elif method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    if hasattr(X, 'columns'):
        selected_features = X.columns[selector.get_support()]
        feature_scores = selector.scores_[selector.get_support()]
        
        print(f"Selected {k} best features:")
        for feature, score in zip(selected_features, feature_scores):
            print(f"  {feature}: {score:.3f}")
        
        return X_selected, selected_features, selector
    else:
        return X_selected, None, selector


def create_dimensionality_reduction_features(X, method='pca', n_components=5):
    """
    Create dimensionality reduction features
    """
    print(f"\n=== DIMENSIONALITY REDUCTION ({method}) ===")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=min(n_components, 3), random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=n_components, random_state=42)
    
    X_reduced = reducer.fit_transform(X)
    
    if method == 'pca':
        print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {reducer.explained_variance_ratio_.cumsum()}")
    
    return X_reduced, reducer


def enhanced_data_visualization(df, output_dir='enhanced_visuals/'):
    """
    Create enhanced visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== ENHANCED DATA VISUALIZATION ===")
    
    # 1. Feature correlation heatmap
    plt.figure(figsize=(15, 12))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True)
    plt.title('Enhanced Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Feature importance via mutual information
    if 'category' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(df['category'])
        
        # Select numeric features for analysis
        feature_cols = [col for col in numeric_df.columns if col not in ['id']]
        if feature_cols:
            X_features = numeric_df[feature_cols].fillna(0)
            
            # Calculate mutual information
            mi_scores = mutual_info_classif(X_features, y_encoded, random_state=42)
            
            # Create feature importance plot
            plt.figure(figsize=(12, 8))
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': mi_scores
            }).sort_values('importance', ascending=True)
            
            plt.barh(range(len(feature_importance)), feature_importance['importance'])
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.xlabel('Mutual Information Score')
            plt.title('Feature Importance (Mutual Information)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}feature_importance_mi.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 3. Enhanced distribution plots
    numeric_columns = [col for col in numeric_df.columns if col not in ['id', 'is_new']]
    if numeric_columns:
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(numeric_columns[:len(axes)]):
            if i < len(axes):
                # Create both histogram and box plot
                ax = axes[i]
                
                # Subplot for histogram
                ax_hist = ax
                ax_hist.hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
                ax_hist.set_title(f'{col} Distribution')
                ax_hist.set_xlabel(col)
                ax_hist.set_ylabel('Frequency')
                ax_hist.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = df[col].mean()
                median_val = df[col].median()
                skew_val = stats.skew(df[col].dropna())
                
                stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nSkew: {skew_val:.2f}'
                ax_hist.text(0.7, 0.8, stats_text, transform=ax_hist.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Remove empty subplots
        for i in range(len(numeric_columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}enhanced_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"Enhanced visualizations saved to {output_dir}")


def main_enhanced_preprocessing():
    """
    Main enhanced preprocessing pipeline
    """
    print("Starting Enhanced Data Preprocessing Pipeline...")
    
    # Step 1: Load data
    df = load_and_combine_data()
    if df is None:
        return None
    
    # Step 2: Advanced feature engineering
    df_engineered = advanced_feature_engineering(df)
    
    # Step 3: Handle missing values (improved)
    print("\n=== ENHANCED MISSING VALUE HANDLING ===")
    # For numeric columns, use median for skewed distributions, mean for normal
    numeric_columns = df_engineered.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if df_engineered[col].isnull().sum() > 0:
            skewness = abs(stats.skew(df_engineered[col].dropna()))
            if skewness > 1:  # Skewed distribution
                fill_value = df_engineered[col].median()
                method = "median"
            else:  # Normal-ish distribution
                fill_value = df_engineered[col].mean()
                method = "mean"
            
            df_engineered[col].fillna(fill_value, inplace=True)
            print(f"Filled {col} with {method}: {fill_value:.3f}")
    
    # For categorical columns
    categorical_cols = df_engineered.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_engineered[col].isnull().sum() > 0:
            df_engineered[col].fillna("Unknown", inplace=True)
    
    # Step 4: Intelligent outlier handling
    df_clean, outlier_stats = intelligent_outlier_handling(df_engineered, method='iqr_adaptive')
    
    # Step 5: Advanced feature scaling
    df_scaled, scaler = advanced_feature_scaling(df_clean, method='robust')
    
    # Step 6: Enhanced visualization
    enhanced_data_visualization(df_scaled)
    
    # Step 7: Save enhanced data
    df_clean.to_csv('enhanced_preprocessed_data.csv', index=False)
    df_scaled.to_csv('enhanced_normalized_data.csv', index=False)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'enhanced_scaler.pkl')
    
    print("\n=== ENHANCED PREPROCESSING COMPLETE ===")
    print("New features created:")
    print("  • Price efficiency and tiers")
    print("  • Popularity metrics")
    print("  • Value proposition scores")
    print("  • Category and brand features")
    print("  • Statistical interactions")
    print("  • Advanced outlier handling")
    print("  • Robust feature scaling")
    
    print(f"\nEnhanced dataset shape: {df_scaled.shape}")
    print(f"Preprocessed data saved as 'enhanced_preprocessed_data.csv'")
    print(f"Scaled data saved as 'enhanced_normalized_data.csv'")
    
    return df_clean, df_scaled, scaler


if __name__ == "__main__":
    enhanced_data, scaled_data, scaler = main_enhanced_preprocessing()