import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_and_combine_data():
    """
    Load all CSV files and combine them into a single dataset
    """
    # Define file names
    categories = [
        "accessories",
        "bags",
        "beauty",
        "house",
        "jewelry",
        "kids",
        "men",
        "shoes",
        "women",
    ]

    # List to store dataframes
    dataframes = []

    # Load each CSV file
    for category in categories:
        try:
            path = f"../assignment/A1_2025_Released/{category}.csv"
            df = pd.read_csv(path)
            print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
            dataframes.append(df)
        except FileNotFoundError:
            print(f"Warning: {path} not found")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(
        f"\nCombined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns"
    )

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
        if df[col].dtype == "object":
            print(f"{col}: {df[col].nunique()} unique values")
        else:
            print(
                f"{col}: {df[col].nunique()} unique values (range: {df[col].min()} - {df[col].max()})"
            )


def select_analysis_columns(df):
    """
    Select columns for analysis based on assignment requirements
    Focus on integer and decimal columns (excluding id)
    """
    print("\n=== COLUMN SELECTION ===")

    # Identify numeric columns (integer and decimal types)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove 'id' column as specified
    if "id" in numeric_columns:
        numeric_columns.remove("id")

    # Check for boolean columns that should be treated as numeric (specifically is_new)
    if "is_new" in df.columns and "is_new" not in numeric_columns:
        # Check if it's actually a boolean column
        if df["is_new"].dtype == "bool" or df["is_new"].dtype == "object":
            numeric_columns.append("is_new")
            print(
                f"Added 'is_new' column to numeric analysis (will be converted to numeric)"
            )

    print(f"Selected numeric columns for analysis: {numeric_columns}")

    # Keep category for target variable and other useful string columns for context
    categorical_columns = ["category", "subcategory", "name", "brand", "currency"]

    # Ensure all requested columns exist in the dataframe
    available_categorical = [col for col in categorical_columns if col in df.columns]
    available_numeric = [col for col in numeric_columns if col in df.columns]

    # Select final columns for analysis
    analysis_columns = available_numeric + available_categorical
    analysis_df = df[analysis_columns].copy()

    print(f"Available numeric columns: {available_numeric}")
    print(f"Available categorical columns: {available_categorical}")
    print(f"Final analysis dataset shape: {analysis_df.shape}")
    return analysis_df, available_numeric


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
            if col in ["current_price", "raw_price"]:
                # Use median for price columns to avoid outlier impact
                df[col].fillna(df[col].median(), inplace=True)
                print(f"Filled missing {col} with median: {df[col].median():.2f}")
            elif col in ["discount", "likes_count"]:
                # Use 0 for discount and likes_count (reasonable defaults)
                df[col].fillna(0, inplace=True)
                print(f"Filled missing {col} with 0")
            else:
                # Use mean for other numeric columns
                df[col].fillna(df[col].mean(), inplace=True)
                print(f"Filled missing {col} with mean: {df[col].mean():.2f}")

    # Handle missing values for categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna("Unknown", inplace=True)
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
        if col != "is_new":  # Skip boolean column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outliers > 0:
                print(
                    f"{col}: {outliers} outliers removed (bounds: {lower_bound:.2f} - {upper_bound:.2f})"
                )
                # Remove outliers
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    final_rows = len(df)
    print(
        f"Rows removed: {initial_rows - final_rows} ({((initial_rows - final_rows)/initial_rows)*100:.1f}%)"
    )
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
    norm_columns = [col for col in numeric_columns if col != "is_new"]

    # Normalize selected columns
    df_normalized[norm_columns] = scaler.fit_transform(df[norm_columns])

    print(f"Normalized columns: {norm_columns}")
    print("Normalization statistics:")
    for col in norm_columns:
        print(
            f"  {col}: mean={df_normalized[col].mean():.3f}, std={df_normalized[col].std():.3f}"
        )

    return df_normalized, scaler


def create_composite_score(df):
    """
    Create composite score for ranking products
    """
    print("\n=== COMPOSITE SCORE CREATION ===")

    # Check if is_new column exists and handle it appropriately
    if "is_new" in df.columns:
        # Convert boolean to numeric if needed
        df["new_score"] = df["is_new"].astype(int)
        print("Converted 'is_new' boolean column to numeric 'new_score'")
    else:
        # Create a default new_score if is_new doesn't exist
        df["new_score"] = 0
        print("Warning: 'is_new' column not found, using default value 0 for new_score")

    # Normalize individual components to 0-1 scale
    df["price_score"] = 1 - (df["current_price"] - df["current_price"].min()) / (
        df["current_price"].max() - df["current_price"].min()
    )
    df["discount_score"] = (df["discount"] - df["discount"].min()) / (
        df["discount"].max() - df["discount"].min()
    )
    df["likes_score"] = (df["likes_count"] - df["likes_count"].min()) / (
        df["likes_count"].max() - df["likes_count"].min()
    )

    # Calculate weighted composite score
    df["composite_score"] = (
        0.40 * df["likes_score"]  # Customer engagement
        + 0.30 * df["discount_score"]  # Value proposition
        + 0.20 * df["price_score"]  # Affordability
        + 0.10 * df["new_score"]  # Innovation
    )

    print("Composite score created with weights:")
    print("  - Likes count: 40%")
    print("  - Discount: 30%")
    print("  - Price (inverse): 20%")
    print("  - New product: 10%")

    return df


def visualize_data_distribution(df, numeric_columns):
    """
    Create visualizations for data distribution
    """
    print("\n=== DATA VISUALIZATION ===")

    # Set up the plotting style
    plt.style.use("default")

    # Filter out boolean columns for histogram plotting
    histogram_columns = [
        col for col in numeric_columns if col not in ["is_new", "new_score"]
    ]

    print(f"Creating histograms for columns: {histogram_columns}")

    if len(histogram_columns) > 0:
        # Calculate subplot layout based on actual columns to plot
        n_cols = len(histogram_columns)
        n_rows = (n_cols + 2) // 3  # Use 3 columns layout for better appearance
        n_subplot_cols = min(3, n_cols)  # Maximum 3 columns

        fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(15, 5 * n_rows))

        # Handle single subplot case
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.ravel()

        # Create histograms for continuous numeric columns
        for i, col in enumerate(histogram_columns):
            axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

        # Remove empty subplots
        total_subplots = n_rows * n_subplot_cols
        for i in range(len(histogram_columns), total_subplots):
            if i < len(axes):
                fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(
            "processed_data/data_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    # Separate visualization for boolean columns
    boolean_columns = [col for col in numeric_columns if col in ["is_new", "new_score"]]
    if boolean_columns:
        fig, axes = plt.subplots(
            1, len(boolean_columns), figsize=(6 * len(boolean_columns), 5)
        )
        if len(boolean_columns) == 1:
            axes = [axes]

        for i, col in enumerate(boolean_columns):
            if col in df.columns:
                value_counts = df[col].value_counts()
                axes[i].bar(
                    value_counts.index.astype(str),
                    value_counts.values,
                    alpha=0.7,
                    edgecolor="black",
                )
                axes[i].set_title(f"Distribution of {col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Count")
                axes[i].grid(True, alpha=0.3)

                # Add value labels on bars
                for j, v in enumerate(value_counts.values):
                    axes[i].text(
                        j,
                        v + max(value_counts.values) * 0.01,
                        str(v),
                        ha="center",
                        va="bottom",
                    )

        plt.tight_layout()
        plt.savefig(
            "processed_data/boolean_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    # Category distribution
    if "category" in df.columns:
        plt.figure(figsize=(12, 6))
        category_counts = df["category"].value_counts()
        bars = plt.bar(
            category_counts.index, category_counts.values, alpha=0.7, edgecolor="black"
        )
        plt.title("Distribution of Products by Category")
        plt.xlabel("Category")
        plt.ylabel("Number of Products")
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, category_counts.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(category_counts.values) * 0.01,
                str(count),
                ha="center",
                va="bottom",
            )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            "processed_data/category_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    print("Visualizations saved:")
    print("  - data_distribution.png (continuous variables)")
    if boolean_columns:
        print("  - boolean_distribution.png (boolean variables)")
    if "category" in df.columns:
        print("  - category_distribution.png (category distribution)")


def main():
    """
    Main preprocessing pipeline
    """
    print("Starting data preprocessing pipeline...")

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

    # Step 6: Create composite score
    df_with_score = create_composite_score(df_no_outliers)

    # Step 7: Normalize features
    df_normalized, scaler = normalize_features(df_with_score, numeric_columns)

    # Step 8: Visualize data
    visualize_data_distribution(df_with_score, numeric_columns)

    # Step 9: Save processed data
    df_with_score.to_csv("preprocessed_data.csv", index=False)
    df_normalized.to_csv("normalized_data.csv", index=False)

    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"Preprocessed data saved as 'preprocessed_data.csv'")
    print(f"Normalized data saved as 'normalized_data.csv'")
    print(f"Final dataset shape: {df_with_score.shape}")

    return df_with_score, df_normalized, scaler, numeric_columns


if __name__ == "__main__":
    processed_data, normalized_data, scaler, numeric_cols = main()
