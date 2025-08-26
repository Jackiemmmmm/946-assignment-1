import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
import warnings

warnings.filterwarnings("ignore")

# Set matplotlib to use English and handle display issues
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

# If you want to support Chinese characters, uncomment the following lines:
# plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False


def load_clustering_results():
    """
    Load clustering results from previous analysis
    """
    try:
        # Try to load existing clustering results
        clustering_df = pd.read_csv("clustering/clustering_results.csv")
        print("Loaded existing clustering results")
        print(f"Clustering results shape: {clustering_df.shape}")

        # Display available clustering columns
        clustering_columns = [
            col for col in clustering_df.columns if "cluster" in col.lower()
        ]
        if clustering_columns:
            print(f"Available clustering features: {clustering_columns}")
            for col in clustering_columns:
                unique_vals = clustering_df[col].nunique()
                print(f"  {col}: {unique_vals} unique values")

        return clustering_df
    except FileNotFoundError:
        print("Error: No clustering results found!")
        print(
            "Please ensure 'clustering/clustering_results.csv' exists with clustering labels."
        )
        return None


def prepare_enhanced_classification_data(df):
    """
    Prepare data for classification analysis with clustering features
    """
    print("=== ENHANCED CLASSIFICATION DATA PREPARATION ===")

    # Load existing clustering results
    clustering_df = load_clustering_results()

    if clustering_df is None:
        print("Cannot proceed without clustering results!")
        print("Please ensure your clustering analysis has been completed and saved.")
        return None, None, None, None, None

    # Merge with existing clustering results
    # Assume both dataframes have the same index/order
    print("Merging original data with clustering results...")

    # Identify clustering columns
    clustering_columns = [
        col
        for col in clustering_df.columns
        if "cluster" in col.lower() or "label" in col.lower()
    ]

    if not clustering_columns:
        print("Warning: No clustering columns found in the clustering results!")
        print("Looking for columns containing 'cluster' or 'label'")
        print(f"Available columns: {list(clustering_df.columns)}")
        # Ask user to specify clustering columns or use all numeric columns
        numeric_cols = clustering_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print(f"Using numeric columns as clustering features: {numeric_cols}")
            clustering_columns = numeric_cols
        else:
            print("No suitable clustering columns found!")
            return None, None, None, None, None

    print(f"Using clustering features: {clustering_columns}")

    # Merge dataframes
    if len(df) == len(clustering_df):
        # Same length, assume same order
        for col in clustering_columns:
            df[col] = clustering_df[col].values
    else:
        print(f"Length mismatch: df={len(df)}, clustering_df={len(clustering_df)}")
        print("Attempting merge by index...")
        df = df.merge(
            clustering_df[clustering_columns],
            left_index=True,
            right_index=True,
            how="left",
        )

    print("Successfully merged clustering features!")

    # Original features
    feature_columns = ["current_price", "discount", "likes_count"]

    # Add is_new if available
    if "is_new" in df.columns:
        df["is_new_numeric"] = df["is_new"].astype(int)
        feature_columns.append("is_new_numeric")

    # Add clustering features
    clustering_features = clustering_columns

    # Combine all features
    all_features = feature_columns + clustering_features

    print(f"Original features: {feature_columns}")
    print(f"Clustering features: {clustering_features}")
    print(f"Total features: {all_features}")

    # Target variable
    target_column = "category"

    # Check if required columns exist
    missing_cols = [
        col for col in all_features + [target_column] if col not in df.columns
    ]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None, None, None, None, None

    # Extract features and target
    X = df[all_features].copy()
    y = df[target_column].copy()

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"Enhanced classification dataset shape: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")

    # Class distribution
    class_counts = pd.Series(y).value_counts()
    print(f"\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(y)*100:.1f}%)")

    # Clustering feature statistics
    if clustering_features:
        print(f"\nClustering feature statistics:")
        for feature in clustering_features:
            if feature in X.columns:
                unique_vals = X[feature].nunique()
                value_range = f"[{X[feature].min()}, {X[feature].max()}]"
                print(f"  {feature}: {unique_vals} unique values, range {value_range}")

    return X, y_encoded, label_encoder, all_features, clustering_features


def compare_with_without_clustering(df):
    """
    Compare classification performance with and without clustering features
    """
    print("=== COMPARING CLASSIFICATION WITH/WITHOUT CLUSTERING ===")

    # Load clustering results first
    clustering_df = load_clustering_results()
    if clustering_df is None:
        print("Cannot perform comparison without clustering results!")
        return None

    # Prepare data WITHOUT clustering features
    print("\n--- Preparing data WITHOUT clustering features ---")
    feature_columns_original = ["current_price", "discount", "likes_count"]
    if "is_new" in df.columns:
        df["is_new_numeric"] = df["is_new"].astype(int)
        feature_columns_original.append("is_new_numeric")

    X_original = df[feature_columns_original].copy()
    y = df["category"].copy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Prepare data WITH clustering features
    print("\n--- Preparing data WITH clustering features ---")

    # Merge with clustering results
    df_enhanced = df.copy()

    # Identify clustering columns
    clustering_columns = [
        col
        for col in clustering_df.columns
        if "cluster" in col.lower() or "label" in col.lower()
    ]

    if not clustering_columns:
        # Use all numeric columns as potential clustering features
        clustering_columns = clustering_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

    print(f"Using clustering features: {clustering_columns}")

    # Merge clustering features
    if len(df_enhanced) == len(clustering_df):
        for col in clustering_columns:
            df_enhanced[col] = clustering_df[col].values
    else:
        df_enhanced = df_enhanced.merge(
            clustering_df[clustering_columns],
            left_index=True,
            right_index=True,
            how="left",
        )

    # Enhanced features
    all_features_enhanced = feature_columns_original + clustering_columns
    X_enhanced = df_enhanced[all_features_enhanced].copy()

    if X_enhanced is None or len(clustering_columns) == 0:
        print("Could not prepare enhanced data with clustering features")
        return None

    # Split both datasets
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_original, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_enh_train, X_enh_test, _, _ = train_test_split(
        X_enhanced, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Test with Random Forest (typically shows good feature importance)
    print("\n--- Training Random Forest models ---")

    # Original features model
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(X_orig_train, y_train)
    orig_pred = rf_original.predict(X_orig_test)
    orig_accuracy = accuracy_score(y_test, orig_pred)
    orig_cv_scores = cross_val_score(rf_original, X_orig_train, y_train, cv=5)

    # Enhanced features model
    rf_enhanced = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_enhanced.fit(X_enh_train, y_train)
    enh_pred = rf_enhanced.predict(X_enh_test)
    enh_accuracy = accuracy_score(y_test, enh_pred)
    enh_cv_scores = cross_val_score(rf_enhanced, X_enh_train, y_train, cv=5)

    # Results comparison
    print(f"\n--- PERFORMANCE COMPARISON ---")
    print(f"Original Features:")
    print(f"  Features used: {feature_columns_original}")
    print(f"  Test Accuracy: {orig_accuracy:.4f}")
    print(
        f"  CV Score: {orig_cv_scores.mean():.4f} (+/- {orig_cv_scores.std() * 2:.4f})"
    )

    print(f"\nEnhanced Features (with clustering):")
    print(f"  Features used: {all_features_enhanced}")
    print(f"  Test Accuracy: {enh_accuracy:.4f}")
    print(f"  CV Score: {enh_cv_scores.mean():.4f} (+/- {enh_cv_scores.std() * 2:.4f})")

    improvement = enh_accuracy - orig_accuracy
    print(f"\nImprovement: {improvement:.4f} ({improvement/orig_accuracy*100:+.1f}%)")

    # Feature importance comparison
    print(f"\n--- FEATURE IMPORTANCE COMPARISON ---")

    orig_importance = dict(
        zip(feature_columns_original, rf_original.feature_importances_)
    )
    enh_importance = dict(zip(all_features_enhanced, rf_enhanced.feature_importances_))

    print("Original model feature importance:")
    for feature, importance in sorted(
        orig_importance.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {feature}: {importance:.4f}")

    print("\nEnhanced model feature importance:")
    for feature, importance in sorted(
        enh_importance.items(), key=lambda x: x[1], reverse=True
    ):
        cluster_marker = " 🔴" if feature in clustering_columns else ""
        print(f"  {feature}: {importance:.4f}{cluster_marker}")

    # Visualize comparison
    visualize_feature_comparison(orig_importance, enh_importance, clustering_columns)

    return {
        "original": {
            "accuracy": orig_accuracy,
            "cv_score": orig_cv_scores.mean(),
            "feature_importance": orig_importance,
            "features": feature_columns_original,
        },
        "enhanced": {
            "accuracy": enh_accuracy,
            "cv_score": enh_cv_scores.mean(),
            "feature_importance": enh_importance,
            "features": all_features_enhanced,
        },
        "improvement": improvement,
        "clustering_features": clustering_columns,
    }


def visualize_feature_comparison(orig_importance, enh_importance, clustering_features):
    """
    Visualize feature importance comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original features
    orig_features = list(orig_importance.keys())
    orig_values = list(orig_importance.values())

    bars1 = ax1.bar(orig_features, orig_values, alpha=0.7, color="skyblue")
    ax1.set_title("Original Features Importance", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Importance Score", fontsize=11)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, value in zip(bars1, orig_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Enhanced features
    enh_features = list(enh_importance.keys())
    enh_values = list(enh_importance.values())

    # Color clustering features differently
    colors = [
        "lightcoral" if feat in clustering_features else "skyblue"
        for feat in enh_features
    ]

    bars2 = ax2.bar(enh_features, enh_values, alpha=0.7, color=colors)
    ax2.set_title(
        "Enhanced Features Importance\n(Clustering features in red)",
        fontweight="bold",
        fontsize=12,
    )
    ax2.set_ylabel("Importance Score", fontsize=11)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, value in zip(bars2, enh_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        "classification/clustering_feature_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print(
        "Feature comparison visualization saved to 'clustering_feature_comparison.png'"
    )


def perform_enhanced_knn_classification(
    X_train, X_test, y_train, y_test, label_encoder, feature_names
):
    """
    Enhanced K-Nearest Neighbors classification with clustering features
    """
    print(f"\n=== ENHANCED K-NEAREST NEIGHBORS CLASSIFICATION ===")
    print(f"Using features: {feature_names}")

    # Find optimal k using cross-validation
    k_range = range(1, 21)
    cv_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        cv_scores.append(scores.mean())

    # Find best k
    best_k = k_range[np.argmax(cv_scores)]
    best_cv_score = max(cv_scores)

    print(f"Optimal k: {best_k}")
    print(f"Best cross-validation score: {best_cv_score:.4f}")

    # Train final model with best k
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train, y_train)

    # Make predictions
    y_pred = knn_best.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")

    # Detailed classification report
    class_names = label_encoder.classes_
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Cross-validation scores
    cv_scores_final = cross_val_score(knn_best, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores_final}")
    print(
        f"Average CV score: {cv_scores_final.mean():.4f} (+/- {cv_scores_final.std() * 2:.4f})"
    )

    return {
        "algorithm": "Enhanced K-Nearest Neighbors",
        "model": knn_best,
        "best_params": {"n_neighbors": best_k},
        "accuracy": accuracy,
        "cv_score": best_cv_score,
        "predictions": y_pred,
        "cv_scores": cv_scores_final,
    }


def perform_enhanced_random_forest_classification(
    X_train, X_test, y_train, y_test, label_encoder, feature_names
):
    """
    Enhanced Random Forest classification with clustering features
    """
    print(f"\n=== ENHANCED RANDOM FOREST CLASSIFICATION ===")
    print(f"Using features: {feature_names}")

    # Parameter grid for optimization
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
    }

    # Grid search with cross-validation
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    rf_best = grid_search.best_estimator_

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Make predictions
    y_pred = rf_best.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")

    # Detailed classification report
    class_names = label_encoder.classes_
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Feature importance
    feature_importance = rf_best.feature_importances_

    print(f"\nFeature Importance:")
    feature_importance_dict = {}
    for feature, importance in zip(feature_names, feature_importance):
        print(f"  {feature}: {importance:.4f}")
        feature_importance_dict[feature] = importance

    # Calculate cross-validation scores
    cv_scores = cross_val_score(rf_best, X_train, y_train, cv=5)
    cv_score_mean = cv_scores.mean()

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_score_mean:.4f} (+/- {cv_scores.std() * 2:.4f})")

    return {
        "algorithm": "Enhanced Random Forest",
        "model": rf_best,
        "best_params": grid_search.best_params_,
        "accuracy": accuracy,
        "cv_score": cv_score_mean,
        "predictions": y_pred,
        "cv_scores": cv_scores,
        "feature_importance": feature_importance_dict,
    }


def main_enhanced_classification():
    """
    Main enhanced classification analysis pipeline with clustering features
    """
    print("Starting enhanced classification analysis with clustering features...")

    # Load preprocessed data
    try:
        df = pd.read_csv("./data_processes/normalized_data.csv")
        print(f"Loaded normalized data: {df.shape}")
    except FileNotFoundError:
        print("Error: Please run data preprocessing first!")
        return

    # First, compare with and without clustering features
    comparison_results = compare_with_without_clustering(df)

    # Prepare enhanced classification data
    X, y, label_encoder, feature_names, clustering_features = (
        prepare_enhanced_classification_data(df)
    )
    if X is None:
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Enhanced feature dimensions: {X_train.shape[1]}")

    # Perform enhanced classification algorithms
    print("Running enhanced classification algorithms...")

    # Enhanced K-Nearest Neighbors
    knn_results = perform_enhanced_knn_classification(
        X_train, X_test, y_train, y_test, label_encoder, feature_names
    )

    # Enhanced Naive Bayes (reuse original function but with enhanced features)
    from classification import perform_naive_bayes_classification

    nb_results = perform_naive_bayes_classification(
        X_train, X_test, y_train, y_test, label_encoder
    )
    nb_results["algorithm"] = "Enhanced Naive Bayes"

    # Enhanced Random Forest
    rf_results = perform_enhanced_random_forest_classification(
        X_train, X_test, y_train, y_test, label_encoder, feature_names
    )

    # Combine results
    all_results = [knn_results, nb_results, rf_results]

    # Visualize results
    from classification import (
        visualize_classification_results,
        analyze_classification_errors,
        compare_classification_algorithms,
    )

    visualize_classification_results(all_results, X_test, y_test, label_encoder)

    # Analyze errors
    analyze_classification_errors(all_results, X_test, y_test, label_encoder)

    # Compare algorithms
    comparison_table = compare_classification_algorithms(all_results)

    # Enhanced feature importance analysis
    enhanced_feature_importance_analysis(
        all_results, feature_names, clustering_features
    )

    # Save results
    results_summary = pd.DataFrame(
        [
            {
                "Algorithm": r["algorithm"],
                "Accuracy": r["accuracy"],
                "CV_Score": r["cv_score"],
                "Best_Params": str(r["best_params"]),
            }
            for r in all_results
        ]
    )

    results_summary.to_csv("enhanced_classification_results_summary.csv", index=False)
    print(
        "Enhanced classification results saved to 'enhanced_classification_results_summary.csv'"
    )

    # Summary comparison
    print("\n=== SUMMARY: CLUSTERING FEATURES IMPACT ===")
    if comparison_results:
        orig_acc = comparison_results["original"]["accuracy"]
        enh_acc = comparison_results["enhanced"]["accuracy"]
        improvement = comparison_results["improvement"]

        print(f"Original Features Accuracy: {orig_acc:.4f}")
        print(f"Enhanced Features Accuracy: {enh_acc:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement/orig_acc*100:+.1f}%)")

        if improvement > 0:
            print("✓ Clustering features IMPROVED classification performance!")
        else:
            print("✗ Clustering features did not improve performance significantly.")
            print(
                "  This could indicate that clustering patterns don't strongly correlate with the target variable."
            )

    print("\n=== ENHANCED CLASSIFICATION ANALYSIS COMPLETE ===")
    return all_results, comparison_table, comparison_results


def enhanced_feature_importance_analysis(
    results_list, feature_names, clustering_features
):
    """
    Enhanced feature importance analysis highlighting clustering features
    """
    print("\n=== ENHANCED FEATURE IMPORTANCE ANALYSIS ===")

    # Collect feature importance
    importance_data = {}

    for result in results_list:
        algorithm_name = result["algorithm"]

        if "feature_importance" in result and result["feature_importance"]:
            importance_data[algorithm_name] = result["feature_importance"]
            print(f"{algorithm_name}: Feature importance available")

    if importance_data:
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for plotting
        all_features = feature_names
        n_features = len(all_features)
        n_algorithms = len(importance_data)

        # Set up the plot
        x = np.arange(n_features)
        width = 0.8 / n_algorithms

        # Plot bars for each algorithm
        for i, (algo_name, feature_dict) in enumerate(importance_data.items()):
            values = [feature_dict.get(feature, 0) for feature in all_features]
            offset = (i - n_algorithms / 2 + 0.5) * width

            # Color clustering features differently
            colors = [
                "lightcoral" if feat in clustering_features else "skyblue"
                for feat in all_features
            ]

            bars = ax.bar(
                x + offset, values, width, label=algo_name, alpha=0.8, color=colors
            )

            # Add value labels for significant values
            for bar, value in zip(bars, values):
                if value > 0.01:  # Only label significant values
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )

        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("Importance Score", fontsize=12)
        ax.set_title(
            "Enhanced Feature Importance Analysis\n(Clustering features in red)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(all_features, rotation=45, ha="right")
        ax.legend(title="Algorithm")
        ax.grid(True, alpha=0.3, axis="y")

        # Add text annotation for clustering features
        clustering_text = f"Clustering features: {', '.join(clustering_features)}"
        ax.text(
            0.02,
            0.98,
            clustering_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(
            "classification/enhanced_feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            "Enhanced feature importance analysis saved to 'enhanced_feature_importance.png'"
        )

        # Print summary
        print(f"\nFeature Importance Summary (including clustering features):")
        for algo_name, feature_dict in importance_data.items():
            print(f"\n{algo_name}:")
            sorted_features = sorted(
                feature_dict.items(), key=lambda x: x[1], reverse=True
            )
            for feature, importance in sorted_features:
                marker = " [Clustering]" if feature in clustering_features else ""
                print(f"  {feature}: {importance:.4f}{marker}")


if __name__ == "__main__":
    enhanced_results, comparison, clustering_impact = main_enhanced_classification()
