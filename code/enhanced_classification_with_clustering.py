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
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings("ignore")

# Set matplotlib to use English and handle display issues
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def load_data():
    """Load all datasets for comparison"""
    print("=== LOADING DATASETS ===")

    # Load base data from data_processes
    base_data = pd.read_csv("code/data_proccesses/normalized_data.csv")
    print(f"Base data loaded: {base_data.shape}")

    # Load clustering results
    clustering_files = {
        "kmeans": "code/clustering/final_data_with_kmeans_k4_clusters.csv",
        "hierarchical": "code/clustering/final_data_with_hierarchical_k4_clusters.csv",
        "dbscan": "code/clustering/data_with_clusters.csv",
    }

    clustering_data = {}
    for method, file_path in clustering_files.items():
        try:
            data = pd.read_csv(file_path)
            clustering_data[method] = data
            print(f"Loaded {method} clustering data: {data.shape}")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, skipping {method}")

    return base_data, clustering_data


def prepare_dataset_variants(base_data, clustering_data):
    """Prepare different dataset variants for comparison"""
    print("\n=== PREPARING DATASET VARIANTS ===")

    datasets = {}

    # 1. Original features only (from data_processes)
    original_features = ["current_price", "discount", "likes_count"]
    if "is_new" in base_data.columns:
        base_data["is_new_numeric"] = base_data["is_new"].astype(int)
        original_features.append("is_new_numeric")

    datasets["original"] = {
        "data": base_data[original_features + ["category"]].copy(),
        "features": original_features,
        "description": "Original features only",
    }

    # 2-4. Add clustering features from each method
    for method, cluster_data in clustering_data.items():
        if method == "kmeans":
            cluster_col = "kmeans_cluster"
        elif method == "hierarchical":
            cluster_col = "hierarchical_cluster"
        else:  # dbscan
            cluster_col = "cluster"

        # Merge with base data
        enhanced_data = base_data.copy()
        if len(enhanced_data) == len(cluster_data):
            enhanced_data[cluster_col] = cluster_data[cluster_col].values
        else:
            enhanced_data = enhanced_data.merge(
                cluster_data[[cluster_col]],
                left_index=True,
                right_index=True,
                how="left",
            )

        enhanced_features = original_features + [cluster_col]
        datasets[f"with_{method}"] = {
            "data": enhanced_data[enhanced_features + ["category"]].copy(),
            "features": enhanced_features,
            "description": f"Original features + {method.upper()} clustering",
        }

    print(f"Prepared {len(datasets)} dataset variants:")
    for name, info in datasets.items():
        print(f"  {name}: {info['description']} ({len(info['features'])} features)")

    return datasets


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, model):
    """Train and evaluate a single model"""
    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "predictions": y_pred,
    }


def compare_all_variants(datasets):
    """Compare classification performance across all dataset variants"""
    print("\n=== COMPARING ALL DATASET VARIANTS ===")

    results = []
    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, criterion="gini", max_depth=None, min_samples_split=2
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", random_state=42),
    }

    for dataset_name, dataset_info in datasets.items():
        print(f"\n--- Processing {dataset_name}: {dataset_info['description']} ---")

        # Prepare data
        data = dataset_info["data"]
        features = dataset_info["features"]

        X = data[features]
        y = data["category"]

        # Encode target labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Features: {features}")

        # Train and evaluate each model
        for model_name, model in models.items():
            try:
                result = train_and_evaluate_model(
                    X_train, X_test, y_train, y_test, model_name, model
                )
                result["dataset"] = dataset_name
                result["dataset_description"] = dataset_info["description"]
                result["n_features"] = len(features)
                results.append(result)

                print(
                    f"  {model_name}: Accuracy = {result['accuracy']:.4f}, "
                    f"CV = {result['cv_mean']:.4f} (±{result['cv_std']:.4f})"
                )

            except Exception as e:
                print(f"  Error with {model_name}: {e}")

    return results


def analyze_feature_importance(datasets):
    """Analyze feature importance for Random Forest across different datasets"""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

    importance_results = {}

    for dataset_name, dataset_info in datasets.items():
        data = dataset_info["data"]
        features = dataset_info["features"]

        X = data[features]
        y = data["category"]

        # Encode target labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y_encoded)

        # Get feature importance
        importance_dict = dict(zip(features, rf.feature_importances_))
        importance_results[dataset_name] = importance_dict

        print(f"\n{dataset_info['description']}:")
        for feature, importance in sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        ):
            cluster_marker = " [CLUSTER]" if "cluster" in feature.lower() else ""
            print(f"  {feature}: {importance:.4f}{cluster_marker}")

    return importance_results


def visualize_comparison_results(results):
    """Create comprehensive visualization of results"""
    print("\n=== CREATING VISUALIZATIONS ===")

    # Convert results to DataFrame for easier plotting
    df_results = pd.DataFrame(results)

    # 1. Accuracy comparison by dataset
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Accuracy by dataset
    pivot_acc = df_results.pivot(
        index="dataset", columns="model_name", values="accuracy"
    )
    pivot_acc.plot(kind="bar", ax=ax1, width=0.8)
    ax1.set_title("Test Accuracy by Dataset Variant", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cross-validation scores
    pivot_cv = df_results.pivot(index="dataset", columns="model_name", values="cv_mean")
    pivot_cv.plot(kind="bar", ax=ax2, width=0.8)
    ax2.set_title(
        "Cross-Validation Scores by Dataset Variant", fontweight="bold", fontsize=12
    )
    ax2.set_ylabel("CV Score")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Best accuracy per dataset
    best_acc_per_dataset = df_results.groupby("dataset")["accuracy"].max()
    bars3 = ax3.bar(
        range(len(best_acc_per_dataset)),
        best_acc_per_dataset.values,
        color=["lightblue", "lightcoral", "lightgreen", "lightyellow"][
            : len(best_acc_per_dataset)
        ],
    )
    ax3.set_title("Best Accuracy per Dataset Variant", fontweight="bold", fontsize=12)
    ax3.set_ylabel("Best Accuracy")
    ax3.set_xticks(range(len(best_acc_per_dataset)))
    ax3.set_xticklabels(best_acc_per_dataset.index, rotation=45)
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars3, best_acc_per_dataset.values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 4: Model performance comparison
    model_avg_acc = df_results.groupby("model_name")["accuracy"].mean()
    bars4 = ax4.bar(
        range(len(model_avg_acc)),
        model_avg_acc.values,
        color=["skyblue", "lightcoral", "lightgreen", "orange"][: len(model_avg_acc)],
    )
    ax4.set_title(
        "Average Model Performance Across All Datasets", fontweight="bold", fontsize=12
    )
    ax4.set_ylabel("Average Accuracy")
    ax4.set_xticks(range(len(model_avg_acc)))
    ax4.set_xticklabels(model_avg_acc.index, rotation=45)
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars4, model_avg_acc.values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        "code/classification/clustering_comparison_results.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def generate_summary_report(results, importance_results):
    """Generate comprehensive summary report"""
    print("\n=== GENERATING SUMMARY REPORT ===")

    df_results = pd.DataFrame(results)

    # Find best performance overall
    best_result = df_results.loc[df_results["accuracy"].idxmax()]

    print(f"\n🏆 BEST OVERALL PERFORMANCE:")
    print(f"   Dataset: {best_result['dataset_description']}")
    print(f"   Model: {best_result['model_name']}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   CV Score: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})")

    # Compare original vs enhanced
    print(f"\n📊 DATASET COMPARISON:")
    dataset_performance = (
        df_results.groupby("dataset")
        .agg({"accuracy": ["mean", "max"], "cv_mean": ["mean", "max"]})
        .round(4)
    )

    original_max = df_results[df_results["dataset"] == "original"]["accuracy"].max()

    print(f"Original features best accuracy: {original_max:.4f}")

    for clustering_method in ["kmeans", "hierarchical", "dbscan"]:
        dataset_name = f"with_{clustering_method}"
        if dataset_name in df_results["dataset"].values:
            enhanced_max = df_results[df_results["dataset"] == dataset_name][
                "accuracy"
            ].max()
            improvement = enhanced_max - original_max
            improvement_pct = (improvement / original_max) * 100

            print(
                f"{clustering_method.upper()} clustering best accuracy: {enhanced_max:.4f} "
                f"(improvement: {improvement:+.4f} / {improvement_pct:+.1f}%)"
            )

    # Feature importance insights
    print(f"\n🔍 CLUSTERING FEATURE IMPACT:")
    for dataset_name, importance_dict in importance_results.items():
        if "cluster" in str(importance_dict.keys()).lower():
            cluster_features = [
                f for f in importance_dict.keys() if "cluster" in f.lower()
            ]
            if cluster_features:
                cluster_importance = importance_dict[cluster_features[0]]
                print(
                    f"   {dataset_name}: Cluster feature importance = {cluster_importance:.4f}"
                )

    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "code/classification/detailed_comparison_results.csv", index=False
    )

    # Create summary table
    summary_table = (
        df_results.groupby(["dataset", "model_name"])
        .agg(
            {
                "accuracy": "first",
                "cv_mean": "first",
                "cv_std": "first",
                "n_features": "first",
            }
        )
        .round(4)
    )

    summary_table.to_csv("code/classification/summary_comparison_results.csv")

    print(f"\n💾 Results saved to:")
    print(f"   - code/classification/detailed_comparison_results.csv")
    print(f"   - code/classification/summary_comparison_results.csv")
    print(f"   - code/classification/clustering_comparison_results.png")

    return summary_table


def main():
    """Main execution function"""
    print("🚀 ENHANCED CLASSIFICATION WITH CLUSTERING FEATURES")
    print("=" * 60)

    # Create classification directory if it doesn't exist
    import os

    os.makedirs("code/classification", exist_ok=True)

    # Load all data
    base_data, clustering_data = load_data()

    if not clustering_data:
        print("❌ No clustering data found. Please run clustering analysis first.")
        return

    # Prepare dataset variants
    datasets = prepare_dataset_variants(base_data, clustering_data)

    # Compare all variants
    results = compare_all_variants(datasets)

    # Analyze feature importance
    importance_results = analyze_feature_importance(datasets)

    # Create visualizations
    visualize_comparison_results(results)

    # Generate summary report
    summary_table = generate_summary_report(results, importance_results)

    print(f"\n✅ ANALYSIS COMPLETE!")
    print(
        f"Check the 'code/classification/' directory for all results and visualizations."
    )

    return results, summary_table, importance_results


if __name__ == "__main__":
    results, summary, importance = main()
