import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA
import joblib
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def prepare_clustering_data(df, numeric_columns):
    """
    Prepare data specifically for clustering analysis with improved preprocessing
    """
    print("=== CLUSTERING DATA PREPARATION ===")

    # Check which features are available in the dataframe
    available_features = [col for col in numeric_columns if col in df.columns]

    # Handle is_new column specially
    df_clustering = df.copy()
    if "is_new" in df.columns:
        # Convert boolean to numeric for clustering
        df_clustering["is_new_numeric"] = df_clustering["is_new"].astype(int)
        available_features.append("is_new_numeric")
        print("Added 'is_new' as numeric feature for clustering")

    print(f"Available clustering features: {available_features}")

    # Apply feature transformations to handle skewed distributions
    print("\nApplying feature transformations...")

    # 1. Handle likes_count (extremely right-skewed)
    if "likes_count" in available_features:
        # Log transformation to reduce skewness (add 1 to avoid log(0))
        df_clustering["likes_count_log"] = np.log1p(df_clustering["likes_count"])
        print(f"Applied log transformation to likes_count")

        # Replace original likes_count with transformed version
        available_features = [
            col if col != "likes_count" else "likes_count_log"
            for col in available_features
        ]

    # 2. Handle current_price (right-skewed)
    if "current_price" in available_features:
        # Square root transformation (milder than log for less skewed data)
        df_clustering["current_price_sqrt"] = np.sqrt(df_clustering["current_price"])
        print(f"Applied square root transformation to current_price")

        # Replace original current_price with transformed version
        available_features = [
            col if col != "current_price" else "current_price_sqrt"
            for col in available_features
        ]

    # 3. Handle discount (bimodal but relatively normal)
    # Keep discount as-is since it's already in a reasonable range and distribution

    print(f"Final clustering features after transformation: {available_features}")

    # Extract clustering data
    X_clustering = df_clustering[available_features].copy()

    # Check for any remaining issues
    print(f"\nClustering dataset shape: {X_clustering.shape}")
    print(f"Feature statistics after transformation:")
    print(X_clustering.describe())

    # Check for infinite or NaN values
    if X_clustering.isnull().sum().sum() > 0:
        print("Warning: Found NaN values after transformation")
        X_clustering = X_clustering.fillna(X_clustering.median())

    if np.isinf(X_clustering.values).sum() > 0:
        print("Warning: Found infinite values after transformation")
        X_clustering = X_clustering.replace([np.inf, -np.inf], np.nan).fillna(
            X_clustering.median()
        )

    # Return mapping for analysis purposes
    analysis_feature_names = []
    for feature in available_features:
        if feature == "is_new_numeric":
            analysis_feature_names.append("is_new")
        elif feature == "likes_count_log":
            analysis_feature_names.append("likes_count")
        elif feature == "current_price_sqrt":
            analysis_feature_names.append("current_price")
        else:
            analysis_feature_names.append(feature)

    return X_clustering, analysis_feature_names


def visualize_feature_transformations(df_original, df_transformed, feature_mapping):
    """
    Visualize the effect of feature transformations
    """
    print("\n=== FEATURE TRANSFORMATION VISUALIZATION ===")

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Original distributions (top row)
    if "likes_count" in df_original.columns:
        axes[0, 0].hist(
            df_original["likes_count"], bins=50, alpha=0.7, edgecolor="black"
        )
        axes[0, 0].set_title("Original likes_count Distribution")
        axes[0, 0].set_xlabel("likes_count")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

    if "current_price" in df_original.columns:
        axes[0, 1].hist(
            df_original["current_price"], bins=50, alpha=0.7, edgecolor="black"
        )
        axes[0, 1].set_title("Original current_price Distribution")
        axes[0, 1].set_xlabel("current_price")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

    if "discount" in df_original.columns:
        axes[0, 2].hist(df_original["discount"], bins=50, alpha=0.7, edgecolor="black")
        axes[0, 2].set_title("Original discount Distribution")
        axes[0, 2].set_xlabel("discount")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].grid(True, alpha=0.3)

    # Transformed distributions (bottom row)
    if "likes_count_log" in df_transformed.columns:
        axes[1, 0].hist(
            df_transformed["likes_count_log"],
            bins=50,
            alpha=0.7,
            edgecolor="black",
            color="orange",
        )
        axes[1, 0].set_title("Transformed likes_count (log1p)")
        axes[1, 0].set_xlabel("log1p(likes_count)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True, alpha=0.3)

    if "current_price_sqrt" in df_transformed.columns:
        axes[1, 1].hist(
            df_transformed["current_price_sqrt"],
            bins=50,
            alpha=0.7,
            edgecolor="black",
            color="orange",
        )
        axes[1, 1].set_title("Transformed current_price (sqrt)")
        axes[1, 1].set_xlabel("sqrt(current_price)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True, alpha=0.3)

    if "discount" in df_transformed.columns:
        axes[1, 2].hist(
            df_transformed["discount"],
            bins=50,
            alpha=0.7,
            edgecolor="black",
            color="orange",
        )
        axes[1, 2].set_title("discount (unchanged)")
        axes[1, 2].set_xlabel("discount")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "clustering/feature_transformations_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(
        "Feature transformation comparison saved to 'feature_transformations_comparison.png'"
    )


def determine_optimal_clusters(X, max_clusters=10):
    """
    Use elbow method and silhouette analysis with improved range for transformed data
    """
    print("\n=== OPTIMAL CLUSTER ANALYSIS ===")

    # Calculate metrics for different cluster numbers
    cluster_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []

    for n_clusters in cluster_range:
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(X)

        # Calculate metrics
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(sil_score)

        print(
            f"Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.2f}, Silhouette: {sil_score:.3f}"
        )

    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow method
    ax1.plot(cluster_range, inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Inertia (Within-cluster sum of squares)")
    ax1.set_title("Elbow Method for Optimal Clusters")
    ax1.grid(True, alpha=0.3)

    # Highlight potential elbow points
    if len(inertias) >= 3:
        # Calculate rate of change
        rates = []
        for i in range(1, len(inertias)):
            rate = inertias[i - 1] - inertias[i]
            rates.append(rate)

        # Find the elbow point (where rate of change decreases significantly)
        max_rate_idx = np.argmax(rates)
        elbow_point = cluster_range[max_rate_idx + 1]
        ax1.axvline(
            x=elbow_point,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Potential Elbow: {elbow_point}",
        )
        ax1.legend()

    # Silhouette scores
    ax2.plot(cluster_range, silhouette_scores, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.grid(True, alpha=0.3)

    # Highlight best silhouette score
    best_sil_idx = np.argmax(silhouette_scores)
    best_sil_clusters = cluster_range[best_sil_idx]
    ax2.axvline(
        x=best_sil_clusters,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best Silhouette: {best_sil_clusters}",
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        "clustering/optimal_clusters_analysis_improved.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Find optimal number of clusters (highest silhouette score)
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"\nRecommended number of clusters: {optimal_clusters}")
    print(f"Best silhouette score: {max(silhouette_scores):.3f}")

    return optimal_clusters


def perform_kmeans_clustering(X, n_clusters=4):
    """
    Perform K-Means clustering analysis
    """
    print(f"\n=== K-MEANS CLUSTERING (k={n_clusters}) ===")

    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X)

    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X, cluster_labels)
    calinski_score = calinski_harabasz_score(X, cluster_labels)
    inertia = kmeans.inertia_

    print(f"Algorithm: K-Means")
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_score:.4f}")
    print(f"Inertia: {inertia:.4f}")

    # Cluster centers
    centers = kmeans.cluster_centers_
    print(f"\nCluster Centers:")
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")

    # Cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster Distribution:")
    for cluster, count in zip(unique, counts):
        print(
            f"Cluster {cluster}: {count} points ({count/len(cluster_labels)*100:.1f}%)"
        )

    return {
        "algorithm": "K-Means",
        "labels": cluster_labels,
        "silhouette_score": silhouette_avg,
        "calinski_score": calinski_score,
        "inertia": inertia,
        "centers": centers,
        "model": kmeans,
    }


def perform_dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering analysis
    """
    print(f"\n=== DBSCAN CLUSTERING (eps={eps}, min_samples={min_samples}) ===")

    # Initialize and fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)

    # Calculate metrics (excluding noise points for silhouette score)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"Algorithm: DBSCAN")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")

    # Calculate silhouette score only if we have clusters and non-noise points
    if n_clusters > 1 and len(set(cluster_labels)) > 1:
        # Filter out noise points for silhouette calculation
        mask = cluster_labels != -1
        if mask.sum() > 1:
            silhouette_avg = silhouette_score(X[mask], cluster_labels[mask])
            calinski_score = calinski_harabasz_score(X[mask], cluster_labels[mask])
        else:
            silhouette_avg = -1
            calinski_score = -1
    else:
        silhouette_avg = -1
        calinski_score = -1

    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_score:.4f}")

    # Cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster Distribution:")
    for cluster, count in zip(unique, counts):
        cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        print(f"{cluster_name}: {count} points ({count/len(cluster_labels)*100:.1f}%)")

    return {
        "algorithm": "DBSCAN",
        "labels": cluster_labels,
        "silhouette_score": silhouette_avg,
        "calinski_score": calinski_score,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "model": dbscan,
    }


def optimize_dbscan_parameters(X):
    """
    Find optimal parameters for DBSCAN
    """
    print("\n=== DBSCAN PARAMETER OPTIMIZATION ===")

    # Test different parameter combinations
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    min_samples_values = [3, 5, 7, 10]

    best_score = -1
    best_params = {}
    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters > 1:
                mask = labels != -1
                if mask.sum() > 1:
                    score = silhouette_score(X[mask], labels[mask])
                    results.append(
                        {
                            "eps": eps,
                            "min_samples": min_samples,
                            "n_clusters": n_clusters,
                            "n_noise": n_noise,
                            "silhouette_score": score,
                        }
                    )

                    if score > best_score:
                        best_score = score
                        best_params = {"eps": eps, "min_samples": min_samples}

    print(
        f"Best DBSCAN parameters: eps={best_params.get('eps', 0.5)}, min_samples={best_params.get('min_samples', 5)}"
    )
    print(f"Best silhouette score: {best_score:.4f}")

    return best_params


def visualize_clusters(X, results_list, feature_names):
    """
    Visualize clustering results
    """
    print("\n=== CLUSTER VISUALIZATION ===")

    # Use PCA for dimensionality reduction if more than 2 features
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)"
    else:
        X_pca = X
        x_label = feature_names[0]
        y_label = feature_names[1]

    # Create subplots for each algorithm
    n_algorithms = len(results_list)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(6 * n_algorithms, 5))

    if n_algorithms == 1:
        axes = [axes]

    for i, result in enumerate(results_list):
        ax = axes[i]
        labels = result["labels"]

        # Plot clusters
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.7
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(
            f"{result['algorithm']} Clustering\n"
            f"Silhouette Score: {result['silhouette_score']:.3f}"
        )

        # Add colorbar
        plt.colorbar(scatter, ax=ax)

        # Plot cluster centers for K-Means
        if "centers" in result:
            centers_pca = (
                pca.transform(result["centers"])
                if X.shape[1] > 2
                else result["centers"]
            )
            ax.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                c="red",
                marker="x",
                s=200,
                linewidths=3,
                label="Centroids",
            )
            ax.legend()

    plt.tight_layout()
    plt.savefig("clustering/clustering_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def analyze_cluster_characteristics(df, cluster_results, feature_names):
    """
    Analyze characteristics of each cluster
    """
    print("\n=== CLUSTER CHARACTERISTICS ANALYSIS ===")

    for result in cluster_results:
        print(f"\n--- {result['algorithm']} Cluster Analysis ---")

        # Add cluster labels to dataframe
        df_temp = df.copy()
        df_temp["cluster"] = result["labels"]

        # Create compatible feature names for analysis
        # Map feature names to actual columns in df
        analysis_features = []
        for feature in feature_names:
            if feature == "is_new_numeric":
                # Use original is_new column if it exists, otherwise skip
                if "is_new" in df.columns:
                    analysis_features.append("is_new")
                elif "new_score" in df.columns:
                    analysis_features.append("new_score")
                # If neither exists, skip this feature
            else:
                # Use the feature as-is if it exists in df
                if feature in df.columns:
                    analysis_features.append(feature)

        print(f"Analyzing features: {analysis_features}")

        # Calculate cluster statistics only for available features
        if analysis_features:
            cluster_stats = df_temp.groupby("cluster")[analysis_features].agg(
                ["mean", "std", "count"]
            )

            print("Cluster Statistics:")
            print(cluster_stats.round(3))
        else:
            print("Warning: No compatible features found for cluster analysis")

        # Category distribution per cluster
        if "category" in df.columns:
            print("\nCategory Distribution per Cluster:")
            category_dist = pd.crosstab(df_temp["cluster"], df_temp["category"])
            print(category_dist)

            # Calculate percentage within each cluster
            category_pct = (
                pd.crosstab(df_temp["cluster"], df_temp["category"], normalize="index")
                * 100
            )
            print("\nCategory Percentage within Clusters:")
            print(category_pct.round(1))

        # Additional cluster insights
        print(f"\nCluster Size Distribution:")
        cluster_sizes = df_temp["cluster"].value_counts().sort_index()
        for cluster_id, size in cluster_sizes.items():
            cluster_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            percentage = (size / len(df_temp)) * 100
            print(f"  {cluster_name}: {size} products ({percentage:.1f}%)")


def compare_clustering_algorithms(results_list):
    """
    Compare different clustering algorithms
    """
    print("\n=== CLUSTERING ALGORITHM COMPARISON ===")

    # Create comparison table
    comparison_data = []
    for result in results_list:
        comparison_data.append(
            {
                "Algorithm": result["algorithm"],
                "Silhouette Score": result["silhouette_score"],
                "Calinski-Harabasz Score": result.get("calinski_score", "N/A"),
                "Number of Clusters": result.get(
                    "n_clusters",
                    len(set(result["labels"])) - (1 if -1 in result["labels"] else 0),
                ),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Determine best algorithm
    valid_results = [r for r in results_list if r["silhouette_score"] > 0]
    if valid_results:
        best_algorithm = max(valid_results, key=lambda x: x["silhouette_score"])
        print(f"\nBest performing algorithm: {best_algorithm['algorithm']}")
        print(f"Best silhouette score: {best_algorithm['silhouette_score']:.4f}")

    return comparison_df


def main_clustering():
    """
    Main clustering analysis pipeline with improved preprocessing
    """
    print("Starting improved clustering analysis...")

    # Load preprocessed data
    try:
        df = pd.read_csv("normalized_data.csv")
        df_original = pd.read_csv(
            "preprocessed_data.csv"
        )  # Also load original for comparison
        print(f"Loaded normalized data: {df.shape}")
        print(f"Loaded original data: {df_original.shape}")
    except FileNotFoundError:
        print("Error: Please run data preprocessing first!")
        return

    # Prepare clustering data with improved transformations
    numeric_columns = ["current_price", "discount", "likes_count"]
    X_clustering, feature_names = prepare_clustering_data(df, numeric_columns)

    # The prepare_clustering_data function creates a df_clustering with transformed features
    # We need to extract this for visualization
    df_transformed_temp = df.copy()
    if "likes_count" in df.columns:
        df_transformed_temp["likes_count_log"] = np.log1p(df["likes_count"])
    if "current_price" in df.columns:
        df_transformed_temp["current_price_sqrt"] = np.sqrt(df["current_price"])

    # Visualize feature transformations
    visualize_feature_transformations(df_original, df_transformed_temp, X_clustering)

    # Apply final standardization to transformed features
    print("\n=== FINAL STANDARDIZATION ===")

    scaler = StandardScaler()
    X_clustering_scaled = scaler.fit_transform(X_clustering)
    X_clustering_scaled = pd.DataFrame(
        X_clustering_scaled, columns=X_clustering.columns
    )

    print("Applied StandardScaler to transformed features")
    print("Final feature statistics (should have mean≈0, std≈1):")
    print(X_clustering_scaled.describe())

    # Determine optimal number of clusters with improved analysis
    optimal_k = determine_optimal_clusters(X_clustering_scaled)

    # Perform K-Means clustering
    kmeans_results = perform_kmeans_clustering(
        X_clustering_scaled, n_clusters=optimal_k
    )

    # Optimize DBSCAN parameters for transformed data
    best_dbscan_params = optimize_dbscan_parameters(X_clustering_scaled)

    # Perform DBSCAN clustering with optimized parameters
    dbscan_results = perform_dbscan_clustering(
        X_clustering_scaled,
        eps=best_dbscan_params.get("eps", 0.5),
        min_samples=best_dbscan_params.get("min_samples", 5),
    )

    # Combine results
    all_results = [kmeans_results, dbscan_results]

    # Visualize clustering results
    visualize_clusters(X_clustering_scaled, all_results, feature_names)

    # Analyze cluster characteristics (use original data for interpretation)
    analyze_cluster_characteristics(df_original, all_results, feature_names)

    # Compare algorithms
    comparison_table = compare_clustering_algorithms(all_results)

    # Save results with scaler information
    for result in all_results:
        df_with_clusters = df_original.copy()
        df_with_clusters[f'{result["algorithm"]}_cluster'] = result["labels"]
        filename = f'clustered_data_{result["algorithm"].lower().replace("-", "_")}.csv'
        df_with_clusters.to_csv(filename, index=False)
        print(f"Saved {result['algorithm']} results to {filename}")

    # Save the scaler for future use

    joblib.dump(scaler, "clustering_scaler.pkl")
    print("Saved StandardScaler to 'clustering_scaler.pkl' for future predictions")

    # Additional analysis: cluster quality assessment
    print("\n=== CLUSTER QUALITY ASSESSMENT ===")
    for result in all_results:
        if result["silhouette_score"] > 0:
            if result["silhouette_score"] > 0.5:
                quality = "Excellent"
            elif result["silhouette_score"] > 0.3:
                quality = "Good"
            elif result["silhouette_score"] > 0.1:
                quality = "Fair"
            else:
                quality = "Poor"

            print(
                f"{result['algorithm']}: {quality} clustering quality (Silhouette: {result['silhouette_score']:.3f})"
            )

    print("\n=== IMPROVED CLUSTERING ANALYSIS COMPLETE ===")
    print("Key improvements applied:")
    print("  • Log transformation for extremely skewed likes_count")
    print("  • Square root transformation for moderately skewed current_price")
    print("  • StandardScaler applied to all transformed features")
    print("  • Enhanced cluster optimization")
    print("  • Comprehensive quality assessment")

    return all_results, comparison_table, scaler


if __name__ == "__main__":
    clustering_results, comparison, scaler = main_clustering()
