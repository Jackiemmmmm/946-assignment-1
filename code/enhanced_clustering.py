import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_mutual_info_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
import os

warnings.filterwarnings("ignore")


def prepare_enhanced_clustering_data(df, feature_strategy='engineered'):
    """
    Enhanced data preparation with multiple feature strategies
    """
    print(f"=== ENHANCED CLUSTERING DATA PREPARATION ({feature_strategy}) ===")
    
    df_clustering = df.copy()
    
    if feature_strategy == 'basic':
        # Basic numeric features
        feature_columns = ['current_price', 'discount', 'likes_count']
        if 'is_new' in df.columns:
            df_clustering['is_new_numeric'] = df_clustering['is_new'].astype(int)
            feature_columns.append('is_new_numeric')
            
    elif feature_strategy == 'engineered':
        # Use all engineered features
        feature_columns = []
        
        # Core numeric features
        core_features = ['current_price', 'discount', 'likes_count', 'likes_log']
        feature_columns.extend([col for col in core_features if col in df.columns])
        
        # Engineered features
        engineered_features = [
            'price_efficiency', 'discount_amount', 'value_score', 'price_per_like',
            'category_size', 'category_price_ratio', 'brand_frequency', 'brand_price_premium',
            'price_discount_interaction', 'discount_likes_interaction'
        ]
        feature_columns.extend([col for col in engineered_features if col in df.columns])
        
        # Add boolean features as numeric
        if 'is_new' in df.columns:
            df_clustering['is_new_numeric'] = df_clustering['is_new'].astype(int)
            feature_columns.append('is_new_numeric')
            
    elif feature_strategy == 'selected':
        # Use feature selection to choose best features
        from enhanced_preprocessing import intelligent_feature_selection
        from sklearn.preprocessing import LabelEncoder
        
        if 'category' in df.columns:
            # Get all numeric features
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [col for col in numeric_features if col not in ['id']]
            
            # Prepare target variable
            le = LabelEncoder()
            y = le.fit_transform(df['category'])
            
            # Feature selection
            X_numeric = df[numeric_features].fillna(0)
            X_selected, selected_features, selector = intelligent_feature_selection(
                X_numeric, y, method='mutual_info', k=min(10, len(numeric_features))
            )
            
            feature_columns = list(selected_features)
        else:
            # Fallback to basic features
            feature_columns = ['current_price', 'discount', 'likes_count']
    
    # Check available features
    available_features = [col for col in feature_columns if col in df_clustering.columns]
    
    if not available_features:
        raise ValueError(f"No features available for clustering from: {feature_columns}")
    
    print(f"Using {len(available_features)} features for clustering:")
    for feature in available_features:
        print(f"  • {feature}")
    
    # Extract clustering data
    X_clustering = df_clustering[available_features].copy()
    
    # Handle missing values
    X_clustering = X_clustering.fillna(X_clustering.median())
    
    # Remove infinite values
    X_clustering = X_clustering.replace([np.inf, -np.inf], np.nan).fillna(X_clustering.median())
    
    print(f"Clustering dataset shape: {X_clustering.shape}")
    print(f"Feature statistics:")
    print(X_clustering.describe())
    
    return X_clustering, available_features, df_clustering


def determine_optimal_clusters_advanced(X, max_clusters=15, algorithms=['kmeans', 'gaussian_mixture']):
    """
    Advanced optimal cluster determination using multiple algorithms and metrics
    """
    print("\n=== ADVANCED OPTIMAL CLUSTER ANALYSIS ===")
    
    cluster_range = range(2, max_clusters + 1)
    results = {}
    
    for algorithm in algorithms:
        print(f"\nAnalyzing {algorithm}...")
        
        algorithm_results = {
            'n_clusters': [],
            'silhouette_scores': [],
            'calinski_scores': [],
            'davies_bouldin_scores': [],
            'inertias': [] if algorithm == 'kmeans' else [],
            'aic': [] if algorithm == 'gaussian_mixture' else [],
            'bic': [] if algorithm == 'gaussian_mixture' else []
        }
        
        for n_clusters in cluster_range:
            try:
                if algorithm == 'kmeans':
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = model.fit_predict(X)
                    algorithm_results['inertias'].append(model.inertia_)
                    
                elif algorithm == 'gaussian_mixture':
                    model = GaussianMixture(n_components=n_clusters, random_state=42)
                    model.fit(X)
                    labels = model.predict(X)
                    algorithm_results['aic'].append(model.aic(X))
                    algorithm_results['bic'].append(model.bic(X))
                
                # Calculate metrics
                if len(set(labels)) > 1:  # Ensure multiple clusters
                    sil_score = silhouette_score(X, labels)
                    cal_score = calinski_harabasz_score(X, labels)
                    db_score = davies_bouldin_score(X, labels)
                    
                    algorithm_results['n_clusters'].append(n_clusters)
                    algorithm_results['silhouette_scores'].append(sil_score)
                    algorithm_results['calinski_scores'].append(cal_score)
                    algorithm_results['davies_bouldin_scores'].append(db_score)
                    
                    print(f"  {n_clusters} clusters: Sil={sil_score:.3f}, Cal={cal_score:.1f}, DB={db_score:.3f}")
                
            except Exception as e:
                print(f"  Error with {n_clusters} clusters: {e}")
                continue
        
        results[algorithm] = algorithm_results
    
    # Create comprehensive visualization
    n_algorithms = len(results)
    fig, axes = plt.subplots(2, n_algorithms, figsize=(6 * n_algorithms, 10))
    
    if n_algorithms == 1:
        axes = axes.reshape(2, 1)
    
    for i, (algorithm, data) in enumerate(results.items()):
        if not data['n_clusters']:
            continue
            
        # Top row: Main optimization metrics
        ax1 = axes[0, i]
        
        # Plot silhouette scores
        ax1.plot(data['n_clusters'], data['silhouette_scores'], 'bo-', label='Silhouette', linewidth=2)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(data['n_clusters'], data['calinski_scores'], 'ro-', label='Calinski-Harabasz', linewidth=2)
        
        # Highlight optimal points
        best_sil_idx = np.argmax(data['silhouette_scores'])
        best_sil_k = data['n_clusters'][best_sil_idx]
        ax1.axvline(x=best_sil_k, color='blue', linestyle='--', alpha=0.7, label=f'Best Sil: {best_sil_k}')
        
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score', color='blue')
        ax1_twin.set_ylabel('Calinski-Harabasz Score', color='red')
        ax1.set_title(f'{algorithm.replace("_", " ").title()} - Optimization Metrics')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Bottom row: Algorithm-specific metrics
        ax2 = axes[1, i]
        
        if algorithm == 'kmeans' and data['inertias']:
            ax2.plot(data['n_clusters'], data['inertias'], 'go-', linewidth=2)
            ax2.set_ylabel('Inertia')
            ax2.set_title(f'{algorithm.replace("_", " ").title()} - Inertia')
            
            # Elbow detection
            if len(data['inertias']) >= 3:
                # Calculate rate of change
                rates = np.diff(data['inertias'])
                elbow_idx = np.argmax(np.abs(np.diff(rates))) + 1
                if elbow_idx < len(data['n_clusters']):
                    elbow_k = data['n_clusters'][elbow_idx]
                    ax2.axvline(x=elbow_k, color='red', linestyle='--', alpha=0.7, label=f'Elbow: {elbow_k}')
                    ax2.legend()
            
        elif algorithm == 'gaussian_mixture' and data['aic']:
            ax2.plot(data['n_clusters'], data['aic'], 'mo-', label='AIC', linewidth=2)
            ax2.plot(data['n_clusters'], data['bic'], 'co-', label='BIC', linewidth=2)
            ax2.set_ylabel('Information Criterion')
            ax2.set_title(f'{algorithm.replace("_", " ").title()} - Model Selection')
            ax2.legend()
            
            # Highlight minimum AIC/BIC
            if data['aic']:
                best_aic_idx = np.argmin(data['aic'])
                best_aic_k = data['n_clusters'][best_aic_idx]
                ax2.axvline(x=best_aic_k, color='magenta', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Number of Clusters')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_visuals/advanced_cluster_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Determine optimal clusters for each algorithm
    optimal_clusters = {}
    for algorithm, data in results.items():
        if data['silhouette_scores']:
            best_idx = np.argmax(data['silhouette_scores'])
            optimal_clusters[algorithm] = {
                'n_clusters': data['n_clusters'][best_idx],
                'silhouette_score': data['silhouette_scores'][best_idx]
            }
    
    print(f"\nOptimal clusters determined:")
    for algorithm, info in optimal_clusters.items():
        print(f"  {algorithm}: {info['n_clusters']} clusters (Silhouette: {info['silhouette_score']:.3f})")
    
    return optimal_clusters, results


def perform_advanced_clustering(X, algorithms=None):
    """
    Perform clustering with multiple advanced algorithms
    """
    print("\n=== ADVANCED CLUSTERING ALGORITHMS ===")
    
    if algorithms is None:
        algorithms = {
            'kmeans': {'n_clusters': 4},
            'gaussian_mixture': {'n_components': 4},
            'agglomerative': {'n_clusters': 4},
            'spectral': {'n_clusters': 4},
            'dbscan': {'eps': 0.5, 'min_samples': 5}
        }
    
    results = {}
    
    for algorithm, params in algorithms.items():
        print(f"\n--- {algorithm.upper()} CLUSTERING ---")
        
        try:
            if algorithm == 'kmeans':
                model = KMeans(random_state=42, n_init=10, **params)
                labels = model.fit_predict(X)
                extra_info = {'centers': model.cluster_centers_, 'inertia': model.inertia_}
                
            elif algorithm == 'gaussian_mixture':
                model = GaussianMixture(random_state=42, **params)
                model.fit(X)
                labels = model.predict(X)
                extra_info = {'aic': model.aic(X), 'bic': model.bic(X), 'means': model.means_}
                
            elif algorithm == 'agglomerative':
                model = AgglomerativeClustering(**params)
                labels = model.fit_predict(X)
                extra_info = {}
                
            elif algorithm == 'spectral':
                model = SpectralClustering(random_state=42, **params)
                labels = model.fit_predict(X)
                extra_info = {}
                
            elif algorithm == 'dbscan':
                model = DBSCAN(**params)
                labels = model.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                extra_info = {'n_clusters': n_clusters, 'n_noise': n_noise}
            
            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 1:
                # Filter out noise points for DBSCAN
                if algorithm == 'dbscan' and -1 in labels:
                    mask = labels != -1
                    if mask.sum() > 1:
                        sil_score = silhouette_score(X[mask], labels[mask])
                        cal_score = calinski_harabasz_score(X[mask], labels[mask])
                        db_score = davies_bouldin_score(X[mask], labels[mask])
                    else:
                        sil_score = cal_score = db_score = -1
                else:
                    sil_score = silhouette_score(X, labels)
                    cal_score = calinski_harabasz_score(X, labels)
                    db_score = davies_bouldin_score(X, labels)
            else:
                sil_score = cal_score = db_score = -1
            
            # Store results
            result = {
                'algorithm': algorithm,
                'model': model,
                'labels': labels,
                'params': params,
                'n_clusters': n_clusters,
                'silhouette_score': sil_score,
                'calinski_score': cal_score,
                'davies_bouldin_score': db_score,
                **extra_info
            }
            
            results[algorithm] = result
            
            # Print metrics
            print(f"Clusters: {n_clusters}")
            print(f"Silhouette Score: {sil_score:.4f}")
            print(f"Calinski-Harabasz Score: {cal_score:.4f}")
            print(f"Davies-Bouldin Score: {db_score:.4f}")
            
            if algorithm == 'dbscan':
                print(f"Noise points: {extra_info.get('n_noise', 0)}")
            elif algorithm == 'gaussian_mixture':
                print(f"AIC: {extra_info['aic']:.2f}, BIC: {extra_info['bic']:.2f}")
            
            # Cluster distribution
            unique, counts = np.unique(labels, return_counts=True)
            print("Cluster distribution:")
            for cluster, count in zip(unique, counts):
                cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
                print(f"  {cluster_name}: {count} points ({count/len(labels)*100:.1f}%)")
                
        except Exception as e:
            print(f"Error with {algorithm}: {e}")
            continue
    
    return results


def create_advanced_cluster_visualization(X, results, feature_names):
    """
    Create advanced cluster visualizations
    """
    print("\n=== ADVANCED CLUSTER VISUALIZATION ===")
    
    os.makedirs('enhanced_visuals', exist_ok=True)
    
    n_algorithms = len(results)
    
    # 1. Main clustering results visualization
    fig, axes = plt.subplots(2, n_algorithms, figsize=(6 * n_algorithms, 12))
    
    if n_algorithms == 1:
        axes = axes.reshape(2, 1)
    
    # Prepare dimensionality reduction
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Also create TSNE for better visualization (if dataset not too large)
    if len(X) <= 5000:  # TSNE is computationally expensive
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
        X_tsne = tsne.fit_transform(X)
    else:
        X_tsne = X_pca  # Fallback to PCA
    
    for i, (algorithm, result) in enumerate(results.items()):
        if 'labels' not in result:
            continue
        
        labels = result['labels']
        
        # Top row: PCA visualization
        ax1 = axes[0, i]
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)')
        ax1.set_title(f'{algorithm.replace("_", " ").title()}\nSilhouette: {result["silhouette_score"]:.3f}')
        plt.colorbar(scatter1, ax=ax1)
        
        # Add cluster centers for applicable algorithms
        if algorithm in ['kmeans'] and 'centers' in result:
            centers_pca = pca.transform(result['centers'])
            ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroids')
            ax1.legend()
        
        # Bottom row: TSNE visualization
        ax2 = axes[1, i]
        scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        ax2.set_xlabel('TSNE 1')
        ax2.set_ylabel('TSNE 2')
        ax2.set_title(f'{algorithm.replace("_", " ").title()} - TSNE View')
        plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('enhanced_visuals/advanced_clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Clustering metrics comparison
    algorithms_list = list(results.keys())
    metrics = ['silhouette_score', 'calinski_score', 'davies_bouldin_score']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        values = [results[alg].get(metric, 0) for alg in algorithms_list]
        valid_values = [(alg, val) for alg, val in zip(algorithms_list, values) if val != -1]
        
        if valid_values:
            algs, vals = zip(*valid_values)
            bars = axes[i].bar(range(len(algs)), vals, alpha=0.8)
            axes[i].set_xticks(range(len(algs)))
            axes[i].set_xticklabels([alg.replace('_', ' ').title() for alg in algs], rotation=45)
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, vals):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('enhanced_visuals/clustering_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Detailed cluster characteristics heatmap
    if len(results) > 0 and feature_names:
        create_cluster_characteristics_heatmap(X, results, feature_names)
    
    print("Advanced visualizations saved to 'enhanced_visuals/' directory")


def create_cluster_characteristics_heatmap(X, results, feature_names):
    """
    Create heatmap showing cluster characteristics
    """
    print("\nCreating cluster characteristics heatmap...")
    
    # Convert X to DataFrame for easier manipulation
    X_df = pd.DataFrame(X, columns=feature_names)
    
    for algorithm, result in results.items():
        if 'labels' not in result or result['n_clusters'] <= 1:
            continue
            
        labels = result['labels']
        
        # Calculate cluster means
        cluster_means = []
        unique_labels = sorted([l for l in set(labels) if l != -1])  # Exclude noise
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            if cluster_mask.sum() > 0:
                cluster_mean = X_df[cluster_mask].mean()
                cluster_means.append(cluster_mean)
        
        if cluster_means:
            cluster_df = pd.DataFrame(cluster_means, 
                                    index=[f'Cluster {i}' for i in unique_labels])
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(cluster_df.T, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, cbar_kws={'label': 'Feature Value'})
            plt.title(f'{algorithm.replace("_", " ").title()} - Cluster Characteristics')
            plt.xlabel('Clusters')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(f'enhanced_visuals/{algorithm}_cluster_characteristics.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()


def compare_clustering_algorithms_advanced(results):
    """
    Advanced comparison of clustering algorithms
    """
    print("\n=== ADVANCED CLUSTERING COMPARISON ===")
    
    # Create comprehensive comparison table
    comparison_data = []
    
    for algorithm, result in results.items():
        comparison_data.append({
            'Algorithm': algorithm.replace('_', ' ').title(),
            'N_Clusters': result.get('n_clusters', 0),
            'Silhouette_Score': result.get('silhouette_score', -1),
            'Calinski_Harabasz': result.get('calinski_score', -1),
            'Davies_Bouldin': result.get('davies_bouldin_score', -1),
            'Special_Metric': result.get('aic', result.get('inertia', result.get('n_noise', 'N/A')))
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Filter out invalid results
    valid_df = comparison_df[comparison_df['Silhouette_Score'] > -1].copy()
    
    if len(valid_df) > 0:
        print("Clustering Algorithm Comparison:")
        print(valid_df.to_string(index=False, float_format='%.4f'))
        
        # Rank algorithms
        valid_df['Silhouette_Rank'] = valid_df['Silhouette_Score'].rank(ascending=False)
        valid_df['Calinski_Rank'] = valid_df['Calinski_Harabasz'].rank(ascending=False)
        valid_df['DB_Rank'] = valid_df['Davies_Bouldin'].rank(ascending=True)  # Lower is better for DB
        
        # Calculate overall rank (average of individual ranks)
        valid_df['Overall_Rank'] = (valid_df['Silhouette_Rank'] + 
                                   valid_df['Calinski_Rank'] + 
                                   valid_df['DB_Rank']) / 3
        
        best_algorithm = valid_df.loc[valid_df['Overall_Rank'].idxmin()]
        
        print(f"\n🏆 Best Overall Algorithm: {best_algorithm['Algorithm']}")
        print(f"   Overall Rank: {best_algorithm['Overall_Rank']:.2f}")
        print(f"   Silhouette Score: {best_algorithm['Silhouette_Score']:.4f}")
        print(f"   Number of Clusters: {best_algorithm['N_Clusters']}")
        
        # Save comparison results
        valid_df.to_csv('enhanced_clustering_comparison.csv', index=False)
        print("\nComparison results saved to 'enhanced_clustering_comparison.csv'")
        
        return valid_df
    else:
        print("No valid clustering results to compare")
        return None


def main_enhanced_clustering():
    """
    Main enhanced clustering pipeline
    """
    print("Starting Enhanced Clustering Analysis...")
    
    # Load enhanced preprocessed data
    try:
        df = pd.read_csv("enhanced_normalized_data.csv")
        print(f"Loaded enhanced normalized data: {df.shape}")
    except FileNotFoundError:
        print("Enhanced data not found. Running enhanced preprocessing first...")
        # Import and run enhanced preprocessing
        try:
            from enhanced_preprocessing import main_enhanced_preprocessing
            _, df, _ = main_enhanced_preprocessing()
        except Exception as e:
            print(f"Error running enhanced preprocessing: {e}")
            print("Falling back to original normalized data...")
            try:
                df = pd.read_csv("normalized_data.csv")
            except FileNotFoundError:
                print("No data available. Please run preprocessing first!")
                return None
    
    # Try different feature strategies
    feature_strategies = ['engineered', 'basic']
    
    best_results = None
    best_strategy = None
    best_score = -1
    
    for strategy in feature_strategies:
        try:
            print(f"\n{'='*60}")
            print(f"TESTING FEATURE STRATEGY: {strategy.upper()}")
            print(f"{'='*60}")
            
            # Prepare clustering data
            X_clustering, feature_names, df_processed = prepare_enhanced_clustering_data(
                df, feature_strategy=strategy
            )
            
            # Determine optimal clusters
            optimal_clusters, optimization_results = determine_optimal_clusters_advanced(
                X_clustering, max_clusters=12, algorithms=['kmeans', 'gaussian_mixture']
            )
            
            # Prepare algorithms with optimal parameters
            algorithms = {}
            
            # Use optimal clusters from analysis
            for alg, info in optimal_clusters.items():
                n_clusters = info['n_clusters']
                
                if alg == 'kmeans':
                    algorithms['kmeans'] = {'n_clusters': n_clusters}
                    algorithms['agglomerative'] = {'n_clusters': n_clusters}
                    algorithms['spectral'] = {'n_clusters': n_clusters}
                elif alg == 'gaussian_mixture':
                    algorithms['gaussian_mixture'] = {'n_components': n_clusters}
            
            # Add DBSCAN with optimized parameters
            # Simple parameter optimization for DBSCAN
            best_eps = 0.5
            best_min_samples = 5
            best_dbscan_score = -1
            
            for eps in [0.3, 0.5, 0.7, 1.0]:
                for min_samples in [3, 5, 7]:
                    dbscan_test = DBSCAN(eps=eps, min_samples=min_samples)
                    test_labels = dbscan_test.fit_predict(X_clustering)
                    
                    n_clusters_test = len(set(test_labels)) - (1 if -1 in test_labels else 0)
                    if n_clusters_test > 1:
                        mask = test_labels != -1
                        if mask.sum() > 1:
                            score = silhouette_score(X_clustering[mask], test_labels[mask])
                            if score > best_dbscan_score:
                                best_dbscan_score = score
                                best_eps = eps
                                best_min_samples = min_samples
            
            algorithms['dbscan'] = {'eps': best_eps, 'min_samples': best_min_samples}
            
            # Perform clustering
            clustering_results = perform_advanced_clustering(X_clustering, algorithms)
            
            # Visualize results
            create_advanced_cluster_visualization(X_clustering, clustering_results, feature_names)
            
            # Compare algorithms
            comparison_df = compare_clustering_algorithms_advanced(clustering_results)
            
            # Track best strategy
            if comparison_df is not None and len(comparison_df) > 0:
                max_silhouette = comparison_df['Silhouette_Score'].max()
                if max_silhouette > best_score:
                    best_score = max_silhouette
                    best_strategy = strategy
                    best_results = {
                        'clustering_results': clustering_results,
                        'comparison_df': comparison_df,
                        'feature_names': feature_names,
                        'X_clustering': X_clustering
                    }
            
        except Exception as e:
            print(f"Error with strategy {strategy}: {e}")
            continue
    
    # Report best results
    if best_results:
        print(f"\n{'='*80}")
        print(f"🏆 BEST FEATURE STRATEGY: {best_strategy.upper()}")
        print(f"🏆 BEST SILHOUETTE SCORE: {best_score:.4f}")
        print(f"{'='*80}")
        
        # Save best results
        for algorithm, result in best_results['clustering_results'].items():
            if 'labels' in result:
                # Add cluster labels to original data
                df_with_clusters = df.copy()
                df_with_clusters[f'{algorithm}_cluster'] = result['labels']
                
                # Save results
                filename = f'enhanced_clustered_data_{algorithm}.csv'
                df_with_clusters.to_csv(filename, index=False)
                print(f"Saved {algorithm} results to {filename}")
        
        print("\n=== ENHANCED CLUSTERING ANALYSIS COMPLETE ===")
        print("Key improvements:")
        print("  • Multiple advanced algorithms (K-Means, Gaussian Mixture, Agglomerative, Spectral, DBSCAN)")
        print("  • Enhanced feature engineering and selection")
        print("  • Comprehensive parameter optimization")
        print("  • Advanced visualization and evaluation metrics")
        print("  • Intelligent algorithm comparison and ranking")
        
        return best_results
    else:
        print("No successful clustering results obtained")
        return None


if __name__ == "__main__":
    enhanced_results = main_enhanced_clustering()