import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")


def load_and_prepare_datasets():
    """Load and prepare all dataset variants"""
    print("🔄 Loading datasets...")
    
    # Load base data
    base_data = pd.read_csv("code/data_proccesses/normalized_data.csv")
    print(f"✓ Base data loaded: {base_data.shape}")
    
    datasets = {}
    
    # 1. Original features only
    original_features = ["current_price", "discount", "likes_count"]
    
    datasets["original"] = {
        "data": base_data[original_features + ["category"]].copy(),
        "name": "Original Features Only",
        "features": original_features
    }
    
    # 2. K-Means clustering features
    try:
        kmeans_data = pd.read_csv("code/clustering/final_data_with_kmeans_k4_clusters.csv")
        kmeans_features = original_features + ["kmeans_cluster"]
        
        enhanced_data = base_data.copy()
        enhanced_data["kmeans_cluster"] = kmeans_data["kmeans_cluster"].values
        
        datasets["kmeans"] = {
            "data": enhanced_data[kmeans_features + ["category"]].copy(),
            "name": "Original + K-Means Clustering",
            "features": kmeans_features
        }
        print("✓ K-Means clustering data loaded")
    except Exception as e:
        print(f"❌ K-Means data error: {e}")
    
    # 3. Hierarchical clustering features
    try:
        hier_data = pd.read_csv("code/clustering/final_data_with_hierarchical_k4_clusters.csv")
        hier_features = original_features + ["hierarchical_cluster"]
        
        enhanced_data = base_data.copy()
        enhanced_data["hierarchical_cluster"] = hier_data["hierarchical_cluster"].values
        
        datasets["hierarchical"] = {
            "data": enhanced_data[hier_features + ["category"]].copy(),
            "name": "Original + Hierarchical Clustering",
            "features": hier_features
        }
        print("✓ Hierarchical clustering data loaded")
    except Exception as e:
        print(f"❌ Hierarchical data error: {e}")
    
    # 4. DBSCAN clustering features
    try:
        dbscan_data = pd.read_csv("code/clustering/data_with_clusters.csv")
        dbscan_features = original_features + ["cluster"]
        
        enhanced_data = base_data.copy()
        enhanced_data["cluster"] = dbscan_data["cluster"].values
        
        datasets["dbscan"] = {
            "data": enhanced_data[dbscan_features + ["category"]].copy(),
            "name": "Original + DBSCAN Clustering",
            "features": dbscan_features
        }
        print("✓ DBSCAN clustering data loaded")
    except Exception as e:
        print(f"❌ DBSCAN data error: {e}")
    
    print(f"📊 Total datasets prepared: {len(datasets)}")
    return datasets


def evaluate_dataset(dataset_info, dataset_name):
    """Evaluate classification performance for one dataset"""
    data = dataset_info["data"]
    features = dataset_info["features"]
    
    # Prepare data
    X = data[features]
    y = data["category"]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Models to test
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    
    results = []
    
    for model_name, model in models.items():
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        result = {
            "dataset": dataset_name,
            "dataset_name": dataset_info["name"],
            "model": model_name,
            "accuracy": accuracy,
            "n_features": len(features),
            "features": features.copy()
        }
        
        # Get feature importance for Random Forest
        if model_name == "Random Forest":
            importance_dict = dict(zip(features, model.feature_importances_))
            result["feature_importance"] = importance_dict
        
        results.append(result)
        print(f"  {model_name}: {accuracy:.4f}")
    
    return results


def main_comparison():
    """Main comparison function"""
    print("🚀 CLUSTERING FEATURES CLASSIFICATION COMPARISON")
    print("=" * 55)
    
    # Create output directory
    os.makedirs('code/classification', exist_ok=True)
    
    # Load datasets
    datasets = load_and_prepare_datasets()
    
    if len(datasets) < 2:
        print("❌ Need at least original and one clustering dataset")
        return None
    
    # Evaluate each dataset
    all_results = []
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\n📈 Evaluating: {dataset_info['name']}")
        print(f"   Features: {dataset_info['features']}")
        
        results = evaluate_dataset(dataset_info, dataset_name)
        all_results.extend(results)
    
    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Display comparison
    print("\n" + "="*70)
    print("🎯 CLASSIFICATION RESULTS COMPARISON")
    print("="*70)
    
    # Create comparison table
    comparison_table = df_results.pivot_table(
        index=['dataset', 'dataset_name', 'n_features'], 
        columns='model', 
        values='accuracy',
        aggfunc='first'
    ).round(4)
    
    print(comparison_table)
    
    # Find best results
    print(f"\n🏆 BEST RESULTS:")
    best_overall = df_results.loc[df_results['accuracy'].idxmax()]
    print(f"   Overall Best: {best_overall['dataset_name']} with {best_overall['model']}")
    print(f"   Accuracy: {best_overall['accuracy']:.4f}")
    
    # Compare original vs clustering-enhanced
    print(f"\n📊 CLUSTERING IMPACT ANALYSIS:")
    original_results = df_results[df_results['dataset'] == 'original']
    original_best = original_results['accuracy'].max()
    
    print(f"   Original Features Best: {original_best:.4f}")
    
    improvements = {}
    for dataset_name in ['kmeans', 'hierarchical', 'dbscan']:
        if dataset_name in df_results['dataset'].values:
            dataset_results = df_results[df_results['dataset'] == dataset_name]
            dataset_best = dataset_results['accuracy'].max()
            improvement = dataset_best - original_best
            improvement_pct = (improvement / original_best) * 100
            improvements[dataset_name] = improvement
            
            status = "✅ Improved" if improvement > 0 else "❌ No improvement"
            print(f"   {dataset_name.upper()}: {dataset_best:.4f} ({improvement:+.4f} / {improvement_pct:+.1f}%) {status}")
    
    # Feature importance analysis for Random Forest
    print(f"\n🔍 FEATURE IMPORTANCE (Random Forest):")
    rf_results = df_results[df_results['model'] == 'Random Forest']
    
    for _, row in rf_results.iterrows():
        if 'feature_importance' in row and row['feature_importance']:
            print(f"\n   {row['dataset_name']}:")
            importance_dict = row['feature_importance']
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features:
                marker = " [CLUSTER]" if any(word in feature.lower() for word in ['cluster', 'dbscan', 'kmeans', 'hierarchical']) else ""
                print(f"     {feature}: {importance:.4f}{marker}")
    
    # Create simple visualization
    create_comparison_plot(df_results)
    
    # Save results
    df_results.to_csv('code/classification/clustering_comparison_results.csv', index=False)
    comparison_table.to_csv('code/classification/comparison_summary.csv')
    
    print(f"\n💾 Results saved to:")
    print(f"   - code/classification/clustering_comparison_results.csv")
    print(f"   - code/classification/comparison_summary.csv")
    print(f"   - code/classification/clustering_comparison_plot.png")
    
    # Summary conclusion
    best_clustering = max(improvements.items(), key=lambda x: x[1]) if improvements else None
    
    print(f"\n🎯 CONCLUSION:")
    if best_clustering and best_clustering[1] > 0:
        print(f"   ✅ Best clustering method: {best_clustering[0].upper()}")
        print(f"   ✅ Improvement: {best_clustering[1]:+.4f} accuracy points")
        print(f"   ✅ Clustering features help classification performance!")
    else:
        print(f"   ❌ Clustering features did not improve classification")
        print(f"   ❌ Original features alone perform best")
    
    return df_results, comparison_table


def create_comparison_plot(df_results):
    """Create visualization comparing results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy by dataset
    dataset_best = df_results.groupby('dataset_name')['accuracy'].max()
    bars1 = ax1.bar(range(len(dataset_best)), dataset_best.values, 
                    color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(dataset_best)])
    ax1.set_title('Best Accuracy by Dataset', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(range(len(dataset_best)))
    ax1.set_xticklabels([name.replace(' + ', '\n+\n') for name in dataset_best.index], 
                       rotation=0, ha='center', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, dataset_best.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Model comparison across datasets
    model_comparison = df_results.pivot_table(
        index='dataset_name', 
        columns='model', 
        values='accuracy'
    )
    
    model_comparison.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Model Performance Across Datasets', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('code/classification/clustering_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    results, summary = main_comparison()