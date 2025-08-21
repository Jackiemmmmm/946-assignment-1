"""
Enhanced Data Analysis Pipeline
==============================

This script runs the complete enhanced analysis pipeline including:
1. Enhanced data preprocessing with advanced feature engineering
2. Enhanced clustering with multiple algorithms and optimization
3. Enhanced classification with hyperparameter tuning and ensemble methods

Author: Enhanced ML Pipeline
Date: 2024
"""

import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

def create_results_summary():
    """Create a comprehensive results summary"""
    
    print("\n" + "="*80)
    print("🚀 ENHANCED MACHINE LEARNING ANALYSIS PIPELINE")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create directories for results
    os.makedirs('enhanced_visuals', exist_ok=True)
    os.makedirs('results_summary', exist_ok=True)
    
    results = {}
    
    # Step 1: Enhanced Preprocessing
    print("🔧 STEP 1: Enhanced Data Preprocessing")
    print("-" * 50)
    
    try:
        from enhanced_preprocessing import main_enhanced_preprocessing
        start_time = time.time()
        
        enhanced_data, scaled_data, scaler = main_enhanced_preprocessing()
        
        preprocessing_time = time.time() - start_time
        results['preprocessing'] = {
            'success': True,
            'time': preprocessing_time,
            'data_shape': scaled_data.shape if scaled_data is not None else None,
            'features_created': 'Advanced feature engineering completed'
        }
        
        print(f"✅ Preprocessing completed in {preprocessing_time:.2f} seconds")
        print(f"   Final dataset shape: {scaled_data.shape}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        results['preprocessing'] = {
            'success': False,
            'error': str(e),
            'time': 0
        }
        return results
    
    # Step 2: Enhanced Clustering
    print(f"\n🎯 STEP 2: Enhanced Clustering Analysis")
    print("-" * 50)
    
    try:
        from enhanced_clustering import main_enhanced_clustering
        start_time = time.time()
        
        clustering_results = main_enhanced_clustering()
        
        clustering_time = time.time() - start_time
        
        if clustering_results:
            best_silhouette = max([
                result.get('silhouette_score', -1) 
                for result in clustering_results.get('clustering_results', {}).values()
                if result.get('silhouette_score', -1) > -1
            ], default=-1)
            
            results['clustering'] = {
                'success': True,
                'time': clustering_time,
                'best_silhouette_score': best_silhouette,
                'algorithms_tested': list(clustering_results.get('clustering_results', {}).keys()),
                'best_strategy': clustering_results.get('best_strategy', 'Unknown')
            }
            
            print(f"✅ Clustering completed in {clustering_time:.2f} seconds")
            print(f"   Best silhouette score: {best_silhouette:.4f}")
            print(f"   Algorithms tested: {len(clustering_results.get('clustering_results', {}))}")
            
        else:
            results['clustering'] = {
                'success': False,
                'error': 'No clustering results obtained',
                'time': clustering_time
            }
            print(f"❌ Clustering failed: No results obtained")
            
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        results['clustering'] = {
            'success': False,
            'error': str(e),
            'time': 0
        }
    
    # Step 3: Enhanced Classification
    print(f"\n🎯 STEP 3: Enhanced Classification Analysis")
    print("-" * 50)
    
    try:
        from enhanced_classification import main_enhanced_classification
        start_time = time.time()
        
        classification_results = main_enhanced_classification()
        
        classification_time = time.time() - start_time
        
        if classification_results:
            best_accuracy = classification_results['best_result']['accuracy']
            best_algorithm = classification_results['best_result']['algorithm']
            
            results['classification'] = {
                'success': True,
                'time': classification_time,
                'best_accuracy': best_accuracy,
                'best_algorithm': best_algorithm,
                'algorithms_tested': len(classification_results['results']),
                'improvement_over_baseline': best_accuracy - 0.548  # Compared to original Random Forest
            }
            
            print(f"✅ Classification completed in {classification_time:.2f} seconds")
            print(f"   Best accuracy: {best_accuracy:.4f}")
            print(f"   Best algorithm: {best_algorithm}")
            print(f"   Algorithms tested: {len(classification_results['results'])}")
            
            if best_accuracy > 0.548:
                improvement = (best_accuracy - 0.548) * 100
                print(f"   🎉 Improvement over baseline: +{improvement:.2f}%")
            
        else:
            results['classification'] = {
                'success': False,
                'error': 'No classification results obtained',
                'time': classification_time
            }
            print(f"❌ Classification failed: No results obtained")
            
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        results['classification'] = {
            'success': False,
            'error': str(e),
            'time': 0
        }
    
    return results


def generate_final_report(results):
    """Generate a comprehensive final report"""
    
    total_time = sum([
        results.get('preprocessing', {}).get('time', 0),
        results.get('clustering', {}).get('time', 0),
        results.get('classification', {}).get('time', 0)
    ])
    
    print(f"\n" + "="*80)
    print("📊 FINAL ANALYSIS REPORT")
    print("="*80)
    
    # Summary statistics
    print("📈 PERFORMANCE SUMMARY:")
    print("-" * 30)
    
    if results.get('preprocessing', {}).get('success', False):
        data_shape = results['preprocessing']['data_shape']
        print(f"✅ Data Processing: SUCCESS")
        print(f"   • Dataset shape: {data_shape}")
        print(f"   • Processing time: {results['preprocessing']['time']:.2f}s")
        print(f"   • Advanced features: Created")
    else:
        print(f"❌ Data Processing: FAILED")
    
    print()
    
    if results.get('clustering', {}).get('success', False):
        best_sil = results['clustering']['best_silhouette_score']
        algorithms = len(results['clustering']['algorithms_tested'])
        print(f"✅ Clustering Analysis: SUCCESS")
        print(f"   • Best silhouette score: {best_sil:.4f}")
        print(f"   • Algorithms tested: {algorithms}")
        print(f"   • Analysis time: {results['clustering']['time']:.2f}s")
        
        # Clustering quality assessment
        if best_sil > 0.5:
            print(f"   • Quality: EXCELLENT 🌟")
        elif best_sil > 0.3:
            print(f"   • Quality: GOOD ✅")
        elif best_sil > 0.1:
            print(f"   • Quality: FAIR ⚠️")
        else:
            print(f"   • Quality: POOR ❌")
    else:
        print(f"❌ Clustering Analysis: FAILED")
    
    print()
    
    if results.get('classification', {}).get('success', False):
        best_acc = results['classification']['best_accuracy']
        best_alg = results['classification']['best_algorithm']
        algorithms = results['classification']['algorithms_tested']
        improvement = results['classification'].get('improvement_over_baseline', 0)
        
        print(f"✅ Classification Analysis: SUCCESS")
        print(f"   • Best accuracy: {best_acc:.4f}")
        print(f"   • Best algorithm: {best_alg}")
        print(f"   • Algorithms tested: {algorithms}")
        print(f"   • Analysis time: {results['classification']['time']:.2f}s")
        
        if improvement > 0:
            print(f"   • Improvement: +{improvement*100:.2f}% 🚀")
        else:
            print(f"   • Improvement: {improvement*100:.2f}%")
            
        # Classification quality assessment
        if best_acc > 0.8:
            print(f"   • Quality: EXCELLENT 🌟")
        elif best_acc > 0.65:
            print(f"   • Quality: GOOD ✅")
        elif best_acc > 0.5:
            print(f"   • Quality: FAIR ⚠️")
        else:
            print(f"   • Quality: POOR ❌")
    else:
        print(f"❌ Classification Analysis: FAILED")
    
    print(f"\n⏱️  TOTAL ANALYSIS TIME: {total_time:.2f} seconds")
    
    # Technical improvements summary
    print(f"\n🔧 TECHNICAL IMPROVEMENTS:")
    print("-" * 30)
    print("• Advanced feature engineering (price efficiency, value scores, interactions)")
    print("• Intelligent outlier detection with adaptive IQR")
    print("• Robust feature scaling and transformation")
    print("• Multiple clustering algorithms (K-Means, DBSCAN, Gaussian Mixture, etc.)")
    print("• Comprehensive hyperparameter optimization")
    print("• Class imbalance handling in classification")
    print("• Ensemble methods for improved accuracy")
    print("• Advanced visualization and evaluation metrics")
    
    # Files generated
    print(f"\n📁 GENERATED FILES:")
    print("-" * 20)
    
    generated_files = [
        'enhanced_preprocessed_data.csv',
        'enhanced_normalized_data.csv',
        'enhanced_clustering_comparison.csv',
        'enhanced_classification_results.csv'
    ]
    
    for filename in generated_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"✅ {filename} ({file_size:.1f} MB)")
        else:
            print(f"❌ {filename} (not found)")
    
    # Visual outputs
    print(f"\n🎨 VISUALIZATIONS:")
    print("-" * 20)
    
    visual_files = [
        'enhanced_visuals/feature_correlation.png',
        'enhanced_visuals/advanced_clustering_results.png',
        'enhanced_visuals/advanced_classification_analysis.png'
    ]
    
    for filename in visual_files:
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} (not found)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if results.get('classification', {}).get('best_accuracy', 0) < 0.7:
        print("• Consider collecting more diverse features")
        print("• Try deep learning approaches for complex patterns")
        print("• Investigate feature interactions further")
    
    if results.get('clustering', {}).get('best_silhouette_score', 0) < 0.3:
        print("• Consider domain-specific clustering approaches")
        print("• Try different distance metrics")
        print("• Investigate hierarchical clustering")
    
    print("• Monitor model performance over time")
    print("• Consider A/B testing for production deployment")
    print("• Validate results with domain experts")
    
    print(f"\n" + "="*80)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Thank you for using the Enhanced ML Pipeline! 🎉")
    print("="*80)
    
    # Save report to file
    save_report_to_file(results)


def save_report_to_file(results):
    """Save the analysis report to a file"""
    
    report_filename = f"results_summary/enhanced_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("Enhanced Machine Learning Analysis Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write results summary
            for step, result in results.items():
                f.write(f"{step.upper()}:\n")
                f.write("-" * 20 + "\n")
                for key, value in result.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"📄 Report saved to: {report_filename}")
        
    except Exception as e:
        print(f"⚠️  Could not save report to file: {e}")


def main():
    """Main function to run the complete enhanced analysis"""
    
    print("🚀 Starting Enhanced Machine Learning Analysis Pipeline...")
    print("This may take several minutes depending on your data size.")
    print()
    
    # Check if data files exist
    required_files = ['../assignment/A1_2025_Released/']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠️  Warning: Required data directory not found: {missing_files}")
        print("Please ensure the data directory exists and contains CSV files.")
        print()
    
    try:
        # Run complete analysis pipeline
        results = create_results_summary()
        
        # Generate comprehensive report
        generate_final_report(results)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Analysis interrupted by user.")
        return
    
    except Exception as e:
        print(f"❌ Fatal error in analysis pipeline: {e}")
        print("Please check the error details and try again.")
        return
    
    print(f"\n🎯 Analysis pipeline completed successfully!")
    print("Check the 'enhanced_visuals' directory for plots and visualizations.")
    print("Check the generated CSV files for detailed results.")


if __name__ == "__main__":
    main()