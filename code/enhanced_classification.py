import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, validation_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import warnings
import os

warnings.filterwarnings("ignore")


def prepare_enhanced_classification_data(df, feature_strategy='auto', target_col='category'):
    """
    Enhanced classification data preparation with advanced feature selection
    """
    print(f"=== ENHANCED CLASSIFICATION DATA PREPARATION ({feature_strategy}) ===")
    
    df_temp = df.copy()
    
    # Handle boolean columns
    if 'is_new' in df_temp.columns:
        df_temp['is_new_numeric'] = df_temp['is_new'].astype(int)
    
    # Get all available features
    potential_features = []
    
    # Core features
    core_features = ['current_price', 'discount', 'likes_count', 'is_new_numeric']
    potential_features.extend([col for col in core_features if col in df_temp.columns])
    
    # Engineered features
    engineered_features = [
        'price_efficiency', 'discount_amount', 'value_score', 'price_per_like',
        'category_size', 'category_price_ratio', 'brand_frequency', 'brand_price_premium',
        'price_discount_interaction', 'discount_likes_interaction', 'likes_log'
    ]
    potential_features.extend([col for col in engineered_features if col in df_temp.columns])
    
    # Statistical features
    statistical_features = [col for col in df_temp.columns if 
                           col.endswith('_mean') or col.endswith('_std') or 
                           col.endswith('_ratio') or col.endswith('_score')]
    potential_features.extend(statistical_features)
    
    # Remove duplicates and target column
    potential_features = list(set(potential_features))
    if target_col in potential_features:
        potential_features.remove(target_col)
    
    print(f"Found {len(potential_features)} potential features")
    
    # Check target availability
    if target_col not in df_temp.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Prepare target variable
    y = df_temp[target_col].copy()
    
    # Handle missing values in target
    if y.isnull().sum() > 0:
        print(f"Removing {y.isnull().sum()} rows with missing target values")
        valid_mask = y.notna()
        y = y[valid_mask]
        df_temp = df_temp[valid_mask]
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Feature selection strategy
    if feature_strategy == 'all':
        feature_columns = potential_features
    elif feature_strategy == 'basic':
        feature_columns = [col for col in core_features if col in df_temp.columns]
    elif feature_strategy == 'auto':
        # Intelligent feature selection
        X_all = df_temp[potential_features].fillna(0)
        
        # Remove constant and quasi-constant features
        feature_variances = X_all.var()
        variable_features = feature_variances[feature_variances > 0.01].index.tolist()
        
        print(f"Removed {len(potential_features) - len(variable_features)} low-variance features")
        
        # Select best features using mutual information
        if len(variable_features) > 20:  # If too many features, select best ones
            selector = SelectKBest(score_func=mutual_info_classif, k=min(15, len(variable_features)))
            X_selected = selector.fit_transform(X_all[variable_features], y_encoded)
            feature_columns = np.array(variable_features)[selector.get_support()].tolist()
        else:
            feature_columns = variable_features
    else:
        feature_columns = potential_features
    
    # Final feature matrix
    X = df_temp[feature_columns].fillna(0)
    
    print(f"Selected {len(feature_columns)} features for classification:")
    for i, feature in enumerate(feature_columns[:10]):  # Show first 10
        print(f"  {i+1}. {feature}")
    if len(feature_columns) > 10:
        print(f"  ... and {len(feature_columns) - 10} more")
    
    print(f"Classification dataset shape: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Class distribution
    class_counts = pd.Series(y).value_counts()
    print(f"\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(y)*100:.1f}%)")
    
    # Check for class imbalance
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    imbalance_ratio = max_class_count / min_class_count
    
    if imbalance_ratio > 3:
        print(f"⚠️  Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        print("  Consider using class weighting or resampling techniques")
    
    return X, y_encoded, label_encoder, feature_columns


def perform_enhanced_knn_classification(X_train, X_test, y_train, y_test, label_encoder):
    """
    Enhanced KNN with distance weighting and optimization
    """
    print(f"\n=== ENHANCED K-NEAREST NEIGHBORS ===")
    
    # Extended parameter grid
    param_grid = {
        'n_neighbors': range(3, 31, 2),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Grid search with cross-validation
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_knn = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Predictions and metrics
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5)
    
    return {
        'algorithm': 'Enhanced KNN',
        'model': best_knn,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'cv_scores': cv_scores
    }


def perform_enhanced_naive_bayes(X_train, X_test, y_train, y_test, label_encoder):
    """
    Enhanced Naive Bayes with hyperparameter tuning
    """
    print(f"\n=== ENHANCED NAIVE BAYES ===")
    
    # Try different smoothing parameters
    param_grid = {'var_smoothing': np.logspace(0, -9, num=20)}
    
    nb = GaussianNB()
    grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_nb = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Predictions and metrics
    y_pred = best_nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_nb, X_train, y_train, cv=5)
    
    return {
        'algorithm': 'Enhanced Naive Bayes',
        'model': best_nb,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'cv_scores': cv_scores
    }


def perform_enhanced_random_forest(X_train, X_test, y_train, y_test, label_encoder):
    """
    Enhanced Random Forest with extensive hyperparameter tuning
    """
    print(f"\n=== ENHANCED RANDOM FOREST ===")
    
    # Class weights for imbalanced data
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    
    # Extended parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': ['balanced', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Predictions and metrics
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_importance = best_rf.feature_importances_
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
    
    return {
        'algorithm': 'Enhanced Random Forest',
        'model': best_rf,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance
    }


def perform_gradient_boosting_classification(X_train, X_test, y_train, y_test, label_encoder):
    """
    Gradient Boosting classifier (new algorithm)
    """
    print(f"\n=== GRADIENT BOOSTING CLASSIFIER ===")
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        gb, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_gb = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Predictions and metrics
    y_pred = best_gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_importance = best_gb.feature_importances_
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_gb, X_train, y_train, cv=5)
    
    return {
        'algorithm': 'Gradient Boosting',
        'model': best_gb,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance
    }


def perform_svm_classification(X_train, X_test, y_train, y_test, label_encoder):
    """
    Support Vector Machine classifier (new algorithm)
    """
    print(f"\n=== SUPPORT VECTOR MACHINE ===")
    
    # Reduced parameter grid due to computational complexity
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'class_weight': ['balanced', None]
    }
    
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_svm = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Predictions and metrics
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_svm, X_train, y_train, cv=5)
    
    return {
        'algorithm': 'SVM',
        'model': best_svm,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'cv_scores': cv_scores
    }


def create_ensemble_classifier(X_train, X_test, y_train, y_test, base_models, label_encoder):
    """
    Create ensemble classifier using best performing models
    """
    print(f"\n=== ENSEMBLE CLASSIFIER ===")
    
    # Select top performing models for ensemble
    valid_models = [(name, result['model']) for name, result in base_models.items() 
                   if result.get('accuracy', 0) > 0.3]  # Only include reasonably performing models
    
    if len(valid_models) < 2:
        print("Not enough valid models for ensemble")
        return None
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=valid_models[:5],  # Use top 5 models to avoid overfitting
        voting='hard'  # Use hard voting
    )
    
    ensemble.fit(X_train, y_train)
    
    # Predictions and metrics
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Ensemble models: {[name for name, _ in valid_models[:5]]}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5)
    
    return {
        'algorithm': 'Ensemble Voting',
        'model': ensemble,
        'best_params': {'models': [name for name, _ in valid_models[:5]]},
        'accuracy': accuracy,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'cv_scores': cv_scores
    }


def create_advanced_visualizations(results_list, X_test, y_test, label_encoder, feature_names):
    """
    Create advanced visualizations for classification results
    """
    print("\n=== ADVANCED CLASSIFICATION VISUALIZATION ===")
    
    os.makedirs('enhanced_visuals', exist_ok=True)
    
    # 1. Comprehensive performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    algorithms = [r['algorithm'] for r in results_list if r is not None]
    accuracies = [r['accuracy'] for r in results_list if r is not None]
    cv_scores = [r['cv_score'] for r in results_list if r is not None]
    cv_stds = [r.get('cv_std', 0) for r in results_list if r is not None]
    
    # Performance comparison with error bars
    ax1 = axes[0, 0]
    x_pos = np.arange(len(algorithms))
    
    bars1 = ax1.bar(x_pos - 0.2, accuracies, 0.4, label='Test Accuracy', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x_pos + 0.2, cv_scores, 0.4, yerr=cv_stds, 
                   label='CV Score ± Std', alpha=0.8, color='lightcoral', capsize=5)
    
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_title('Algorithm Performance Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (acc, cv) in enumerate(zip(accuracies, cv_scores)):
        ax1.text(i - 0.2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + 0.2, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Feature importance comparison (for tree-based models)
    ax2 = axes[0, 1]
    importance_data = {}
    
    for result in results_list:
        if result is not None and 'feature_importance' in result:
            importance_data[result['algorithm']] = result['feature_importance']
    
    if importance_data and feature_names:
        # Create importance DataFrame
        importance_df = pd.DataFrame(importance_data, index=feature_names)
        
        # Plot top 10 most important features
        mean_importance = importance_df.mean(axis=1).sort_values(ascending=True)
        top_features = mean_importance.tail(10)
        
        ax2.barh(range(len(top_features)), top_features.values, alpha=0.8)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features.index)
        ax2.set_xlabel('Average Feature Importance')
        ax2.set_title('Top 10 Most Important Features')
        ax2.grid(True, alpha=0.3, axis='x')
    else:
        ax2.text(0.5, 0.5, 'No feature importance data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Feature Importance')
    
    # 3. Confusion matrix for best algorithm
    best_result = max([r for r in results_list if r is not None], key=lambda x: x['accuracy'])
    
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, best_result['predictions'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=ax3)
    ax3.set_title(f'Confusion Matrix - {best_result["algorithm"]}')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Model complexity vs performance
    ax4 = axes[1, 1]
    
    # Extract model complexity metrics where available
    complexity_metrics = []
    performance_metrics = []
    model_names = []
    
    for result in results_list:
        if result is None:
            continue
            
        model_names.append(result['algorithm'])
        performance_metrics.append(result['accuracy'])
        
        # Estimate complexity based on model type
        if 'Random Forest' in result['algorithm'] or 'Gradient Boosting' in result['algorithm']:
            # Use number of estimators as complexity measure
            n_estimators = result['best_params'].get('n_estimators', 100)
            complexity_metrics.append(n_estimators)
        elif 'KNN' in result['algorithm']:
            # Use inverse of k (more neighbors = simpler model)
            k = result['best_params'].get('n_neighbors', 5)
            complexity_metrics.append(100 / k)  # Scale for visualization
        else:
            # Default complexity
            complexity_metrics.append(50)
    
    scatter = ax4.scatter(complexity_metrics, performance_metrics, 
                         s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    
    for i, name in enumerate(model_names):
        ax4.annotate(name.split()[0], (complexity_metrics[i], performance_metrics[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Model Complexity (Relative)')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Model Complexity vs Performance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_visuals/advanced_classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Detailed classification report visualization
    create_detailed_classification_report(results_list, y_test, label_encoder)


def create_detailed_classification_report(results_list, y_test, label_encoder):
    """
    Create detailed classification report visualization
    """
    best_result = max([r for r in results_list if r is not None], key=lambda x: x['accuracy'])
    
    # Get classification report as dict
    report = classification_report(y_test, best_result['predictions'], 
                                 target_names=label_encoder.classes_, 
                                 output_dict=True)
    
    # Create DataFrame from report
    df_report = pd.DataFrame(report).transpose()
    
    # Remove unnecessary rows
    df_metrics = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_metrics.iloc[:, :-1], annot=True, fmt='.3f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Score'})
    plt.title(f'Detailed Classification Report - {best_result["algorithm"]}')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.tight_layout()
    plt.savefig('enhanced_visuals/detailed_classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()


def main_enhanced_classification():
    """
    Main enhanced classification pipeline
    """
    print("Starting Enhanced Classification Analysis...")
    
    # Load enhanced data
    try:
        df = pd.read_csv("enhanced_normalized_data.csv")
        print(f"Loaded enhanced data: {df.shape}")
    except FileNotFoundError:
        print("Enhanced data not found. Using original normalized data...")
        try:
            df = pd.read_csv("normalized_data.csv")
            print(f"Loaded original data: {df.shape}")
        except FileNotFoundError:
            print("No data available. Please run preprocessing first!")
            return None
    
    # Prepare data with automatic feature selection
    X, y, label_encoder, feature_names = prepare_enhanced_classification_data(
        df, feature_strategy='auto'
    )
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Run all classification algorithms
    results = {}
    
    print(f"\n{'='*60}")
    print("RUNNING ENHANCED CLASSIFICATION ALGORITHMS")
    print(f"{'='*60}")
    
    # 1. Enhanced KNN
    try:
        results['knn'] = perform_enhanced_knn_classification(
            X_train, X_test, y_train, y_test, label_encoder
        )
    except Exception as e:
        print(f"Error with Enhanced KNN: {e}")
        results['knn'] = None
    
    # 2. Enhanced Naive Bayes
    try:
        results['nb'] = perform_enhanced_naive_bayes(
            X_train, X_test, y_train, y_test, label_encoder
        )
    except Exception as e:
        print(f"Error with Enhanced Naive Bayes: {e}")
        results['nb'] = None
    
    # 3. Enhanced Random Forest
    try:
        results['rf'] = perform_enhanced_random_forest(
            X_train, X_test, y_train, y_test, label_encoder
        )
    except Exception as e:
        print(f"Error with Enhanced Random Forest: {e}")
        results['rf'] = None
    
    # 4. Gradient Boosting (new)
    try:
        results['gb'] = perform_gradient_boosting_classification(
            X_train, X_test, y_train, y_test, label_encoder
        )
    except Exception as e:
        print(f"Error with Gradient Boosting: {e}")
        results['gb'] = None
    
    # 5. SVM (new)
    try:
        if X_train.shape[0] <= 10000:  # Only run SVM if dataset is not too large
            results['svm'] = perform_svm_classification(
                X_train, X_test, y_train, y_test, label_encoder
            )
        else:
            print("Dataset too large for SVM, skipping...")
            results['svm'] = None
    except Exception as e:
        print(f"Error with SVM: {e}")
        results['svm'] = None
    
    # 6. Ensemble classifier
    try:
        valid_results = {k: v for k, v in results.items() if v is not None}
        if len(valid_results) >= 2:
            results['ensemble'] = create_ensemble_classifier(
                X_train, X_test, y_train, y_test, valid_results, label_encoder
            )
    except Exception as e:
        print(f"Error with Ensemble: {e}")
        results['ensemble'] = None
    
    # Filter valid results
    valid_results = [r for r in results.values() if r is not None]
    
    if not valid_results:
        print("No successful classification results!")
        return None
    
    # Create advanced visualizations
    create_advanced_visualizations(valid_results, X_test, y_test, label_encoder, feature_names)
    
    # Create comprehensive comparison
    print(f"\n{'='*60}")
    print("ENHANCED CLASSIFICATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    comparison_data = []
    for result in valid_results:
        comparison_data.append({
            'Algorithm': result['algorithm'],
            'Test_Accuracy': result['accuracy'],
            'CV_Score': result['cv_score'],
            'CV_Std': result.get('cv_std', 0),
            'Best_Params': str(result['best_params'])[:50] + '...' if len(str(result['best_params'])) > 50 else str(result['best_params'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Save results
    comparison_df.to_csv('enhanced_classification_results.csv', index=False)
    
    # Find best model
    best_result = max(valid_results, key=lambda x: x['accuracy'])
    
    print(f"\n🏆 BEST PERFORMING ALGORITHM:")
    print(f"   Algorithm: {best_result['algorithm']}")
    print(f"   Test Accuracy: {best_result['accuracy']:.4f}")
    print(f"   CV Score: {best_result['cv_score']:.4f} ± {best_result.get('cv_std', 0):.4f}")
    print(f"   Best Parameters: {best_result['best_params']}")
    
    print("\n=== ENHANCED CLASSIFICATION COMPLETE ===")
    print("Improvements achieved:")
    print("  • Advanced hyperparameter optimization")
    print("  • Intelligent feature selection")
    print("  • Class imbalance handling")
    print("  • Multiple new algorithms (Gradient Boosting, SVM)")
    print("  • Ensemble methods")
    print("  • Comprehensive evaluation metrics")
    print("  • Advanced visualization and reporting")
    
    return {
        'results': valid_results,
        'comparison_df': comparison_df,
        'best_result': best_result,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    enhanced_classification_results = main_enhanced_classification()