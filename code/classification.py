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
import warnings

warnings.filterwarnings("ignore")


def prepare_classification_data(df):
    """
    Prepare data for classification analysis
    """
    print("=== CLASSIFICATION DATA PREPARATION ===")

    # Features for classification (same as clustering)
    feature_columns = ["current_price", "discount", "likes_count"]

    # Add is_new if available
    if "is_new" in df.columns:
        # Create a temporary dataframe with is_new converted to numeric
        df_temp = df.copy()
        df_temp["is_new_numeric"] = df_temp["is_new"].astype(int)
        feature_columns.append("is_new_numeric")
        print("Added 'is_new' as numeric feature for classification")
    else:
        df_temp = df.copy()
        print("'is_new' column not found, proceeding without it")

    # Target variable
    target_column = "category"

    # Check if required columns exist
    missing_cols = [
        col for col in feature_columns + [target_column] if col not in df_temp.columns
    ]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None, None, None, None

    # Extract features and target
    X = df_temp[feature_columns].copy()
    y = df_temp[target_column].copy()

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"Classification dataset shape: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Feature columns: {feature_columns}")

    # Class distribution
    class_counts = pd.Series(y).value_counts()
    print(f"\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(y)*100:.1f}%)")

    return X, y_encoded, label_encoder, feature_columns


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    print(f"\n=== DATA SPLITTING ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Feature dimensions: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def perform_knn_classification(X_train, X_test, y_train, y_test, label_encoder):
    """
    Perform K-Nearest Neighbors classification
    """
    print(f"\n=== K-NEAREST NEIGHBORS CLASSIFICATION ===")

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
        "algorithm": "K-Nearest Neighbors",
        "model": knn_best,
        "best_params": {"n_neighbors": best_k},
        "accuracy": accuracy,
        "cv_score": best_cv_score,
        "predictions": y_pred,
        "cv_scores": cv_scores_final,
    }


def perform_naive_bayes_classification(X_train, X_test, y_train, y_test, label_encoder):
    """
    Perform Naive Bayes classification
    """
    print(f"\n=== NAIVE BAYES CLASSIFICATION ===")

    # Initialize and train Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Make predictions
    y_pred = nb.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")

    # Detailed classification report
    class_names = label_encoder.classes_
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Cross-validation scores
    cv_scores = cross_val_score(nb, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Feature probabilities for each class
    print(f"\nClass priors: {nb.class_prior_}")

    return {
        "algorithm": "Naive Bayes",
        "model": nb,
        "best_params": {},
        "accuracy": accuracy,
        "cv_score": cv_scores.mean(),
        "predictions": y_pred,
        "cv_scores": cv_scores,
    }


def perform_random_forest_classification(
    X_train, X_test, y_train, y_test, label_encoder
):
    """
    Perform Random Forest classification (bonus algorithm)
    """
    print(f"\n=== RANDOM FOREST CLASSIFICATION ===")

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
    feature_names = ["current_price", "discount", "likes_count"]

    # Add is_new feature name if it exists
    if len(feature_importance) > 3:
        feature_names.append("is_new")

    print(f"\nFeature Importance:")
    for feature, importance in zip(feature_names, feature_importance):
        print(f"  {feature}: {importance:.4f}")

    # Calculate cross-validation scores for consistency with other algorithms
    cv_scores = cross_val_score(rf_best, X_train, y_train, cv=5)
    cv_score_mean = cv_scores.mean()

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_score_mean:.4f} (+/- {cv_scores.std() * 2:.4f})")

    return {
        "algorithm": "Random Forest",
        "model": rf_best,
        "best_params": grid_search.best_params_,
        "accuracy": accuracy,
        "cv_score": cv_score_mean,  # Use the mean of CV scores
        "predictions": y_pred,
        "cv_scores": cv_scores,  # Add the full CV scores array
        "feature_importance": dict(zip(feature_names, feature_importance)),
    }


def visualize_classification_results(results_list, X_test, y_test, label_encoder):
    """
    Visualize classification results
    """
    print("\n=== CLASSIFICATION VISUALIZATION ===")

    n_algorithms = len(results_list)

    # Filter out results that don't have cv_scores (if any)
    valid_results = [r for r in results_list if "cv_scores" in r or "cv_score" in r]
    n_valid = len(valid_results)

    if n_valid == 0:
        print("No valid results to visualize")
        return

    # Create subplots - 2 rows, n_valid columns
    fig, axes = plt.subplots(2, n_valid, figsize=(6 * n_valid, 10))

    # Handle single algorithm case
    if n_valid == 1:
        axes = axes.reshape(2, 1)
    elif n_valid == 2:
        # axes is already correct shape
        pass

    class_names = label_encoder.classes_

    for i, result in enumerate(valid_results):
        # Confusion Matrix (top row)
        cm = confusion_matrix(y_test, result["predictions"])

        # Plot confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0, i],
        )
        axes[0, i].set_title(f'{result["algorithm"]}\nConfusion Matrix')
        axes[0, i].set_xlabel("Predicted")
        axes[0, i].set_ylabel("Actual")

        # Performance metrics (bottom row)
        cv_score = result.get("cv_score", 0)
        test_accuracy = result.get("accuracy", 0)

        # Create bar chart
        metrics = ["CV Score", "Test Accuracy"]
        values = [cv_score, test_accuracy]

        bars = axes[1, i].bar(metrics, values, color=["skyblue", "lightcoral"])
        axes[1, i].set_title(f'{result["algorithm"]}\nPerformance Metrics')
        axes[1, i].set_ylabel("Score")
        axes[1, i].set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, i].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # Add grid for better readability
        axes[1, i].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        "classification/classification_results.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Create a separate detailed performance comparison chart
    plt.figure(figsize=(12, 6))

    # Prepare data for comparison
    algorithms = [r["algorithm"] for r in valid_results]
    cv_scores = [r.get("cv_score", 0) for r in valid_results]
    test_accuracies = [r.get("accuracy", 0) for r in valid_results]

    x = np.arange(len(algorithms))
    width = 0.35

    # Create grouped bar chart
    plt.bar(
        x - width / 2, cv_scores, width, label="CV Score", alpha=0.8, color="skyblue"
    )
    plt.bar(
        x + width / 2,
        test_accuracies,
        width,
        label="Test Accuracy",
        alpha=0.8,
        color="lightcoral",
    )

    # Customize the chart
    plt.xlabel("Algorithms")
    plt.ylabel("Score")
    plt.title("Classification Algorithms Performance Comparison")
    plt.xticks(x, algorithms, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (cv, acc) in enumerate(zip(cv_scores, test_accuracies)):
        plt.text(i - width / 2, cv + 0.01, f"{cv:.3f}", ha="center", va="bottom")
        plt.text(i + width / 2, acc + 0.01, f"{acc:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(
        "classification/classification_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("Classification visualizations saved:")
    print("  - classification_results.png (confusion matrices and metrics)")
    print("  - classification_performance_comparison.png (algorithm comparison)")


def analyze_classification_errors(results_list, X_test, y_test, label_encoder):
    """
    Analyze classification errors and misclassifications
    """
    print("\n=== CLASSIFICATION ERROR ANALYSIS ===")

    class_names = label_encoder.classes_

    for result in results_list:
        print(f"\n--- {result['algorithm']} Error Analysis ---")

        y_pred = result["predictions"]

        # Find misclassified instances
        misclassified_mask = y_test != y_pred
        misclassified_count = misclassified_mask.sum()

        print(
            f"Total misclassified: {misclassified_count} out of {len(y_test)} ({misclassified_count/len(y_test)*100:.1f}%)"
        )

        if misclassified_count > 0:
            # Analyze misclassification patterns
            misclass_df = pd.DataFrame(
                {
                    "actual": [class_names[i] for i in y_test[misclassified_mask]],
                    "predicted": [class_names[i] for i in y_pred[misclassified_mask]],
                }
            )

            print("Most common misclassifications:")
            misclass_patterns = (
                misclass_df.groupby(["actual", "predicted"])
                .size()
                .sort_values(ascending=False)
            )
            print(misclass_patterns.head(10))


def compare_classification_algorithms(results_list):
    """
    Compare different classification algorithms
    """
    print("\n=== CLASSIFICATION ALGORITHM COMPARISON ===")

    # Create comparison table
    comparison_data = []
    for result in results_list:
        comparison_data.append(
            {
                "Algorithm": result["algorithm"],
                "Test Accuracy": result["accuracy"],
                "CV Score": result["cv_score"],
                "Best Parameters": str(result["best_params"]),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Determine best algorithm
    best_algorithm = max(results_list, key=lambda x: x["accuracy"])
    print(f"\nBest performing algorithm: {best_algorithm['algorithm']}")
    print(f"Best test accuracy: {best_algorithm['accuracy']:.4f}")

    # Statistical significance test could be added here
    return comparison_df


def feature_importance_analysis(results_list, feature_names):
    """
    Analyze feature importance across algorithms
    """
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

    # Collect feature importance from algorithms that provide it
    importance_data = {}

    for result in results_list:
        algorithm_name = result["algorithm"]

        if "feature_importance" in result and result["feature_importance"]:
            # Random Forest provides feature importance directly
            importance_data[algorithm_name] = result["feature_importance"]
            print(f"{algorithm_name}: Feature importance available")

        elif algorithm_name == "K-Nearest Neighbors":
            # For KNN, we can't get direct feature importance, but we can note this
            print(
                f"{algorithm_name}: Feature importance not directly available (distance-based algorithm)"
            )

        elif algorithm_name == "Naive Bayes":
            # For Naive Bayes, we can analyze feature variances or use dummy importance
            print(
                f"{algorithm_name}: Feature importance not directly available (probabilistic algorithm)"
            )
            # We could add a placeholder or calculate based on feature variances

        else:
            print(f"{algorithm_name}: Feature importance not available")

    # Debug: Print the collected importance data
    print(f"\nCollected importance data: {importance_data}")

    if importance_data:
        # Convert to DataFrame for easier handling
        try:
            importance_df = pd.DataFrame(importance_data)
            print(f"Importance DataFrame shape: {importance_df.shape}")
            print(f"Importance DataFrame:\n{importance_df}")

            if not importance_df.empty:
                # Create the visualization
                fig, ax = plt.subplots(figsize=(10, 6))

                # Create bar plot directly on the axis
                importance_df.plot(kind="bar", ax=ax, alpha=0.8, width=0.8)

                ax.set_title(
                    "Feature Importance Comparison Across Algorithms",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.set_xlabel("Features", fontsize=12)
                ax.set_ylabel("Importance Score", fontsize=12)
                ax.tick_params(axis="x", rotation=45)

                # Improve legend positioning
                ax.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
                ax.grid(True, alpha=0.3, axis="y")

                # Add value labels on bars
                for container in ax.containers:
                    # Get the heights and add labels
                    labels = [
                        f"{v:.3f}" if v > 0.001 else "" for v in container.datavalues
                    ]
                    ax.bar_label(
                        container, labels=labels, rotation=0, fontsize=9, padding=3
                    )

                plt.tight_layout()
                plt.savefig(
                    "classification/feature_importance.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()

                print("Feature importance comparison saved to 'feature_importance.png'")

                # Print summary table
                print("\nFeature Importance Summary:")
                print(importance_df.round(4))

            else:
                print("Warning: Importance DataFrame is empty")

        except Exception as e:
            print(f"Error creating importance DataFrame: {e}")
            print("Attempting alternative visualization...")

            # Alternative approach: create simple bar plot
            if importance_data:
                # Get the first algorithm's data to determine features
                first_algo = list(importance_data.keys())[0]
                features = list(importance_data[first_algo].keys())

                plt.figure(figsize=(10, 6))

                # Create bars for each algorithm
                x = np.arange(len(features))
                width = 0.8 / len(importance_data)

                for i, (algo_name, feature_dict) in enumerate(importance_data.items()):
                    values = [feature_dict.get(feature, 0) for feature in features]
                    offset = (i - len(importance_data) / 2 + 0.5) * width

                    bars = plt.bar(
                        x + offset, values, width, label=algo_name, alpha=0.8
                    )

                    # Add value labels
                    for bar, value in zip(bars, values):
                        if value > 0.001:
                            plt.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.001,
                                f"{value:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                            )

                plt.xlabel("Features", fontsize=12)
                plt.ylabel("Importance Score", fontsize=12)
                plt.title(
                    "Feature Importance Comparison (Alternative View)",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.xticks(x, features, rotation=45, ha="right")
                plt.legend(title="Algorithm")
                plt.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()

                plt.savefig(
                    "classification/feature_importance_alternative.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()

                print(
                    "Alternative feature importance plot saved to 'feature_importance_alternative.png'"
                )
            else:
                create_manual_importance_plot(importance_data)

    else:
        print("No feature importance data available for visualization")
        print(
            "This is normal for distance-based (KNN) and probabilistic (Naive Bayes) algorithms"
        )

        # Create alternative analysis
        create_alternative_feature_analysis(results_list, feature_names)


def create_manual_importance_plot(importance_data):
    """
    Create feature importance plot manually when DataFrame approach fails
    """
    if not importance_data:
        return

    plt.figure(figsize=(10, 6))

    # Get all unique features
    all_features = set()
    for algorithm_importances in importance_data.values():
        all_features.update(algorithm_importances.keys())

    all_features = sorted(list(all_features))

    # Plot bars for each algorithm
    x = np.arange(len(all_features))
    width = 0.8 / len(importance_data)  # Width of bars

    for i, (algorithm, importances) in enumerate(importance_data.items()):
        values = [importances.get(feature, 0) for feature in all_features]
        offset = (i - len(importance_data) / 2 + 0.5) * width

        bars = plt.bar(x + offset, values, width, label=algorithm, alpha=0.8)

        # Add value labels
        for bar, value in zip(bars, values):
            if value > 0:  # Only label non-zero values
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.title("Feature Importance Comparison Across Algorithms")
    plt.xticks(x, all_features, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    plt.savefig(
        "classification/feature_importance_manual.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("Manual feature importance plot saved to 'feature_importance_manual.png'")


def create_alternative_feature_analysis(results_list, feature_names):
    """
    Create alternative analysis when feature importance is not available
    """
    print("\nCreating alternative feature analysis...")

    # Analyze model performance instead
    algorithms = [r["algorithm"] for r in results_list]
    accuracies = [r.get("accuracy", 0) for r in results_list]
    cv_scores = [r.get("cv_score", 0) for r in results_list]

    plt.figure(figsize=(10, 6))

    x = np.arange(len(algorithms))
    width = 0.35

    plt.bar(
        x - width / 2, cv_scores, width, label="CV Score", alpha=0.8, color="skyblue"
    )
    plt.bar(
        x + width / 2,
        accuracies,
        width,
        label="Test Accuracy",
        alpha=0.8,
        color="lightcoral",
    )

    plt.xlabel("Algorithms")
    plt.ylabel("Score")
    plt.title("Algorithm Performance Analysis (Alternative to Feature Importance)")
    plt.xticks(x, algorithms, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (cv, acc) in enumerate(zip(cv_scores, accuracies)):
        plt.text(i - width / 2, cv + 0.01, f"{cv:.3f}", ha="center", va="bottom")
        plt.text(i + width / 2, acc + 0.01, f"{acc:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("algorithm_performance_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(
        "Algorithm performance analysis saved to 'algorithm_performance_analysis.png'"
    )
    print(f"\nUsed features in analysis: {feature_names}")
    print("Note: Only Random Forest provides direct feature importance.")
    print("KNN and Naive Bayes use different mechanisms for feature evaluation.")


def main_classification():
    """
    Main classification analysis pipeline
    """
    print("Starting classification analysis...")

    # Load preprocessed data
    try:
        df = pd.read_csv("normalized_data.csv")
        print(f"Loaded normalized data: {df.shape}")
    except FileNotFoundError:
        print("Error: Please run data preprocessing first!")
        return

    # Prepare classification data
    X, y, label_encoder, feature_names = prepare_classification_data(df)
    if X is None:
        return

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Perform different classification algorithms
    print("Running classification algorithms...")

    # K-Nearest Neighbors
    knn_results = perform_knn_classification(
        X_train, X_test, y_train, y_test, label_encoder
    )

    # Naive Bayes
    nb_results = perform_naive_bayes_classification(
        X_train, X_test, y_train, y_test, label_encoder
    )

    # Random Forest (bonus algorithm)
    rf_results = perform_random_forest_classification(
        X_train, X_test, y_train, y_test, label_encoder
    )

    # Combine results
    all_results = [knn_results, nb_results, rf_results]

    # Visualize results
    visualize_classification_results(all_results, X_test, y_test, label_encoder)

    # Analyze errors
    analyze_classification_errors(all_results, X_test, y_test, label_encoder)

    # Compare algorithms
    comparison_table = compare_classification_algorithms(all_results)

    # Feature importance analysis
    feature_importance_analysis(all_results, feature_names)

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

    results_summary.to_csv("classification_results_summary.csv", index=False)
    print("Classification results saved to 'classification_results_summary.csv'")

    print("\n=== CLASSIFICATION ANALYSIS COMPLETE ===")
    return all_results, comparison_table


if __name__ == "__main__":
    classification_results, comparison = main_classification()
