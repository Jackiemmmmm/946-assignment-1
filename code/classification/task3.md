# Task 3: Classification Analysis Report

## Data Preprocessing for Classification

### 1. Feature Selection and Target Variable

**Features Used**:

- **current_price**: Product pricing (transformed with sqrt)
- **discount**: Discount percentage offered  
- **likes_count**: Customer engagement metric (transformed with log1p)
- **is_new**: Boolean indicator converted to numeric (0/1)

**Target Variable**:

- **category**: 9 product categories (accessories, bags, beauty, house, jewelry, kids, men, shoes, women)

**Rationale**: These features capture pricing strategy, customer appeal, promotional approach, and product lifecycle - key determinants for product categorization.

### 2. Data Preprocessing Steps

1. **Feature Transformation**: Applied same transformations as clustering
   - log₁₊(likes_count) to handle extreme skewness
   - √(current_price) to reduce moderate skewness
   - Standard scaling for all features

2. **Label Encoding**: Converted categorical target to numerical labels
   - 9 categories encoded as integers 0-8
   - Preserved original category names for interpretation

3. **Data Splitting**: 
   - **Training Set**: 80% (stratified sampling)
   - **Test Set**: 20% (stratified sampling)
   - **Stratification**: Ensured balanced representation across categories

## Classification Algorithm Implementation

### Algorithm 1: K-Nearest Neighbors (KNN)

#### Detailed Steps:

1. **Hyperparameter Optimization**:
   - Tested k values from 1 to 20
   - Used 5-fold cross-validation for evaluation
   - Selected optimal k based on highest CV accuracy

2. **Parameter Selection**:
   - **n_neighbors**: 11 (optimal k from grid search)
   - **weights**: uniform (equal weight for all neighbors)
   - **metric**: euclidean (standard distance measure)
   - **algorithm**: auto (automatic algorithm selection)

3. **Algorithm Process**:
   - For each test sample, find k=11 nearest training samples
   - Use majority voting among neighbors for classification
   - Handle ties using nearest neighbor's class

#### KNN Results:

- **Optimal k**: 11
- **Cross-validation Score**: 0.454 ± 0.018
- **Test Accuracy**: 0.455
- **Training Time**: ~2.3 seconds
- **Prediction Time**: ~8.7 seconds

### Algorithm 2: Naive Bayes (Gaussian)

#### Detailed Steps:

1. **Algorithm Assumptions**:
   - Features follow Gaussian (normal) distribution
   - Features are conditionally independent given the class
   - Equal class priors (adjusted based on training data)

2. **Parameter Selection**:
   - **var_smoothing**: 1e-9 (default Laplace smoothing)
   - **priors**: None (estimated from training data)

3. **Algorithm Process**:
   - Calculate mean and variance for each feature per class
   - Apply Gaussian probability density function
   - Use Bayes' theorem for posterior probability calculation
   - Classify based on maximum a posteriori (MAP) estimation

#### Naive Bayes Results:

- **Cross-validation Score**: 0.303 ± 0.015
- **Test Accuracy**: 0.302
- **Training Time**: ~0.1 seconds
- **Prediction Time**: ~0.3 seconds

### Algorithm 3: Random Forest (Ensemble Method)

#### Detailed Steps:

1. **Hyperparameter Optimization**:
   - **Grid Search Parameters**:
     - n_estimators: [50, 100, 200]
     - max_depth: [3, 5, 7, None]
     - min_samples_split: [2, 5, 10]
   - Used 3-fold cross-validation for efficiency

2. **Optimal Parameters**:
   - **n_estimators**: 200 (number of trees)
   - **max_depth**: 7 (maximum tree depth)
   - **min_samples_split**: 2 (minimum samples to split)
   - **random_state**: 42 (reproducibility)

3. **Algorithm Process**:
   - Build 200 decision trees with random feature subsets
   - Each tree trained on bootstrap sample of training data
   - Aggregate predictions through majority voting
   - Calculate feature importance through tree splits

#### Random Forest Results:

- **Best Parameters**: {n_estimators: 200, max_depth: 7, min_samples_split: 2}
- **Cross-validation Score**: 0.545 ± 0.012
- **Test Accuracy**: 0.548
- **Training Time**: ~45.2 seconds
- **Prediction Time**: ~1.8 seconds

## Algorithm Results Comparison

| Algorithm               | CV Score  | Test Accuracy | Training Time | Prediction Time | Hyperparameters                                    |
| ----------------------- | --------- | ------------- | ------------- | --------------- | -------------------------------------------------- |
| **Random Forest**       | **0.545** | **0.548**     | 45.2s         | 1.8s            | n_estimators=200, max_depth=7, min_samples_split=2 |
| **K-Nearest Neighbors** | 0.454     | 0.455         | 2.3s          | 8.7s            | n_neighbors=11, weights=uniform, metric=euclidean  |
| **Naive Bayes**         | 0.303     | 0.302         | 0.1s          | 0.3s            | var_smoothing=1e-9, priors=None                    |

## Detailed Performance Analysis

### Per-Class Performance (Random Forest - Best Algorithm)

| Category    | Precision | Recall | F1-Score | Support |
| ----------- | --------- | ------ | -------- | ------- |
| accessories | 0.42      | 0.48   | 0.45     | 1,172   |
| bags        | 0.58      | 0.52   | 0.55     | 1,021   |
| beauty      | 0.51      | 0.47   | 0.49     | 611     |
| house       | 0.59      | 0.61   | 0.60     | 2,102   |
| jewelry     | 0.43      | 0.42   | 0.43     | 813     |
| kids        | 0.48      | 0.42   | 0.45     | 774     |
| men         | 0.58      | 0.68   | 0.62     | 1,667   |
| shoes       | 0.61      | 0.60   | 0.61     | 1,633   |
| women       | 0.69      | 0.69   | 0.69     | 2,449   |

### Feature Importance Analysis (Random Forest)

| Feature           | Importance Score | Business Interpretation                                                  |
| ----------------- | ---------------- | ------------------------------------------------------------------------ |
| **current_price** | 0.478            | Primary differentiator - different categories have distinct price ranges |
| **discount**      | 0.267            | Secondary factor - categories use different promotional strategies       |
| **likes_count**   | 0.245            | Customer preference varies significantly across categories               |
| **is_new**        | 0.010            | Minimal impact - new product distribution similar across categories      |

## Algorithm Comparison and Analysis

### Strengths and Weaknesses

#### Random Forest

**Strengths**:

- **Best Overall Performance**: Highest accuracy (54.8%)
- **Robust to Overfitting**: Good generalization through ensemble
- **Feature Importance**: Provides interpretable feature rankings
- **Handles Non-linearity**: Captures complex feature interactions
- **Class Imbalance Tolerance**: Better handling of uneven class distribution

**Weaknesses**:

- **Computational Cost**: Longest training time (45.2s)
- **Model Complexity**: Less interpretable than simple algorithms
- **Memory Usage**: Requires storage for 200 decision trees

#### K-Nearest Neighbors

**Strengths**:

- **Simple and Intuitive**: Easy to understand and implement
- **Non-parametric**: No assumptions about data distribution
- **Decent Performance**: Moderate accuracy (45.5%)
- **Fast Training**: Quick model building (2.3s)

**Weaknesses**:

- **Slow Prediction**: Longest prediction time (8.7s)
- **Curse of Dimensionality**: Performance degrades with more features
- **Sensitive to Scale**: Requires careful feature normalization
- **Memory Intensive**: Stores entire training dataset

#### Naive Bayes

**Strengths**:

- **Extremely Fast**: Fastest training (0.1s) and prediction (0.3s)
- **Low Memory**: Minimal storage requirements
- **Probabilistic Output**: Provides class probabilities
- **Scalable**: Handles large datasets efficiently

**Weaknesses**:

- **Poor Performance**: Lowest accuracy (30.2%)
- **Independence Assumption**: Unrealistic for this dataset
- **Class Imbalance Issues**: Biased toward majority classes
- **Limited Complexity**: Cannot capture feature interactions

### Performance Evaluation

#### Model Reliability:

- **Random Forest**: CV-Test consistency (0.545 vs 0.548) indicates stable model
- **KNN**: Good consistency (0.454 vs 0.455) shows no overfitting
- **Naive Bayes**: Perfect consistency (0.303 vs 0.302) but poor overall performance

#### Computational Efficiency:

- **Training**: Naive Bayes > KNN > Random Forest
- **Prediction**: Naive Bayes > Random Forest > KNN
- **Overall**: Random Forest offers best accuracy-speed tradeoff for deployment

## Best Result Determination

### Winner: **Random Forest**

**Quantitative Justification**:

1. **Highest Accuracy**: 54.8% test accuracy (20.8% better than KNN, 81.5% better than Naive Bayes)
2. **Best Cross-validation**: 0.545 CV score with low variance (±0.012)
3. **Balanced Performance**: Good precision and recall across most categories
4. **Reasonable Speed**: Acceptable prediction time for production use

**Qualitative Justification**:

1. **Business Applicability**: Feature importance insights valuable for strategy
2. **Scalability**: Can handle larger datasets and more features
3. **Robustness**: Less sensitive to outliers and noise
4. **Interpretability**: Feature importance provides actionable insights

### Class-Specific Insights:

- **Best Predicted**: women (69% F1-score), shoes (61% F1-score)
- **Challenging**: accessories (45% F1-score), kids (45% F1-score)
- **Improvement Needed**: jewelry (43% F1-score) requires feature engineering

## Business Implications and Recommendations

### 1. Pricing Strategy Optimization

- **Key Finding**: Price is the most important predictor (47.8% importance)
- **Recommendation**: Develop category-specific pricing models
- **Action**: Analyze price distributions per category for competitive positioning

### 2. Promotional Strategy Enhancement  

- **Key Finding**: Discount strategy varies significantly by category (26.7% importance)
- **Recommendation**: Customize discount policies per product category
- **Action**: A/B test category-specific promotional campaigns

### 3. Customer Engagement Focus

- **Key Finding**: Likes count shows category-dependent patterns (24.5% importance)
- **Recommendation**: Develop category-specific engagement strategies
- **Action**: Analyze high-performing products within each category

### 4. Product Classification System

- **Application**: Implement Random Forest for automated product categorization
- **Accuracy**: Expect ~55% automatic classification accuracy
- **Human Review**: Focus manual review on low-confidence predictions

### 5. Inventory Management

- **High-Confidence Categories**: women, shoes (>60% accuracy)
- **Review-Required Categories**: accessories, jewelry, kids (<50% accuracy)
- **Strategy**: Automated routing for high-confidence, manual review for others

## Model Deployment Considerations

### Production Readiness:

- **Model**: Random Forest with identified optimal parameters
- **Preprocessing**: Implement same transformation pipeline
- **Monitoring**: Track prediction confidence and accuracy over time
- **Updating**: Retrain quarterly with new product data

### Performance Benchmarks:

- **Target Accuracy**: Maintain >50% classification accuracy
- **Speed Requirement**: <2 seconds prediction time
- **Confidence Threshold**: Flag predictions with <60% probability for review

This comprehensive analysis demonstrates the successful implementation of multiple classification algorithms with clear performance metrics and actionable business insights.
