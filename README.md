# Credit Card Fraud Detection using KNN

A machine learning project implementing K-Nearest Neighbors (KNN) algorithm for detecting fraudulent credit card transactions.

## 📋 Project Overview

This project uses the K-Nearest Neighbors (KNN) algorithm to classify credit card transactions as either legitimate or fraudulent. The dataset contains anonymized credit card transactions with features derived from PCA transformation.

## 📊 Dataset

- **File**: `data/creditcard.csv`
- **Total Transactions**: Records of credit card transactions
- **Features**: 30 (28 PCA-transformed features + Time + Amount)
- **Target**: Class (0 = Normal, 1 = Fraud)
- **Class Distribution**: Highly imbalanced dataset (majority legitimate transactions)

### Features:
- **V1 to V28**: PCA-transformed features
- **Time**: Seconds elapsed between transaction and first transaction
- **Amount**: Transaction amount in dollars
- **Class**: Target variable (0 or 1)

## 🎯 Algorithm: K-Nearest Neighbors (KNN)

### How KNN Works:
1. **Calculate Distance**: Computes distance (Euclidean, Manhattan, etc.) from test point to all training points
2. **Find K Neighbors**: Identifies the K nearest neighbors
3. **Majority Vote**: Assigns class based on majority vote among neighbors
4. **Predict**: Returns the predicted class

### Advantages:
- ✅ Simple and intuitive
- ✅ No training phase needed
- ✅ Works well with non-linear data
- ✅ Effective for binary classification

### Disadvantages:
- ❌ Computationally expensive for large datasets
- ❌ Sensitive to feature scaling
- ❌ Requires optimal K selection

## 🚀 Project Structure

```
credit-card-fraud-detection-knn/
├── data/
│   └── creditcard.csv          # Dataset
├── results/
│   ├── confusion_matrix.png    # Confusion matrix plot
│   ├── roc_curve.png           # ROC curve plot
│   └── k_optimization.png      # K value optimization plot
├── main.py                      # Main script
├── data_exploration.py          # Data analysis script
├── predict.py                   # Prediction module
├── requirements.txt             # Project dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore file
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/sgsinghashka-del/credit-card-fraud-detection-knn.git
cd credit-card-fraud-detection-knn
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🏃 Usage

### Run the main model
```bash
python main.py
```

### Run data exploration
```bash
python data_exploration.py
```

### Run prediction on new data
```bash
python predict.py
```

## 📈 Model Performance

The model evaluates performance using multiple metrics:

### Metrics Used:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of correct fraud predictions to all fraud predictions
- **Recall**: Ratio of detected frauds to all actual frauds
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: True Positives, True Negatives, False Positives, False Negatives

### Results Interpretation:
```
Confusion Matrix:
                Predicted Normal    Predicted Fraud
Actual Normal        TN                 FP
Actual Fraud         FN                 TP

- True Positive (TP): Correctly identified fraud
- True Negative (TN): Correctly identified normal
- False Positive (FP): Normal flagged as fraud
- False Negative (FN): Fraud marked as normal (worst case)
```

## 🔧 Hyperparameter Tuning

### Key Parameters:
- **n_neighbors (K)**: Number of nearest neighbors to consider
  - Small K: More flexible, prone to overfitting
  - Large K: More stable, may underfit
  - Optimal: Usually found through cross-validation (typically 3-15)

- **metric**: Distance metric
  - 'euclidean': Standard Euclidean distance
  - 'manhattan': Manhattan distance
  - 'minkowski': Generalized distance

- **weights**: Weight scheme
  - 'uniform': All neighbors weighted equally
  - 'distance': Closer neighbors weighted more

## 📊 Data Processing Pipeline

1. **Load Data**: Read CSV file
2. **Handle Missing Values**: Fill with mean values
3. **Separate Features & Target**: X and y split
4. **Train-Test Split**: 80-20 split with stratification
5. **Feature Scaling**: StandardScaler normalization
6. **Model Training**: Fit KNN classifier
7. **Model Evaluation**: Generate metrics and visualizations

## 🔍 Feature Scaling Importance

KNN is distance-based, so feature scaling is **crucial**:
- **Without scaling**: Features with larger ranges dominate distance calculation
- **With scaling**: All features contribute equally
- **StandardScaler**: Centers data at 0 with unit variance

## 📉 Class Imbalance Handling

Since fraud is rare, the dataset is imbalanced:

**Techniques Used:**
- Stratified train-test split (preserves class ratio)
- Focus on metrics beyond accuracy (precision, recall, F1-score)
- ROC-AUC score for comprehensive evaluation

**Additional approaches (optional):**
- SMOTE (Synthetic Minority Over-sampling)
- Class weights adjustment
- Threshold tuning on probability predictions

## 🧪 Testing & Validation

```python
# Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv=5)

# ROC-AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred_proba)
```

## 💡 Tips for Improvement

1. **Increase K**: May improve generalization
2. **Feature Engineering**: Create new relevant features
3. **Handle Imbalance**: Use SMOTE or class weights
4. **Ensemble Methods**: Combine with other algorithms
5. **Hyperparameter Grid Search**: Use GridSearchCV for optimization

## 📚 References

- [Scikit-learn KNeighborsClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [KNN Algorithm Explanation](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Feature Scaling in ML](https://scikit-learn.org/stable/modules/preprocessing.html)

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ❓ FAQ

**Q: Why use KNN for fraud detection?**
A: KNN works well for fraud detection because fraudulent transactions are often locally clustered - similar to other frauds. It's simple and interpretable.

**Q: How do I handle class imbalance?**
A: Use stratified splitting, adjust sample weights, or use SMOTE for synthetic oversampling.

**Q: What's the best K value?**
A: Run the k_optimization to find the optimal K. Typically between 3-15 for this dataset.

**Q: Can I use this in production?**
A: Yes, but consider the preprocessing pipeline, model persistence, and real-time latency requirements.

---

**Author**: Ashka Singh  
**Status**: Active Development
