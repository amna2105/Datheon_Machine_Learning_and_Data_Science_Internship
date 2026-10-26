# Iris Flower Classification: Logistic Regression vs Decision Tree

A supervised machine-learning project that classifies iris flowers into three species
(*setosa*, *versicolor*, *virginica*) from four measurements, and benchmarks two classifiers
using accuracy, precision, recall and F1-score.

## Objectives

- Load and explore the Iris dataset with Pandas
- Visualise feature relationships with a Seaborn pairplot
- Scale features with `StandardScaler`
- Split data 80/20 (stratified) into train and test sets
- Train `LogisticRegression` and `DecisionTreeClassifier`
- Evaluate with accuracy, precision, recall, F1-score and 5-fold cross-validation
- Plot confusion-matrix heatmaps and decision boundaries

## Dataset

The dataset is **not included** in this repository (it is listed in `.gitignore`).
Download it into the project folder before running the notebook:

```bash
curl -o iris.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
```

150 samples, 4 numeric features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`)
and a `species` label (50 samples per class).

## Project Structure

```
.
├── iris.ipynb      # full analysis notebook
├── iris.csv        # dataset (download locally, git-ignored)
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook iris.ipynb
```

## Methodology

1. **Load and explore**: `head`, `info`, `shape`, and a Seaborn pairplot coloured by species.
2. **Preprocess**: encode the target with `LabelEncoder`, then split 80/20 with
   `stratify=y` and `random_state=42`.
3. **Scale**: `StandardScaler` is fit on the training set only and then applied to the test set
   to avoid data leakage.
4. **Train**: Logistic Regression (`max_iter=200`) and Decision Tree (`max_depth=3`).
5. **Evaluate**: benchmark table (accuracy, precision, recall, F1), per-class
   `classification_report`, and stratified 5-fold cross-validation using a `Pipeline`.
6. **Visualise**: confusion-matrix heatmaps and 2-D decision-boundary plots (petal length vs petal width).

## Results

Held-out test set (30 samples):

| Model               | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---------------------|----------|-----------------|--------------|----------|
| Logistic Regression | 0.933    | 0.933           | 0.933        | 0.933    |
| Decision Tree       | 0.967    | 0.970           | 0.967        | 0.967    |S

5-fold cross-validation accuracy (full dataset, scaler re-fit inside each fold): **95.3%** for Logistic Regression and **96.0%** for Decision Tree.

> The test set has only 30 samples, so one misclassified flower shifts accuracy by about 3.3%.
> Cross-validation gives a more reliable comparison, and on it the two models are very close.

## Key Takeaways

- *Setosa* is perfectly separable; the few errors occur between *versicolor* and *virginica*.
- Both models perform similarly on this small, clean dataset.
- Petal measurements separate the classes far better than sepal measurements.

## Tech Stack

Python · Pandas · NumPy · Scikit-Learn · Matplotlib · Seaborn

## Possible Extensions

- Try the Wine dataset or other classifiers (KNN, SVM, Random Forest)
- Tune hyperparameters with `GridSearchCV`
- Wrap scaling and model in a `Pipeline`