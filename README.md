# Credit Risk Prediction Early-Warning System

A machine-learning benchmarking project for evaluating credit-risk early-warning models with a business-focused metric: **trigger precision at recall >= 0.65**.

The project compares multiple classification algorithms using LendingClub consumer-loan data and evaluates how effectively each model identifies high-risk cases for further review.

## Business Problem

Credit-risk teams need to identify likely defaults early without overwhelming analysts with false alerts. Standard accuracy alone is not sufficient for this use case.

This project evaluates models using:

- **Recall:** proportion of risky cases identified
- **Precision:** proportion of alerts that are relevant
- **Trigger precision at recall >= 0.65:** alert quality while maintaining a minimum detection rate
- **ROC-AUC:** overall classification performance
- **F1 score:** balance between precision and recall
- **Matthews correlation coefficient (MCC):** robust evaluation for imbalanced data

## Dataset

The notebook is designed for LendingClub loan data.

- **Records loaded:** 2,260,701
- **Original features:** 151
- **Domain:** consumer lending
- **Target:** loan-default risk

The dataset is not included in this repository due to its size. Update the dataset path in the notebook before running it.

## Models Benchmarked

The notebook defines eight classification models:

1. Linear Discriminant Analysis
2. Logistic Regression
3. Decision Tree
4. Support Vector Machine
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. Neural Network

Hyperparameter tuning is performed with `GridSearchCV`, using ROC-AUC as the scoring metric.

## Research Context

This implementation extends the evaluation approach used in a published credit-risk study based on commercial banking data.

The project focuses on a practical question:

> Which model produces the most useful early-warning alerts while maintaining recall of at least 0.65?

The notebook compares model performance against a reference trigger-precision range of `0.12-0.15`.

## Workflow

1. Load and preprocess LendingClub loan data
2. Split data into training and test sets
3. Scale model features
4. Tune model hyperparameters
5. Evaluate classification metrics
6. Compare trigger precision at the target recall threshold
7. Visualize model rankings and precision-recall trade-offs
8. Save the best-performing model and benchmark results

## Visualizations

The notebook generates:

- ROC-AUC model comparison
- Precision-recall trade-off chart
- Trigger-precision ranking
- Comparison against the reference study

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab
- Joblib

## Repository Structure

```text
  .
  ├── Credit-Risk-Assessment-Machine-Learning-Paper.ipynb
  └── README.md

## Getting Started

Clone the repository:

git clone https://github.com/ambika1us/Credit-Risk-Assessment-Machine-Learning-Paper.git
cd Credit-Risk-Assessment-Machine-Learning-Paper

Install the required packages:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib jupyter

Open the notebook:

jupyter notebook Credit-Risk-Assessment-Machine-Learning-Paper.ipynb

Update the dataset path before running the notebook:

df = pd.read_csv("path/to/loan_data.csv", low_memory=False)

## Current Status

The repository contains the model-benchmarking and evaluation workflow. The public notebook still needs the preprocessing steps and executed benchmark outputs before it can be reproduced end to end.

Planned improvements:

- Add preprocessing and feature-selection steps
- Add final benchmark results table
- Add generated charts
- Add a sample configuration for local execution
- Add the research-paper link when publicly available

## Author

Ambika Prasad Rath

- LinkedIn (https://www.linkedin.com/in/ambika-prasad-rath/)
- GitHub (https://github.com/ambika1us)
