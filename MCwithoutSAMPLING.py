# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 20:04:54 2025

@author: 86198
"""


# -*- coding: utf-8 -*-
"""

Model construction process without data sampling
@author: Li Na == Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

# Set working path
work_path = 'your path'
os.chdir(work_path)

def load_data(file_path):
    """
    Load data from an Excel file.
    :param file_path: Path to the Excel file.
    :return: Features (X) and target (y).
    """
    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    return X, y

def get_results(X, y):
    """
    Train and evaluate models without any oversampling.
    :param X: Features.
    :param y: Target.
    :return: Results DataFrame and probability predictions.
    """
    # Define machine learning algorithms
    ml_algorithms = {
        "Logistic Regression": LogisticRegression(random_state=100, n_jobs=-1),
        "Random Forest": RandomForestClassifier(random_state=100, n_jobs=-1),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(random_state=100, n_jobs=-1),
        "SVM": SVC(random_state=100, probability=True),
        "Decision Trees": DecisionTreeClassifier(random_state=100),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(random_state=100),
    }
    
    # Initialize result and probability lists
    results = []
    prob_list = []
    
    # Define cross-validation strategy
    kf = RepeatedStratifiedKFold(n_splits=5, random_state=100, n_repeats=100)
    
    # Iterate through folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train and evaluate each machine learning algorithm
        for ml_algorithm_name, ml_algorithm in ml_algorithms.items():
            ml_algorithm.fit(X_train, y_train)
            y_pred = ml_algorithm.predict(X_test)
            y_prob = ml_algorithm.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate evaluation metrics
            auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division="warn")
            recall = recall_score(y_test, y_pred)
            f_score = f1_score(y_test, y_pred)

            # Append results
            results.append({
                "Fold": fold,
                "ML Algorithm": ml_algorithm_name,
                "AUC": auc,
                "ACC": acc,
                "Precision": precision,
                "Recall": recall,
                "F-score": f_score
            })
            
            # Append probability predictions
            for pred, resp in zip(y_prob, y_test):
                prob_list.append({
                    "Fold": fold,
                    "ML Algorithm": ml_algorithm_name,
                    "pred": pred,
                    "response": resp
                })
    
    # Convert results to DataFrames
    results = pd.DataFrame(results)
    prob = pd.DataFrame(prob_list)
    return results, prob

def calculating_CI(results):
    """
    Calculate 95% confidence intervals for evaluation metrics.
    :param results: DataFrame containing evaluation results.
    :return: DataFrame with confidence intervals.
    """
    CI_df = []
    algorithms = results["ML Algorithm"].unique()  # Unique machine learning algorithms
    
    for algorithm in algorithms:
        algorithm_group = results[results["ML Algorithm"] == algorithm]
        selected_columns = algorithm_group.iloc[:, 2:7]  # Select evaluation metrics
        
        for selected_column in selected_columns.columns:
            # Calculate mean and standard error
            mean = algorithm_group[selected_column].mean()
            std_error = algorithm_group[selected_column].sem()
            
            # Calculate 95% confidence interval
            df = len(algorithm_group[selected_column]) - 1  # Degrees of freedom
            confidence_interval = stats.t.interval(0.95, df, loc=mean, scale=std_error)
            
            # Append results
            CI_df.append({
                "ML Algorithm": algorithm,
                "Index": selected_column,
                "Mean": mean,
                "Standard Error": std_error,
                "Confidence Interval Lower": confidence_interval[0],
                "Confidence Interval Upper": confidence_interval[1]
            })
    
    # Convert results to DataFrame
    CI_df = pd.DataFrame(CI_df)
    return CI_df

# Main function
# origin without pruning
if __name__ == "__main__":
    file_path = "fdata.xlsx"
    results, prob = get_results(file_path)
    results.to_csv("results_kfold_origin_nonsampling.csv", index=True)
    prob.to_csv("prob_kfold_origin_nonsampling.csv", index=True)
    CI_df = calculating_CI(results)
    CI_df.to_csv("CI_kfold_origin_nonsampling.csv", index=True)
    
# Lasso
if __name__ == "__main__":
    file_path = "Lasso.xlsx"
    results, prob = get_results(file_path)
    results.to_csv("results_kfold_Lasso_nonsampling.csv", index=True)
    prob.to_csv("prob_kfold_Lasso_nonsampling.csv", index=True)
    CI_df = calculating_CI(results)
    CI_df.to_csv("CI_kfold_Lasso_nonsampling.csv", index=True)
    
# Boruta
if __name__ == "__main__":
    file_path = "Boruta.xlsx"
    results, prob = get_results(file_path)
    results.to_csv("results_kfold_Boruta_nonsampling.csv", index=True)
    prob.to_csv("prob_kfold_Boruta_nonsampling.csv", index=True)
    CI_df = calculating_CI(results)
    CI_df.to_csv("CI_kfold_Boruta_nonsampling.csv", index=True)

# Boruta&Lasso
if __name__ == "__main__":
    file_path = "Boruta&Lasso.xlsx"
    results, prob = get_results(file_path)
    results.to_csv("results_kfold_Boruta&Lasso_nonsampling.csv", index=True)
    prob.to_csv("prob_kfold_Boruta&Lasso_nonsampling.csv", index=True)
    CI_df = calculating_CI(results)
    CI_df.to_csv("CI_kfold_Boruta&Lasso_nonsampling.csv", index=True)