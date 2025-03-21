# -*- coding: utf-8 -*-
"""

Model construction process by voting model
@author: Li Na == Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, RandomOverSampler, ADASYN, SMOTENC
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
from imblearn.over_sampling import ADASYN

# Set working directory
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

def get_results(file_path):
    """
    Train and evaluate a VotingClassifier model using ADASYN for oversampling.
    :param file_path: Path to the dataset file.
    :return: Results DataFrame and probability predictions.
    """
    X, y = load_data(file_path)
    
    # Define base machine learning models
    model_lr = LogisticRegression(random_state=100)
    model_knn = KNeighborsClassifier()
    model_svc = SVC(random_state=100, probability=True)
    
    # Combine models into a VotingClassifier
    estimators = [('LR', model_lr), ('KNN', model_knn), ('SVC', model_svc)]
    model_vot = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1, 1], n_jobs=-1)
    
    # Initialize result and probability lists
    results = []
    prob_list = []
    
    # Define cross-validation strategy
    kf = RepeatedStratifiedKFold(n_splits=5, random_state=100, n_repeats=100)
    
    # Iterate through different folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Apply ADASYN oversampling
        sampling = ADASYN(random_state=100)
        X_train_sampling, y_train_sampling = sampling.fit_resample(X_train, y_train)
        
        # Train the VotingClassifier (with or without oversampling)
        model_vot.fit(X_train_sampling, y_train_sampling)  # Use oversampled data
        # model_vot.fit(X_train, y_train)  # Use original data (no oversampling)
        
        # Make predictions
        y_pred = model_vot.predict(X_test)
        y_prob = model_vot.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate evaluation metrics
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division="warn")
        recall = recall_score(y_test, y_pred)
        f_score = f1_score(y_test, y_pred)

        # Append results
        results.append({
            "Fold": fold,
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
    CI_df = []  # Initialize an empty list to store results
    
    # Select evaluation metrics columns
    selected_columns = results.iloc[:, 1:6]
    
    # Calculate mean, standard error, and confidence intervals for each metric
    for selected_column in selected_columns.columns:
        mean = results[selected_column].mean()
        std_error = results[selected_column].sem()
        df = len(results[selected_column]) - 1  # Degrees of freedom
        confidence_interval = stats.t.interval(0.95, df, loc=mean, scale=std_error)
        
        # Append results
        CI_df.append({
            'Index': selected_column,
            'Mean': mean,
            'Standard Error': std_error,
            'Confidence Interval Lower': confidence_interval[0],
            'Confidence Interval Upper': confidence_interval[1]
        })
    
    # Convert results to DataFrame
    CI_df = pd.DataFrame(CI_df)
    return CI_df

## Main function
# origin without pruning
if __name__ == "__main__":
    file_path = "fdata.xlsx"
    results, prob = get_results(file_path)
    results.to_csv("results_vot_kfold_origin_nonsampling.csv", index=True)
    prob.to_csv("prob_vot_kfold_origin_nonsampling.csv", index=True)
    CI_df = calculating_CI(results)
    CI_df.to_csv("vot_CI_vot_kfold_origin_nonsampling.csv", index=True)
    
# Lasso
if __name__ == "__main__":
    file_path = "Lasso.xlsx"
    results, prob = get_results(file_path)
    results.to_csv("results_kfold_Lasso_nonsampling.csv", index=True)
    prob.to_csv("prob_kfold_Lasso_nonsampling.csv", index=True)
    CI_df = calculating_CI(results)
    CI_df.to_csv("vot_CI_kfold_Lasso_nonsampling.csv", index=True)
    
# Boruta
if __name__ == "__main__":
    file_path = "Boruta.xlsx"
    results, prob = get_results(file_path)
    results.to_csv("results_kfold_Boruta_nonsampling.csv", index=True)
    prob.to_csv("prob_kfold_Boruta_nonsampling.csv", index=True)
    CI_df = calculating_CI(results)
    CI_df.to_csv("vot_CI_kfold_Boruta_nonsampling.csv", index=True)

# Boruta&Lasso
if __name__ == "__main__":
    file_path = "Boruta&Lasso.xlsx"
    results, prob = get_results(file_path)
    results.to_csv("results_kfold_Boruta&Lasso_nonsampling.csv", index=True)
    prob.to_csv("prob_kfold_Boruta&Lasso_nonsampling.csv", index=True)
    CI_df = calculating_CI(results)
    CI_df.to_csv("vot_CI_kfold_Boruta&Lasso_nonsampling.csv", index=True)