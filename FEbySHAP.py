# -*- coding: utf-8 -*-
"""

Feature explanation process of voting model with/without ADASYN by SHAP 
@author: Li Na == Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import ADASYN
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Set working directory
work_path = 'your path'
os.chdir(work_path)

# Load data
data = pd.read_excel("Boruta&Lasso.xlsx")
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Define machine learning models
model_lr1 = LogisticRegression(random_state=100)
model_knn = KNeighborsClassifier()
model_svc = SVC(random_state=100, probability=True)

# Combine models into a VotingClassifier
estimators = [('LR', model_lr1), ('KNN', model_knn), ('SVC', model_svc)]
model_vot = VotingClassifier(estimators=estimators, voting='soft', 
                             weights=[0.8769, 0.8906, 0.8916], n_jobs=-1)

# SHAP analysis settings
shap_values_combined = []  # Stores all SHAP values
np.random.seed(0)  # Set random seed for reproducibility
n_repeats = 100  # Number of cross-validation repeats
random_states = np.random.randint(10000, size=n_repeats)  # Random states for CV splits
print("Random states for CV splits:", random_states)

# Add index column to X for tracking samples
X['index'] = range(len(X))

# Flag to control whether to use ADASYN oversampling
use_adasyn = True  # Set to False to disable ADASYN

# Perform SHAP analysis
for random_state in random_states:
    combined_df = pd.DataFrame()  # Store SHAP values for each fold
    
    # Cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        # Split data into training and test sets
        x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        # Apply ADASYN oversampling if enabled
        if use_adasyn:
            sampling = ADASYN(random_state=100)
            x_train_fold, y_train_fold = sampling.fit_resample(x_train_fold.drop(columns=['index']), y_train_fold)
        else:
            x_train_fold = x_train_fold.drop(columns=['index'])
        
        # Train the VotingClassifier
        model_vot.fit(x_train_fold, y_train_fold)
        
        # Initialize SHAP explainer
        explainer = shap.KernelExplainer(model_vot.predict_proba, x_test_fold.drop(columns=['index']))
        shap_values = explainer.shap_values(x_test_fold.drop(columns=['index']))[:, :, 1]  # SHAP values for positive class
        
        # Add SHAP values to the combined DataFrame with index information
        shap_values_with_index = pd.DataFrame(shap_values, columns=x_test_fold.drop(columns=['index']).columns)
        shap_values_with_index['index'] = x_test_fold['index'].reset_index(drop=True)
        combined_df = pd.concat([combined_df, shap_values_with_index], ignore_index=True)
    
    # Reorder SHAP values to match the original data order
    sorted_shap_values = combined_df.sort_values(by='index')
    
    # Append SHAP values for this repeat to the combined list
    shap_values_combined.append(sorted_shap_values.drop(columns=['index']))

# Reshape SHAP values into (number of samples, number of features, number of repeats)
shap_values_combined = np.swapaxes(shap_values_combined, 0, 1)
xgb_average_shap = np.mean(shap_values_combined, axis=1)

# Calculate the mean absolute SHAP value for each feature
mean_shap_values = np.mean(np.absolute(xgb_average_shap), axis=0)
mean_shap_dict = dict(zip(X.drop(columns=['index']).columns, mean_shap_values))

# Store SHAP values of each feature
xgb_shap = []
for feature, mean_shap in mean_shap_dict.items():
    xgb_shap.append({
        'Feature': feature,
        'Mean SHAP Value': mean_shap
    })
xgb_shap_values = pd.DataFrame(xgb_shap)
xgb_shap_values.to_excel("shap_xgb.xlsx", index=False)

# Visualize the average SHAP values
shap.summary_plot(xgb_average_shap, features=X.drop(columns=['index']), 
                  feature_names=X.drop(columns=['index']).columns, show=False)
plt.title("Average SHAP Values (Voting Model)" + (" with ADASYN" if use_adasyn else " without ADASYN"))
plt.show()

# Visualize the range of SHAP values
xgb_shap_range = np.max(shap_values_combined, axis=1) - np.min(shap_values_combined, axis=1)
xgb_range_long_df = pd.DataFrame(xgb_shap_range, columns=X.drop(columns=['index']).columns).melt(var_name='Features', value_name='Values')

# Standardize the range of SHAP values
mean_abs_effects = xgb_range_long_df.groupby(['Features']).mean()
xgb_standardized = xgb_range_long_df.groupby(xgb_range_long_df.Features).transform(lambda x: x / x.mean())
xgb_standardized['Features'] = xgb_range_long_df.Features
xgb_standardized.to_excel("shap_xgb_standardized.xlsx", index=False)

# Plot the standardized range of SHAP values
sns.catplot(data=xgb_standardized, x='Values', y='Features').set(
    xlabel='Scaled Range of SHAP Values', 
    ylabel='Features', 
    title='Scaled Range of SHAP Values (Voting Model)' + (" with ADASYN" if use_adasyn else " without ADASYN")
)
plt.tight_layout()
plt.show()