# -*- coding: utf-8 -*-
"""


这部分是使用Boruta算法进行特征选择的代码
Feature selection process by Boruta
@author: Li Na == Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
# Read Excel data
mydata = pd.read_excel('fdata.xlsx')

# Split data into features and target variable
x = mydata.loc[:, 'DrugusedAround_Yes':'ACEs']
y = mydata['Drugused']

# Initializes the random forest classifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=100)

# Initializes Boruta
boruta_feature_selector = BorutaPy(rf, n_estimators='auto', random_state=100)

# Fit Boruta
boruta_feature_selector.fit(x.values, y.values)

# Gets the selected feature
selected_features = boruta_feature_selector.support_

# Print the selected feature
print("Selected Features: ", x.columns[selected_features])

# check ranking of features
print(boruta_feature_selector.ranking_)