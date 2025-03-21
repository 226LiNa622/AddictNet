# -*- coding: utf-8 -*-
"""


这部分是对数据进行预处理的代码
Data preprocessing with one-hot encoding and normalization
@author: Li Na == Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

import pandas as pd
#import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
#Read data
data = pd.read_excel('after_missforest.xlsx')
print(data.iloc[:, -10:].columns)
#The variable of ACEs were processed
Abuse = data[['Emotional_abuse', 'Physical_abuse', 'Sexual_abuse']].sum(axis=1)
Neglect = data[['Emotional_neglect', 'Physical_neglect']].sum(axis=1)
Household_dysfunction = data[['Violence_against_mother', 'Parental_separatio0Rdivorce',
                              'Household_substance_abuse',	'Household_mental_illness',
                              'Household_incarceration']].sum(axis=1)
print(Abuse)
ACEs = data.iloc[:, -10:].sum(axis=1)
data['Abuse'] = Abuse
data['Neglect'] = Neglect
data['Household_dysfunction'] = Household_dysfunction
data['ACEs'] = ACEs
# Dummy variables are encoded and reference categories are set
x_cator_multi = data.loc[:, 'Education':'CurrentResidentCitySize']  # Extract multiple categorical variables
x_encoded = pd.get_dummies(x_cator_multi)
X_e = x_encoded.drop(columns=['Education_HighSchool', 'SocialRole_Students',
                              'ParentsOccupation_NoneStable',
                              'FamilyIncomeMonthly_3000below',
                              'FamilyResidentCitySize_Large',
                              'CurrentResidentCitySize_Large'])
newdata = pd.concat([data.loc[:, 'DrugusedAround_Yes': 'Gender_Male'], X_e,
                  data.loc[:, 'TradDrug_Recognized': 'Hedonic_Hunger'], 
                  data['ACEs'], data['Drugused']], axis=1)
# Normalize 
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler.fit(newdata)
fdata = pd.DataFrame(scaler.transform(newdata), columns=newdata.columns)
fdata.to_excel("fdata.xlsx", index=False)
fdata = pd.read_excel('fdata.xlsx')