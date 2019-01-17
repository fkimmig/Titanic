# -*- coding : utf-8 -*-

### IMPORTATION PYTHON
# General management
import os
import sys
# Data analysis ans wrangling
import numpy as np
import pandas as pd
import random as random
# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

################################################################################
################################################################################
## ACQUIRE DATA
path = os.path.dirname(os.path.realpath(__file__))
train_df = pd.read_csv(path+"/train.csv")
test_df = pd.read_csv(path+"/test.csv")
## REMOVE ROWS
train_df= train_df[train_df["Age"]>0]
train_df.dropna(subset=["Age"])
test_df= test_df[test_df["Age"]>0]
test_df.dropna(subset=["Age"])
## COMBINE FOR TREATMENT
combine = [train_df, test_df]

################################################################################
################################################################################
## Siblings
# Transform Sibs to binary
for dataset in combine:
    temp_df = map(lambda x: x if (x==0) else 1, dataset["SibSp"])
    dataset["SibSp"]=list(temp_df)
## Parents-children
# Add an information - Adult/Children
for dataset in combine :
    temp_df = map(lambda x: 0 if (x<16) else 1, dataset["Age"])
    dataset["Adultw"] = list(temp_df)
# Transform Parch to
# 0 - adult, no parch
# 1 - is a parent and has a children on board
# 2 - is a children and has a parent on board
# 3 - is a children, no parch
for dataset in combine:
    dataset.loc[(dataset['Adultw'] ==1)&(dataset['Parch'] == 0), 'Parch'] = 0
    dataset.loc[(dataset['Adultw'] ==1)&(dataset['Parch'] >0), 'Parch'] = 1
    dataset.loc[(dataset['Adultw'] ==0)&(dataset['Parch'] >0), 'Parch'] = 2
    dataset.loc[(dataset['Adultw'] ==0)&(dataset['Parch'] == 0), 'Parch'] = 3

##################################################################################
### MODEL - PREDICT - SOLVE
# Preparing data
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

# Lorgistic regression
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print(acc_log)
