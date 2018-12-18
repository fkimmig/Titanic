# -*- coding : utf-8 -*-


### IMPORTATION PYTHON
# General management
import os
import sys
# Data analysis ans wrangling
import numpy as np
import math
import pandas as pd
import random as random
# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
#
# # machine learning
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier

# --- read data ser
path = os.path.dirname(os.path.realpath(__file__))
data_raw = pd.read_csv(path+"/train.csv")

# --- clean data set
data_clean_age = data_raw.dropna(subset=['Age']) # remove lines with missing age


# --- survival rate with Pclass
p_class = data_raw.groupby('Pclass').mean()
p_class['Count'] = data_raw['Pclass'].value_counts()

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))
sns.countplot(x='Pclass', data=data_raw, order=['1','2','3'], ax=axis1)
sns.barplot(x=p_class.index, y='Survived', data=p_class, order=['1','2','3'], ax=axis2)

# --- survival rate with age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
age = data_raw[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=age)

# --- survival rate with fare
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
fare = data_raw[["Fare", "Survived"]].groupby(['Fare'],as_index=False).mean()
sns.barplot(x='Fare', y='Survived', data=fare)


plt.show()

# --- TODO
# survival rate with gender (compare with embarked gender distribution)
# survival rate whether child or not
# survival rate with "fare category"




