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
data_training_raw = pd.read_csv(path+"/train.csv")


# --- preview data
data_training_raw.head(5)
data_training_raw.info()

# --- clean data set
data_clean_age = data_training_raw.dropna(subset=['Age']) # remove lines with missing age

# # --- data processing
child_age = 16
def is_child(person):
    age,sex = person
    return 'child' if age < child_age else sex
    
data_training_raw['Person'] = data_training_raw[['Age','Sex']].apply(is_child,axis=1) # axis = 1 : data in column

# --- data visualisation
categories =  [ 'Pclass', 'Sex', 'Fare', 'Age', 'Person']

fig = plt.figure(figsize=(30, 10))
for i in range (0,len(categories)):
    fig.add_subplot(3,3,i+1)
    if categories[i] == 'Age':
        plt.hist(x=categories[i], data=data_training_raw, bins = 20, label='Age')
        plt.xlabel("Age")
    elif categories[i] == 'Fare':
        plt.hist(x=categories[i], data=data_training_raw, bins = 20)
        plt.xlabel("Fare")
    else:
        sns.countplot(x=categories[i], data=data_training_raw)  

i=i+1
fig.add_subplot(3,3,i+1)
sns.countplot(x="Pclass", hue='Sex', data=data_training_raw)

i=i+1
fig.add_subplot(3,3,i+1)
sns.countplot(x="Pclass", hue='Person', data=data_training_raw)

i=i+1
fig.add_subplot(3,3,i+1)
sns.countplot(x="Pclass", hue='Survived', data=data_training_raw)


sns.catplot(x="Pclass", hue="Person", col="Survived",data=data_training_raw, kind="count",height=4, aspect=.7)

sns.factorplot("Pclass", "Survived", "Person", data=data_training_raw, kind="bar", legend=True)

sns.factorplot("Pclass", "Survived", data=data_training_raw, kind="bar", legend=True)

sns.factorplot("Person", "Survived", data=data_training_raw, kind="bar", legend=True)

# i=i+1
# fig.add_subplot(3,3,i+1)
# sns.barplot(x='Person', y='Survived', data=data_training_raw,order=['male','female', 'child'])

plt.show()
fig.clear()

# --- survival rate with Pclass
# p_class = data_training_raw.groupby('Pclass').mean()
# p_class['Count'] = data_training_raw['Pclass'].value_counts()
#
# fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))
# sns.countplot(x='Pclass', data=data_training_raw, order=['1','2','3'], ax=axis1)
# sns.barplot(x=p_class.index, y='Survived', data=p_class, order=['1','2','3'], ax=axis2)

# # --- survival rate with age
# fig, axis1 = plt.subplots(1,1,figsize=(18,4))
# age = data_training_raw[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
# sns.barplot(x='Age', y='Survived', data=age)
#
# # --- survival rate with fare
# fig, axis1 = plt.subplots(1,1,figsize=(18,4))
# fare = data_training_raw[["Fare", "Survived"]].groupby(['Fare'],as_index=False).mean()
# sns.barplot(x='Fare', y='Survived', data=fare)


plt.show()

# --- TODO
# survival rate with gender (compare with embarked gender distribution)
# survival rate whether child or not
# survival rate with "fare category"
# other info like: same price for men and women (by class)
# does "embarked" information play a role ?

# data_training_raw.hist(bins=50, figsize=(20,15))
# plt.show()


