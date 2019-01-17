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
# ACQUIRE DATA
path = os.path.dirname(os.path.realpath(__file__))
train_df = pd.read_csv(path+"/train.csv")
test_df = pd.read_csv(path+"/test.csv")
#print(train_df.columns.values)
################################################################################
################################################################################


# GET GENERAL INFORMATION
## Get counting values and type
#train_df.info()
## Get counting of each label
nb_pass = train_df["PassengerId"].size
nb_survival = train_df["Survived"].sum()
print("La probabilité de survie est {}".format(nb_survival/nb_pass))
## Sum specific label/column
#print(train_df.Parch.sum())
#print(train_df.Parch.min())
#print(train_df.Parch.max())
#print(train_df.Parch.mean())

# GROUPING AND AGGREGATION
#print(train_df.groupby("Sex").mean())
## Describing group infos
#parch_grp = train_df.groupby("Parch")
#print(parch_grp.describe())

# SORTING DATA - Reset Index
#train_df.sort_values(by=["Age"],ascending=False).reset_index()

################################################################################
################################################################################

## REPLACE NAN VALUE
#nb_nan = train_df.isnull().sum().sum()
#nb_nan = train_df.isnull().any()
#print(nb_nan)
## No Nan value for Siblings or Parent/Children
# Replace nan value in age column by median
#age_median = train_df.Age.median()
#print("L'age médian de l'échantillon est : {}".format(age_median))
#values = {"Age":age_median}
#train_df_c = train_df.fillna(value=values)
print("######################################")
## AND CHECK RESULT
#age_1 = sns.FacetGrid(train_df, col='Survived')
#age_1.map(plt.hist, 'Age', bins=20)
#age_c = sns.FacetGrid(train_df_c, col='Survived')
#age_c.map(plt.hist, 'Age', bins=20)
## REMOVE ROWS
train_df= train_df[train_df["Age"]>0]
train_df.dropna(subset=["Age"])
test_df= test_df[test_df["Age"]>0]
test_df.dropna(subset=["Age"])

combine = [train_df, test_df]
################################################################################
################################################################################


# Syblings - level of survival
# Number of siblings / spouses aboard the Titanic
syb_survival = train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False)\
.mean().sort_values(by='Survived', ascending=False)
#print(syb_survival)
survival_syb = train_df[["Survived", "SibSp"]].groupby(['Survived'], as_index=False)\
.mean().sort_values(by='SibSp', ascending=True)
#print(survival_syb)

# Parent Children - level of survival
# Number of parents / children aboard the Titanic
parch_survival = train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False)\
.mean().sort_values(by='Survived', ascending=False)
#print(parch_survival)
survival_parch = train_df[["Survived", "Parch"]].groupby(['Survived'], as_index=False)\
.mean().sort_values(by='Parch', ascending=False)
#print(survival_parch)

### HISTOGRAM - survival according to sibligns
# grid = sns.FacetGrid(train_df, col='Survived', row='SibSp', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=100)
# grid.add_legend()

#Syblings - Parent Children comparaison
cross_sbp = pd.crosstab(train_df["SibSp"],train_df["Parch"])
print(cross_sbp)
print("######################################")

syb_parch = train_df[["SibSp", "Parch"]].groupby(['SibSp'], as_index=False)\
.mean().sort_values(by='Parch', ascending=True)
#print(syb_parch)
#print("######################################")

# Transform Sibs to binary
for dataset in combine:
    temp_df = map(lambda x: x if (x==0) else 1, dataset["SibSp"])
    dataset["SibSp"]=list(temp_df)
cross_sbp = pd.crosstab(train_df["SibSp"],train_df["Parch"])
print(cross_sbp)
print("######################################")

###############################################################################
###############################################################################

# Add an information - Adult/Children
for dataset in combine :
    temp_df = map(lambda x: 0 if (x<16) else 1, dataset["Age"])
    dataset["Adultw"] = list(temp_df)
cross_ap = pd.crosstab(train_df["Adultw"],train_df["Parch"])
print(cross_ap)
print("######################################")

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

# Probabylity of survival P(S|Parch = i)
parch_surv = train_df[["Survived", "Parch"]].groupby(['Parch'], as_index=False)\
.mean().sort_values(by='Survived', ascending=False)
print("PROBABILITE CONDITIONNELLE DE SURVIE")
print(parch_surv)
print("######################################")

### HISTOGRAM - survival according to sibligns
# grid = sns.FacetGrid(train_df, col='Adultw')
grid = sns.FacetGrid(train_df, row='Adultw', height=2.2, aspect=1.6)
grid.map(plt.plot, "Parch", "Survived", marker=".")
grid.add_legend()

# Total probability of survival P(SnParch=i)
parch_surv = train_df[["Survived", "Parch"]].groupby(['Parch'], as_index=False)\
.sum().sort_values(by='Survived', ascending=False)
print("NOMBRE TOTAL DE SURVIVANT DANS CHAQUE CATEGORIE")
print(parch_surv)
print("######################################")
ps = nb_survival/nb_pass
print("La probabilité totale de survie est de {}".format(nb_survival/nb_pass))
parch_surv["Survived"] = parch_surv["Survived"]/nb_pass
print("PROBABILITE TOTALE DE SURVIE ET DE PARCH")
print(parch_surv)


#plt.show()
