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
from sklearn import metrics
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
combine_0 = [train_df, test_df]

##
for dataset in combine_0:
    ## REMOVE COLUMN
    dataset.drop(['PassengerId','Pclass', 'Name','Sex',\
    'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1,inplace=True)
    ## REMOVE ROWS
    dataset.dropna(subset=["Age"],how='any',inplace=True)
## COMBINE FOR TREATMENT
combine_1 = [train_df, test_df]

################################################################################
###############################################################################
## Siblings
# Transform Sibs to binary
for dataset in combine_1:
    temp_df = map(lambda x: x if (x==0) else 1, dataset["SibSp"])
    dataset["SibSp"]=list(temp_df)
## Parents-children
# Add an information - Adult/Children
for dataset in combine_1 :
    temp_df = map(lambda x: 0 if (x<16) else 1, dataset["Age"])
    dataset["Adultw"] = list(temp_df)
# Transform Parch to
# 0 - adult, no parch
# 1 - is a parent and has a children on board
# 2 - is a children and has a parent on board
# 3 - is a children, no parch
for dataset in combine_1:
    dataset.loc[(dataset['Adultw'] ==1)&(dataset['Parch'] == 0), 'Parch'] = 0
    dataset.loc[(dataset['Adultw'] ==1)&(dataset['Parch'] >0), 'Parch'] = 1
    dataset.loc[(dataset['Adultw'] ==0)&(dataset['Parch'] >0), 'Parch'] = 2
    dataset.loc[(dataset['Adultw'] ==0)&(dataset['Parch'] == 0), 'Parch'] = 3

##################################################################################
### MODEL - PREDICT - SOLVE
# Preparing data
X_train = train_df.drop("Survived", axis=1).copy()
Y_train = train_df["Survived"]
X_test  = test_df.copy()

# Lorgistic regression
# Create an instance of the MODEL
logreg = LogisticRegression()
#Training the model on the data, storing the information learned from the data
logreg.fit(X_train, Y_train)
#Predict labels for new data (new images)
Y_pred = logreg.predict(X_test)
#Measuring Model Performance (for train_df)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("le score de la régression linéaire est de {}%".format(acc_log))

##################################################################################
### PLOT CONFUSION matrix
Y_test = np.random.rand(332)
Y_test = list(map(lambda x: 1 if (x>0.38) else 0, Y_test))
cm = metrics.confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["0", "1"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x),
  horizontalalignment='center',
  verticalalignment='center')
plt.show()
