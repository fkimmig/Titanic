import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

###########Import data
###load data
training_file = pd.read_csv('./train.csv')
test_file = pd.read_csv('./test.csv')

###########Choose and clean relevant data
###keep only 'Sex' 'Age' and 'Survived'
train_data = training_file
train_data = train_data.drop(['PassengerId', 'Pclass', 'Name', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
###replace sex by float
sex_dict = {'male':0,'female':1}
train_data = train_data.replace(sex_dict)
###drop nan values
train_data = train_data.dropna()
###separate train columns and target column
train_target = train_data["Survived"]
train_data = train_data.drop('Survived',axis=1)

###same on test data
test_data = test_file.drop(['PassengerId', 'Pclass', 'Name', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
test_data = test_data.replace(sex_dict)
test_data = test_data.dropna()

##########Perform RandomForest algorithm
randomforest = RandomForestClassifier()
randomforest.fit(train_data, train_target)
y_pred = randomforest.predict(test_data)

##########Evaluate accuracy
x_train, x_val, y_train, y_val = train_test_split(train_data, train_target, test_size = 0.25)

randomforest_acc = RandomForestClassifier()
randomforest_acc.fit(x_train, y_train)
y_pred = randomforest_acc.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print('accuracy if no change in age column',acc_randomforest)

##########Evaluate accuracy for age bins
###Define bins
bins = [0, 5, 18, 40, np.inf]
labels = ['0', '1', '2', '3']
###Change train data
train_data2 = train_data
train_data2['AgeGroup'] = pd.cut(train_data2['Age'], bins, labels=labels)
train_data2 = train_data2.drop('Age', axis=1)
###evaluate accuracy
x_train, x_val, y_train, y_val = train_test_split(train_data2, train_target, test_size = 0.25)
randomforest_acc2 = RandomForestClassifier()
randomforest_acc2.fit(x_train, y_train)
y_pred = randomforest_acc2.predict(x_val)
acc_randomforest2 = round(accuracy_score(y_pred, y_val) * 100, 2)
print('accuracy if age categories',acc_randomforest2)
