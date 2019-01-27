import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

###########Import data
###load data
train_data = pd.read_csv('./train.csv')

# print data
# print data.head()
# print data.tail()
# print data.index
# print data.columns

#first statistics
# print data.describe()
# print data.describe(include='all')
# print data.groupby('Sex').mean()
# print data.groupby('Survived').mean()

###sex
# 0 for men 1 for women
# sex_dict = {'male':0,'female':1}
# data = data.replace(sex_dict)
# print data
# sns.barplot(x='Sex', y='Survived',data=train_data)
# plt.show()

###age
#step 1 year
plt.figure()
plt.subplot(121)
train_data_nonan = train_data['Age'][np.logical_not(np.isnan(train_data['Age']))]
sns.distplot(train_data_nonan, bins=100)
plt.subplot(122)
sns.barplot(x='Age', y='Survived', data=train_data)
#step 10 years
plt.figure()
plt.subplot(122)
train_data['Age']=train_data['Age'].fillna(-0.5)
bins1 = [-1] + list(range(0,100,10))
labels1 = ['Unknown'] + [str(age) for age in range(10,100,10)]
train_data['AgeGroup10'] = pd.cut(train_data['Age'], bins1, labels=labels1)
sns.barplot(x='AgeGroup10', y='Survived', data=train_data)
plt.subplot(121)
plt.title('Data state')
train_data['AgeGroup10'].value_counts().loc[labels1].plot(kind='bar')
#step 20 years
plt.figure()
plt.subplot(122)
bins2 = [-1] + list(range(0,100,20))
labels2 = ['Unknown'] + [str(age) for age in range(20,100,20)]
train_data['AgeGroup20'] = pd.cut(train_data['Age'], bins2, labels=labels2)
sns.barplot(x='AgeGroup20', y='Survived', data=train_data)
plt.subplot(121)
plt.title('Data state')
train_data['AgeGroup20'].value_counts().loc[labels2].plot(kind='bar')
#categories
plt.figure()
plt.subplot(122)
bins3 = [-1, 0, 5, 12, 18, 25, 40, 60, np.inf]
labels3 = ['Unknown', 'Baby', 'Child', 'Teenager', 'Young', 'Adult', 'Senior', 'Old']
train_data['AgeGroupCat1'] = pd.cut(train_data['Age'], bins3, labels=labels3)
sns.barplot(x='AgeGroupCat1', y='Survived', data=train_data)
plt.subplot(121)
plt.title('Data state')
train_data['AgeGroupCat1'].value_counts().loc[labels3].plot(kind='bar')
#less categories
plt.figure()
plt.subplot(122)
plt.title('Survival rate')
bins4 = [-1, 0, 18, 35, 60, np.inf]
labels4 = ['Unknown', 'Child', 'Young Adult', 'Adult', 'Senior']
train_data['AgeGroupCat2'] = pd.cut(train_data['Age'], bins4, labels=labels4)
sns.barplot(x='AgeGroupCat2', y='Survived', data=train_data)
plt.subplot(121)
plt.title('Data state')
train_data['AgeGroupCat2'].value_counts().loc[labels4].plot(kind='bar')
#last one
plt.figure()
plt.subplot(122)
plt.title('Survival rate')
bins5 = [0, 5, 18, 40, np.inf]
labels5 = ['0', '1', '2', '3']
train_data['AgeGroupCat3'] = pd.cut(train_data['Age'], bins5, labels=labels5)
sns.barplot(x='AgeGroupCat3', y='Survived', data=train_data)
plt.subplot(121)
plt.title('Data state')
train_data['AgeGroupCat3'].value_counts().loc[labels5].plot(kind='bar')

plt.show()
