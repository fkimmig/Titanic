import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

###########Import data
training_file = pd.read_csv('../train.csv')

###########Choose and clean relevant data
train_data = training_file
###choose columns : keep 'Sex' 'Age' 'Pclass' and 'Survived'
train_data = train_data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
###replace sex by float
sex_dict = {'male':0,'female':1}
train_data = train_data.replace(sex_dict)
###drop nan values
train_data = train_data.dropna()
###add age bins
bins = [0, 5, 18, 40, np.inf]
labels = ['0', '1', '2', '3']
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins, labels=labels)

###########Split train_data into one part for training and one part for testing accuracy of algorithm
Xy_train, Xy_test = train_test_split(train_data, test_size = 0.25)

###########Export csv files from new train and test data
Xy_train.to_csv(path_or_buf='train_SAPS.csv', index=False)
Xy_test.to_csv(path_or_buf='test_SAPS.csv', index=False)
