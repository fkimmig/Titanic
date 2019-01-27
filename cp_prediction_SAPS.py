import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


###########Import data
train_file = pd.read_csv('Cecile/train_SAPS.csv')
test_file = pd.read_csv('Cecile/test_SAPS.csv')

##########Function to predict
def predict_randomforest(train, test):

    y_train = train['Survived']
    X_train = train.drop(['Survived'], axis=1)
    y_test = test['Survived']
    X_test = test.drop(['Survived'], axis=1)

    randomforest_acc = RandomForestClassifier()
    randomforest_acc.fit(X_train, y_train)
    y_pred = randomforest_acc.predict(X_test)
    acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)

    return acc_randomforest

##################
###Use Sex and Age
##################
train1 = train_file
test1 = test_file

train1 = train1.drop(['Pclass','AgeGroup'], axis=1)
test1 = test1.drop(['Pclass', 'AgeGroup'], axis=1)

acc_randomforest1 = predict_randomforest(train1, test1)
acc_randomforest1_on_test = predict_randomforest(train1, train1)

##################
###Use Sex and Age bins
##################
train2 = train_file
test2 = test_file

train2 = train2.drop(['Pclass', 'Age'], axis=1)
test2 = test2.drop(['Pclass', 'Age'], axis=1)

acc_randomforest2 = predict_randomforest(train2, test2)
acc_randomforest2_on_test = predict_randomforest(train2, train2)

##################
###Use Sex and Age and PClass
##################
train3 = train_file
test3 = test_file

train3 = train3.drop(['AgeGroup'], axis=1)
test3 = test3.drop(['AgeGroup'], axis=1)

acc_randomforest3 = predict_randomforest(train3, test3)
acc_randomforest3_on_test = predict_randomforest(train3, train3)

##################
###Use Sex and Age bins and PClass
##################
train4 = train_file
test4 = test_file

train4 = train4.drop(['Age'], axis=1)
test4 = test4.drop(['Age'], axis=1)
acc_randomforest4 = predict_randomforest(train4, test4)
acc_randomforest4_on_test = predict_randomforest(train4, train4)


###################
###RESULT##########
###################

print('accuracy with sex and age',acc_randomforest1)
print('accuracy with sex and age',acc_randomforest1_on_test)
print('accuracy with sex and age bins',acc_randomforest2)
print('accuracy with sex and age bins',acc_randomforest2_on_test)
print('accuracy with sex and age and pclass',acc_randomforest3)
print('accuracy with sex and age and pclass',acc_randomforest3_on_test)
print('accuracy with sex and age bins and pclass',acc_randomforest4)
print('accuracy with sex and age bins and pclass',acc_randomforest4_on_test)
