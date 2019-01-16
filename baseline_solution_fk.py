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

# --- load data
path = os.path.dirname(os.path.realpath(__file__))
data_training_raw = pd.read_csv(path+"/train.csv")
data_test_raw = pd.read_csv(path+"/test.csv")

# --- model parametrisation definition
proba_male_first_class = 0.35
proba_female_second_class = 0.9
proba_male_second_class = 0.15
# proba_child_second_class = 1
proba_female_third_class = 0.4
proba_male_third_class = 0.15
proba_child_third_class = 0.3

# --- baseline prediction 
def baseline_prediction(passenger_data):
    # categories =  ['Pclass', 'Sex', 'Fare', 'Age', 'Person']
    Pclass,Person = passenger_data
    
    random_draw = random.random()
    
    if Pclass == 1:
        if Person == 'female' or Person == 'child':
            return 1
        else:
            if random_draw < proba_male_first_class:
                return 1
            else:
                return 0
    elif Pclass == 2:
        if Person == 'child':
            return 1
        elif Person == 'female':
            if random_draw < proba_female_second_class:
                return 1
            else:
                return 0
        else:
            if random_draw < proba_male_second_class:
                return 1
            else:
                return 0
    elif Pclass == 3:
        if Person == 'child':
            if random_draw < proba_child_third_class:
                return 1
            else:
                return 0
        elif Person == 'female':
            if random_draw < proba_female_third_class:
                return 1
            else:
                return 0
        else:
            if random_draw < proba_male_third_class:
                return 1
            else:
                return 0
    
# --- add Person feature in the data set 
child_age = 16
def is_child(person):
    age,sex = person
    return 'child' if age < child_age else sex
    
data_training_raw['Person'] = data_training_raw[['Age','Sex']].apply(is_child,axis=1) # axis = 1 : data in column
data_test_raw['Person']     = data_test_raw[['Age','Sex']].apply(is_child,axis=1) # axis = 1 : data in column

# --- make prediction
prediction_training_set = data_training_raw[['Pclass','Person']].apply(baseline_prediction,axis=1) # axis = 1 : data in column
prediction_test_set     = data_test_raw[['Pclass','Person']].apply(baseline_prediction,axis=1) # axis = 1 : data in column

# --- compare with truth
def prediction_accuracy(prediction, truth):
    if len(prediction) != len(truth):
        raise ValueError("'prediction' and 'truth' must have the same size") 

    cpt = 0
    
    for i in range(0, len(prediction) - 1):
        if prediction[i] == truth[i]:
            cpt = cpt + 1
    
    accuracy = cpt / len(truth)
    return accuracy
    

prediction_accuracy_training_set = prediction_accuracy(prediction_training_set, data_training_raw['Survived'])

# --- print out results
print("######################################")
print(prediction_accuracy_training_set)
