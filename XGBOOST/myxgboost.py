# -*- coding : utf-8 -*-

### IMPORTATION PYTHON
# General management
import os
import sys
# Data analysis ans wrangling
import numpy as np
import pandas as pd
import random as random
from scipy.optimize import minimize
# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
# machine learning
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

################################################################################
################################################################################

### CREATE DATA
# Input sample
# Month in our phd
x = np.linspace(0,35,36)
x_train = x.reshape(-1, 1)
# Output sample
# Level of work during phd
s = np.linspace(0.1,0.3,3)
y = np.concatenate((0.1*np.random.rand(6),\
      0.2*np.ones(6),\
      np.zeros(3),\
      np.square(s),\
      s,s+0.3, s+0.6,\
      np.ones(6),\
      2-np.exp(s)))
y_train = y.reshape(-1, 1)
### PLOTS !!
plt.title("Jeu de données")
plt.plot(x, y, '+', label = "phd work level")
plt.xlabel('temps')
plt.ylabel('travail')



################################################################################
################################################################################

### BOOSTING STEP 0
### defining the loss function
def Lossfunction0(gamma,xi,yi):
    loss = np.linalg.norm(yi-gamma*np.ones(36))**2
    return loss
# optimize the learner
bstep0 = minimize(Lossfunction0, 0, args=(x,y),method='nelder-mead',\
options={'xtol': 1e-8, 'disp': True})
f0 = bstep0.x*np.ones(36)
# plot the result
plt.plot(x, f0, '-', label = "f0")

### BOOSTING STEP 1
def Lossfunction1(h,xi,yi,f0i):
    yihat = map(lambda x: \
    h[1] if (x<h[0]) else h[2], xi)
    yihat = np.array(list(yihat))
    loss = np.linalg.norm(yi-f0i-yihat)**2
    return loss
# optimize the learner
bstep1 = minimize(Lossfunction1, np.zeros(3), \
args=(x,y,f0),method='nelder-mead',\
options={'xtol': 1e-8, 'disp': True})
h1 = bstep1.x
print(h1)
f1 = map(lambda x: \
h1[1] if (x<h1[0]) else h1[2], x)
f1 = np.array(list(f1))

plt.plot(x,f0+f1, '-', label = "f1")

### BOOSTING STEP 2
# def Lossfunction2(h,xi,yi,f0i,f1i):
#     yihat = map(lambda x: \
#     h[1] if (x<h[0]) else h[2], xi)
#     yihat = np.array(list(yihat))
#     loss = np.linalg.norm(yi-f0i-f1i-yihat)**2
#     return loss
# # optimize the learner
# bstep2 = minimize(Lossfunction2, h1, \
# args=(x,y,f0,f1),method='nelder-mead',\
# options={'xtol': 1e-8, 'disp': True})
# h2 = bstep2.x
# print(h2)
# f2= x - f0 -f1
# f2 = map(lambda x: \
# h2[1] if (x<h2[0]) else h2[2], f2)
# f2 = np.array(list(f2))
# plt.plot(x,f2, '-', label = "f2")


################################################################################
################################################################################

ft = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,\
max_depth=1, random_state=0, loss='ls').fit(x_train, y_train)

error = mean_squared_error(y_train, ft.predict(x_train))

ft = ft.predict(x_train).reshape(36)

# plot the result
plt.plot(x, ft, '-', label = "error ={}".format(error))

################################################################################
################################################################################
plt.legend()
plt.show()
