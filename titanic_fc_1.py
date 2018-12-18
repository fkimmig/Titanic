import csv
## Load the Numpy libraries with alias 'np'
import numpy as np
## Load the Pandas libraries
import pandas as pd
## Load matplotlib libraries
import matplotlib.pyplot as plt
## Load seaborn libraries
import seaborn as sns
#matplotlib inline
#matplotlib.style.use('ggplot')

## Read data from file 'train.csv'
df = pd.read_csv("train.csv")
print(df)

## Preview the first 5 lines of the loaded data
df.head()

## drop rows with NaN (axis = 0)
#df.dropna()
## drop columns with NaN
#df.dropna(axis=1)



## replace NaN with 'NotANumber'
#df.replace(pd.NaN,value = 'NotANumber',inplace = True)
df.fillna(value = 'NotANumber',inplace = True)

## create new .csv
df.to_csv("trainModifiedFC.csv", index = False)

## create new .xls
#df.to_excel("trainModifiedFC.xls", index = False)


## plot scatterplot
grid = sns.FacetGrid(df, col='Survived', hue='Pclass')
grid.map(plt.scatter, 'Pclass', color = 'c')
grid.add_legend();
plt.show()

## plot histogram
bins = np.arange(0, 3, 1)
grid = sns.FacetGrid(df, col='Survived', row='Pclass')
grid.map(plt.hist, 'Pclass', bin=bins, color = 'c')
grid.add_legend();
plt.show()


