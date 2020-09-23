# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:28:18 2019

#Maryam Assaedi
#Mst. Mahfuja Akter
#Mahpara Hyder Chowdhury
"""
import pandas as pd
import seaborn as sns

#a
df=pd.read_csv('winequality-red.csv', sep=';')
print(df.head())

#b
#range from 3 till 8
sns.distplot(df['quality'],kde=False);

#c
#replace quality with labels
df.quality.replace([3,4,5,6,7,8], ['low','low','medium','medium','high','high'], inplace=True)
df.rename(columns={'quality':'quality bin'}, inplace=True)

#d
highLowQuality = df[df['quality bin']!='medium']

#e
sns.pairplot(highLowQuality, hue="quality bin")

#g
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectPercentile, f_classif

X = highLowQuality.drop('quality bin',axis=1).values
y = highLowQuality['quality bin'].values

plt.figure(1)
plt.clf()

X_indices = np.arange(X.shape[-1])

selector = SelectPercentile(f_classif)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices, scores)

plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.show()

#g
filtered=pd.DataFrame()
filtered['volatile acidity']=highLowQuality['volatile acidity']
filtered['citric acid']=highLowQuality['citric acid']
filtered['pH']=highLowQuality['pH']
filtered['sulphates']=highLowQuality['sulphates']
filtered['alcohol']=highLowQuality['alcohol']
filtered['quality bin']=highLowQuality['quality bin']

#h
g = sns.PairGrid(filtered, hue = 'quality bin')
g = g.map_upper(plt.scatter)
g=g.map_lower( sns.regplot,scatter=False)
g=g.map_diag(sns.kdeplot)
