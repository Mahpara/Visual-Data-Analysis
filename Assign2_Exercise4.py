# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:36:20 2019

#Mst. Mahfuja Akter
#Maryam Assaedi
#Mahpara Hyder Chowdhury
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#a
data_from_excel = pd.read_excel("chronic_kidney_disease_numerical.xls", 
                                sheet_name="chronic_kidney_disease_full");

#b
melt_wide_to_long = pd.melt(data_from_excel, id_vars=["class"], 
                            var_name="Name", value_name="value");
melt_wide_to_long.to_excel(r'wide_to_long_output.xlsx', 
                           sheet_name='transformedData', index=False);

#c
normalizedf=pd.DataFrame()
normalizedf['class'] = data_from_excel['class'].values
def norm_to_zero_one(data_from_excel):
    return (data_from_excel - data_from_excel.min())*1.0 /(data_from_excel.max()-data_from_excel.min())
for column in data_from_excel:
    if(column != 'class'):
        columndfe = data_from_excel[column]
        normalized_x =norm_to_zero_one(columndfe)
        normalizedf[column]=normalized_x.values
dnormalizedf = pd.melt(normalizedf, id_vars=["class"], var_name= "Name", value_name="value");
sns.boxplot(x="Name", y="value", hue=dnormalizedf._series['class'],data=dnormalizedf)
plt.gcf().set_size_inches(25, 15)