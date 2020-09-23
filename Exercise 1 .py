#!/usr/bin/env python
# coding: utf-8

# In[22]:


#Maryam Assaedi
#Mst. Mahfuja Akter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
tips = sns.load_dataset("tips")

dayMeal=sns.relplot(x="time",y="day",data=tips)
plt.xlabel("Meal Time")
plt.ylabel("Day of Week")
plt.title("Recorded Meal Time per Day of Week")
plt.show(dayMeal)


tipPerTotal = sns.relplot(x="total_bill", y="tip",hue="sex", data=tips);
plt.xlabel("Total Bill")
plt.ylabel("Tips")
plt.title("Tips per Total Bill Amount")
plt.show(tipPerTotal)




# In[ ]:





# In[ ]:





# In[ ]:
