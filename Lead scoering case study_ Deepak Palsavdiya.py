#!/usr/bin/env python
# coding: utf-8

# In[35]:


#Import libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import warnings
import warnings
warnings.filterwarnings('ignore')


# In[36]:


#importing data

data=pd.read_csv("Leads.csv")
data.head()


# In[26]:


data.info()


# In[37]:


data.describe()


# # Data handling
# # check for Null values
# 

# In[ ]:


#there are multiple columns with values as "select", it means end user did not selcte anything while filling the form/survey


# In[38]:


#Need to deal with "selects"
data=data.replace('Select',np.nan)


# In[29]:


data.head()


# In[ ]:


#checking for null values


# In[39]:


data.isnull().sum()


# In[40]:


round(data.isnull().sum()/len(data.index),2)*100


# In[41]:


# Dropping the columns which have 40% are greater missing values
data=data.drop(columns=['I agree to pay the amount through cheque','Asymmetrique Profile Score','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Activity Index','Lead Quality','Lead Profile','How did you hear about X Education'])


# In[42]:


#checking null% once again 
round(data.isnull().sum()/len(data.index),2)*100


# In[20]:


#there are 2 column with 37% and 36% missing values 
-Specialization 37% missing values 
- Tags 36% missing values
- city also have 40% missing values


# In[43]:


#lets check the distribution of each one above mentioned columns 

#Specialization

plt.figure(figsize= (22,7))
sns.countplot(data['Specialization'])
plt.xticks(rotation=90)


# In[44]:


#assumption - as 37% missing values in specilaization column might be due to -lead did not select any specilazation. 
#to deal with this we can add another Category  'Others' 

data['Specialization']= data['Specialization'].replace(np.nan, 'others')


# In[46]:


#deal with second clolums 'Tags'
plt.figure(figsize=(20,14))
sns.countplot(data['Tags'])
plt.xticks(rotation =90)


# In[47]:


# as most of the values under "revert after reading the email", we impute this for missing values
data['Tags']= data['Tags'].replace(np.nan, 'Will revert after reading the email')


# In[48]:



#checking null% once again 
round(data.isnull().sum()/len(data.index),2)*100


# In[50]:


#city column has 40% missing values, lets deal with missing values for this column

plt.figure(figsize=(15,8))
sns.countplot(data['City'])
plt.xticks(rotation=90)


# In[51]:


#as more than 50% leads are from Mumbai, we can impute Mumbai for missing values
data['City']=data['City'].replace(np.nan,'Mumbai')


# In[52]:


#Next column to deal with missing values is 'Country' which have 27% missing values

plt.figure(figsize=(15,8))
sns.countplot(data['Country'])
plt.xticks(rotation=90)


# In[53]:


# as most of the leads are from India we can impute missing values with India

data['Country']=data['Country'].replace(np.nan,'India')


# In[54]:


#checking null% once again 
round(data.isnull().sum()/len(data.index),2)*100


# In[56]:


#still we have 2 columns with missing values
#1- What matters most to you in choosing a course    29.0% lets deal with this first

plt.figure(figsize=(10,6))
sns.countplot(data['What matters most to you in choosing a course'])
plt.xticks(rotation=90)


# In[57]:


#as this is highly skewes column which might influence model output so we need to remove this column 

data= data.drop('What matters most to you in choosing a course', axis=1)


# In[58]:


#another column with missing values is "What is your current occupation" 
plt.figure(figsize=(10,6))
sns.countplot(data['What is your current occupation'])
plt.xticks(rotation=90)


# In[59]:


# As most of the leads are Unemployed we can impute this for missing values
data['What is your current occupation']= data['What is your current occupation'].replace(np.nan,'Unemployed')


# In[60]:


#checking null% once again 
round(data.isnull().sum()/len(data.index),2)*100


# In[61]:


#dropping the missing values rows as these are 1% missing values
data.dropna(inplace= True)


# In[62]:


#checking null% once again 
round(data.isnull().sum()/len(data.index),2)*100


# In[63]:


data.info()


# # Exploraory Data Analysis 
# 
# 1 Univarite Analysis and bivariate Analysis
# 
# 

# In[67]:


###1 column  Converted : This is our target variable. If its value is 1 , it means lead converted Successfully
                       # If its value is 0 then it means lead is not converted properly 
    
  #checking for conversion in shared data = 

Conversion_rate=round(sum(data['Converted'])/len(data['Converted'].index),2)*100
Conversion_rate


# In[76]:


## 2 lead origin 

plt.figure(figsize=(6,3))
sns.countplot(x='Lead Origin',hue="Converted", data=data)
plt.xticks(rotation=30)


# # Understanding 
# #1 - lead genrated from API and Landing page submission has lower converstion rate (30-50%)
# #2 - Lead genrated from Lead add form also has high converstion rate (almost 100%) but the leads are very low in terms of count
# #3 - Lead import count are vary low as compared to other categories 
# 
# --- To Improve overall conversation rate we need to more focus on API leads and Landing page leads---

# In[84]:


# Lead Source 

plt.figure(figsize=(15,5))
sns.countplot(x="Lead Source", hue="Converted", data=data)
plt.xticks(rotation=90)


# In[ ]:




