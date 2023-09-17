#!/usr/bin/env python
# coding: utf-8

# In[217]:


#Import libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve

#import warnings
import warnings
warnings.filterwarnings('ignore')


# In[5]:


#importing data

m_data=pd.read_csv("Leads.csv")
m_data.head()


# In[6]:


m_data.info()


# In[7]:


m_data.describe()


# # Data handling
# # check for Null values
# 

# In[ ]:


#there are multiple columns with values as "select", it means end user did not selcte anything while filling the form/survey


# In[8]:


#Need to deal with "selects"
m_data=m_data.replace('Select',np.nan)


# In[9]:


m_data.head()


# In[ ]:


#checking for null values


# In[10]:


m_data.isnull().sum()


# In[11]:


round(m_data.isnull().sum()/len(m_data.index),2)*100


# In[12]:


# Dropping the columns which have 40% are greater missing values
m_data=m_data.drop(columns=['I agree to pay the amount through cheque','Asymmetrique Profile Score','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Activity Index','Lead Quality','Lead Profile','How did you hear about X Education'])


# In[13]:


#checking null% once again 
round(m_data.isnull().sum()/len(m_data.index),2)*100


# In[ ]:


#there are 2 column with 37% and 36% missing values 
-Specialization 37% missing values 
- Tags 36% missing values
- city also have 40% missing values


# In[14]:


#lets check the distribution of each one above mentioned columns 

#Specialization

plt.figure(figsize= (22,7))
sns.countplot(m_data['Specialization'])
plt.xticks(rotation=90)


# In[15]:


#assumption - as 37% missing values in specilaization column might be due to -lead did not select any specilazation. 
#to deal with this we can add another Category  'Others' 

m_data['Specialization']= m_data['Specialization'].replace(np.nan, 'others')


# In[16]:


#deal with second clolums 'Tags'
plt.figure(figsize=(20,14))
sns.countplot(m_data['Tags'])
plt.xticks(rotation =90)


# In[17]:


# as most of the values under "revert after reading the email", we impute this for missing values
m_data['Tags']= m_data['Tags'].replace(np.nan, 'Will revert after reading the email')


# In[18]:



#checking null% once again 
round(m_data.isnull().sum()/len(m_data.index),2)*100


# In[19]:


#city column has 40% missing values, lets deal with missing values for this column

plt.figure(figsize=(15,8))
sns.countplot(m_data['City'])
plt.xticks(rotation=90)


# In[20]:


#as more than 50% leads are from Mumbai, we can impute Mumbai for missing values
m_data['City']=m_data['City'].replace(np.nan,'Mumbai')


# In[21]:


#Next column to deal with missing values is 'Country' which have 27% missing values

plt.figure(figsize=(15,8))
sns.countplot(m_data['Country'])
plt.xticks(rotation=90)


# In[22]:


# as most of the leads are from India we can impute missing values with India

m_data['Country']=m_data['Country'].replace(np.nan,'India')


# In[23]:


#checking null% once again 
round(m_data.isnull().sum()/len(m_data.index),2)*100


# In[24]:


#still we have 2 columns with missing values
#1- What matters most to you in choosing a course    29.0% lets deal with this first

plt.figure(figsize=(10,6))
sns.countplot(m_data['What matters most to you in choosing a course'])
plt.xticks(rotation=90)


# In[25]:


#as this is highly skewes column which might influence model output so we need to remove this column 

m_data= m_data.drop('What matters most to you in choosing a course', axis=1)


# In[26]:


#another column with missing values is "What is your current occupation" 
plt.figure(figsize=(10,6))
sns.countplot(m_data['What is your current occupation'])
plt.xticks(rotation=90)


# In[27]:


# As most of the leads are Unemployed we can impute this for missing values
m_data['What is your current occupation']= m_data['What is your current occupation'].replace(np.nan,'Unemployed')


# In[28]:


#checking null% once again 
round(m_data.isnull().sum()/len(m_data.index),2)*100


# In[29]:


#dropping the missing values rows as these are 1% missing values
m_data.dropna(inplace= True)


# In[30]:


#checking null% once again 
round(m_data.isnull().sum()/len(m_data.index),2)*100


# In[31]:


m_data.info()


# # Exploraory Data Analysis 
# 
# 1 Univarite Analysis and bivariate Analysis
# 
# 

# In[32]:


###1 column  Converted : This is our target variable. If its value is 1 , it means lead converted Successfully
                       # If its value is 0 then it means lead is not converted properly 
    
  #checking for conversion in shared data = 

Conversion_rate=round(sum(m_data['Converted'])/len(m_data['Converted'].index),2)*100
Conversion_rate


# In[33]:


## 2 lead origin 

plt.figure(figsize=(6,3))
sns.countplot(x='Lead Origin',hue="Converted", data=m_data)
plt.xticks(rotation=30)


# # Understanding 
# #1 - lead genrated from API and Landing page submission has lower converstion rate (30-50%)
# #2 - Lead genrated from Lead add form also has high converstion rate (almost 100%) but the leads are very low in terms of count
# #3 - Lead import count are vary low as compared to other categories 
# 
# --- To Improve overall conversation rate we need to more focus on API leads and Landing page leads---

# In[34]:


# Lead Source


plt.figure(figsize=(12,6))
sns.countplot(x='Lead Source',hue="Converted", data=m_data)
plt.xticks(rotation=90)


# In[35]:


## as Google and google are same source we need to replace it

m_data['Lead Source']=m_data['Lead Source'].replace(['google'],'Google')


# In[36]:


#As few lead sources don't have much values we can create a Category other which include all of them

m_data['Lead Source']=m_data['Lead Source'].replace(['Click2call','Live Chat','welearnblog_Home','youtubechannel','Press_Release','bing','testone','Press_Release','NC_EDM','WeLearn','Pay per Click Ads','Social Media'],'Others')


# In[37]:


plt.figure(figsize=(9,4))
sns.countplot(x='Lead Source',hue="Converted", data=m_data)
plt.xticks(rotation=90)


# In[38]:


## Outcomes

#1- Direct Traffice and Google are the highest lead genrators
#2  lead received through Refrence and Welingak websites has highest conversion rates
#3  To improve conversion rate we need to more focus on "Olark Chat", "Organic Search", "Direct Traffic" and "Google leads" 


# In[39]:


## Do Not email

plt.figure(figsize=(6,4))
sns.countplot(x='Do Not Email',hue="Converted", data=m_data)


# In[40]:


## Outcome - maximum entries are no, so its not relevent 


# In[41]:


## 5 Do not call

plt.figure(figsize=(6,3))
sns.countplot(x='Do Not Call',hue="Converted", data=m_data)


# In[42]:


## Outcome- As all entries are no, so no outcome can be drwan from this


# In[43]:


# check percentile for TotalVisits
m_data['TotalVisits'].describe(percentiles=[0.05,0.25,0.5,0.75,0.90,0.95,0.99])


# In[44]:


sns.boxplot(m_data['TotalVisits'])


# In[45]:


# as we have multipleoutliers so we can not cosider this for our analysis 


# In[46]:


## Page Views per visit

m_data['Page Views Per Visit'].describe()


# In[47]:


sns.boxplot(m_data['Page Views Per Visit'])


# In[48]:


# as there are outliers in this columns as well so we not consider this in our analysis else it will Influence our final results


# In[49]:


# 9 - Last Activity 

m_data['Last Activity'].describe()


# In[50]:


plt.figure(figsize=(9,4))
sns.countplot(x='Last Activity', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[51]:


# for better understandig and visibility we will club few activities to others whos values are very low

m_data['Last Activity']=m_data['Last Activity'].replace(['Had a Phone Conversation','View in browser link Clicked','Visited Booth in Tradeshow','Approached upfront','Resubscribed to emails','Email Received','Email Marked Spam'],'Others')


# In[52]:


plt.figure(figsize=(9,4))
sns.countplot(x='Last Activity', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[53]:


## Observations - 
##1- last activity as "Email Opend" is highest
##2- COnvertion rate for last activity "SMS sent" is highest


# In[54]:


#country
plt.figure(figsize=(9,4))
sns.countplot(x='Country', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[55]:


# As maximum values are India, nothing can be concluded from this


# In[56]:


## Speialization

plt.figure(figsize=(9,4))
sns.countplot(x='Specialization', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[57]:


## Insted of others we can put our major focus on specialization


# In[58]:


# 12 what is your current Occupation 


# In[59]:


plt.figure(figsize=(9,4))
sns.countplot(x='What is your current occupation', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[60]:


## observation
#1- Working professional having highest conversion 
#2- lead for Unemployed had low conversion


# In[61]:


## 13 Search
plt.figure(figsize=(9,4))
sns.countplot(x='Search', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[62]:


#most repsonce are no, so nothing can be concluded


# In[63]:


## 14 Magazine 
plt.figure(figsize=(9,4))
sns.countplot(x='Magazine', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[64]:


#most repsonce are no, so nothing can be concluded


# In[65]:


##15 Newspaper Article
plt.figure(figsize=(9,4))
sns.countplot(x='Newspaper Article', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[66]:


#most repsonce are no, so nothing can be concluded


# In[67]:


# 16 X Education Forums
plt.figure(figsize=(9,4))
sns.countplot(x='X Education Forums', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[68]:


#most repsonce are no, so nothing can be concluded


# In[69]:


## 17 Newspaper
plt.figure(figsize=(9,4))
sns.countplot(x='Newspaper', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[70]:


#most repsonce are no, so nothing can be concluded


# In[71]:


# 18 Digital Advertisment
plt.figure(figsize=(9,4))
sns.countplot(x='Digital Advertisement', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[ ]:


#most repsonce are no, so nothing can be concluded


# In[72]:


# Through Recommendations 

plt.figure(figsize=(9,4))
sns.countplot(x='Through Recommendations', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[73]:


#most repsonce are no, so nothing can be concluded


# In[74]:


##20 Receive More Updates ABout Our Courses
plt.figure(figsize=(9,4))
sns.countplot(x='Receive More Updates About Our Courses', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[75]:


#most repsonce are no, so nothing can be concluded


# In[76]:


# 21 Tags

plt.figure(figsize=(9,4))
sns.countplot(x='Tags', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[77]:


## As Tag is the column which genrated by internal sales team, so we will remove this before start building model


# In[78]:


## 22 Update me on Supply Chain Content

plt.figure(figsize=(9,4))
sns.countplot(x='Update me on Supply Chain Content', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[79]:


#most repsonce are no, so nothing can be concluded


# In[80]:


## Get Updates on DM Content
plt.figure(figsize=(9,4))
sns.countplot(x='Get updates on DM Content', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[81]:


#most repsonce are no, so nothing can be concluded


# In[82]:


## 24 City
plt.figure(figsize=(9,4))
sns.countplot(x='City', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[83]:


## Most leads are from Mumbai


# In[84]:


## 25 A free copy of Mastering The Interview
plt.figure(figsize=(9,4))
sns.countplot(x='A free copy of Mastering The Interview', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[85]:


#most repsonce are no, so nothing can be concluded


# In[86]:


#26 Last Notable Activity
plt.figure(figsize=(9,4))
sns.countplot(x='Last Notable Activity', hue ="Converted", data= m_data)
plt.xticks(rotation=90)


# In[87]:


# Based on above analysis we observe that most of the columns are not adding any value to our analysis, so we are dropping them


# In[88]:


m_data=m_data.drop(['Lead Number',
       'Country','Search', 'Magazine', 'Newspaper Article', 'X Education Forums',
       'Newspaper', 'Digital Advertisement', 'Through Recommendations',
       'Receive More Updates About Our Courses', 'Tags',
       'Update me on Supply Chain Content', 'Get updates on DM Content',
       'A free copy of Mastering The Interview'],1)


# In[89]:


m_data.info()


# # Final data preparation 
# 
# ##1 Converting yes/no columns to 1/o for better analysis 

# In[90]:


binary_data_c=['Do Not Email', 'Do Not Call']
def con_bin (x):
    return x.map({"Yes":1, "No":0})

m_data[binary_data_c] = m_data[binary_data_c].apply (con_bin)


# In[91]:


## creating dummy variables  for categorical features 
# Lead Origin, Lead Source , Last Activity , Specialization ,  What is your current occupation, city, Last Notable Activity


# In[92]:


dummy_data=pd.get_dummies(m_data[["Lead Origin", 
                                  "Lead Source" , "Last Activity" , "Specialization" , 
                                  "What is your current occupation", "City", "Last Notable Activity"]], drop_first=True)
dummy_data.head()


# In[93]:


# concatenating with primary data

m_data=pd.concat([m_data, dummy_data],axis=1)
m_data.head()


# In[94]:


## Dropping the original columns for which we created dummies

m_data=m_data.drop(["Lead Origin", 
                                  "Lead Source" , "Last Activity" , "Specialization" , 
                                  "What is your current occupation", "City", "Last Notable Activity"],axis=1)
m_data.head()


# # Data splitting 
# 
# 

# In[95]:


# adding feature variable to X
X= m_data.drop(['Prospect ID', 'Converted'],axis=1)
X.head()


# In[96]:


y=m_data['Converted']
y.head()


# In[97]:


# split into train and test set

X_train, X_test,y_train,y_test=train_test_split(X,y, train_size=0.7, test_size=0.3, random_state =75)


# In[98]:


# Scaling the features 

scaler=StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]= scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
X_train.head()


# # Feature selection uusing RFE
# 
# 

# In[99]:


pip install --upgrade scikit-learn


# In[104]:




# Create a Logistic Regression model
logreg = LogisticRegression()

# Create an RFE object with the Logistic Regression model and specify the number of features to select (20 in this case)
rfe = RFE(estimator=logreg, n_features_to_select=20)

# Fit the RFE model to your data
rfe.fit(X_train, y_train)


# In[106]:


rfe.support_


# In[108]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[109]:


cols= X_train.columns[rfe.support_]
cols


# # Model Building
# 
# Using statsModel
# 

# In[113]:


X_train_sm= sm.add_constant(X_train[cols])
Model1=sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
result=Model1.fit()
result.summary()


# In[116]:


# Pvalue of What is your current occupation_Housewife is very high we can drop this one

col1=cols.drop('What is your current occupation_Housewife')


# # Model-2

# In[118]:


X_train_sm= sm.add_constant(X_train[col1])
Model2=sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
result=Model2.fit()
result.summary()


# In[ ]:


# since Last Notable Activity_Had a Phone Conversation p value is very high we can drop this


# In[119]:


col1=col1.drop('Last Notable Activity_Had a Phone Conversation')


# # Model-3

# In[120]:


X_train_sm= sm.add_constant(X_train[col1])
Model3=sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
result=Model3.fit()
result.summary()


# In[ ]:


#  as P value Last Activity_Unsubscribed for is very high we can drop this 


# In[121]:


col1=col1.drop('Last Activity_Unsubscribed')


# In[ ]:





# # Model 4

# In[122]:


X_train_sm= sm.add_constant(X_train[col1])
Model4=sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
result=Model4.fit()
result.summary()


# In[132]:


# As Lead Source_Reference has high P value, we can drop this
col1=col1.drop('Lead Source_Reference')


# # Model 5 

# In[133]:


X_train_sm= sm.add_constant(X_train[col1])
Model5=sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
result=Model5.fit()
result.summary()


# In[134]:


# as Lead Source_Facebook has high p value we can drop this column
col1=col1.drop('Lead Source_Facebook')


# # Model 6

# In[139]:


X_train_sm= sm.add_constant(X_train[col1])
Model6=sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
result=Model6.fit()
result.summary()


# # Checking for VIF

# In[136]:


vif=pd.DataFrame()
vif['Features']= X_train[col1].columns
vif['VIF']=[variance_inflation_factor(X_train[col1].values,i) for i in range (X_train[col1].shape[1])]
vif['VIF']= round(vif["VIF"],2)
vif=vif.sort_values(by="VIF", ascending = False)
vif


# In[138]:


## Dropping column Last Notable Activity_Unsubscribed
col1=col1.drop('Last Notable Activity_Unsubscribed')


# # Model 7

# In[140]:


X_train_sm= sm.add_constant(X_train[col1])
Model7=sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
result=Model7.fit()
result.summary()


# In[141]:


## Checking for VIF values again
vif=pd.DataFrame()
vif['Features']= X_train[col1].columns
vif['VIF']=[variance_inflation_factor(X_train[col1].values,i) for i in range (X_train[col1].shape[1])]
vif['VIF']= round(vif["VIF"],2)
vif=vif.sort_values(by="VIF", ascending = False)
vif


# In[142]:


## Dropping 'Last Notable Activity_Unreachable' to reduce variables

col1=col1.drop('Last Notable Activity_Unreachable')


# # Model -8

# In[143]:


X_train_sm= sm.add_constant(X_train[col1])
Model8=sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
result=Model8.fit()
result.summary()


# In[144]:


## Checking for VIF values again
vif=pd.DataFrame()
vif['Features']= X_train[col1].columns
vif['VIF']=[variance_inflation_factor(X_train[col1].values,i) for i in range (X_train[col1].shape[1])]
vif['VIF']= round(vif["VIF"],2)
vif=vif.sort_values(by="VIF", ascending = False)
vif


# In[145]:


# dropping "Last Notable Activity_SMS Sent"
col1=col1.drop('Last Notable Activity_SMS Sent')


# # Model 9

# In[146]:


X_train_sm= sm.add_constant(X_train[col1])
Model8=sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
result=Model8.fit()
result.summary()


# In[147]:


## Checking for VIF values again
vif=pd.DataFrame()
vif['Features']= X_train[col1].columns
vif['VIF']=[variance_inflation_factor(X_train[col1].values,i) for i in range (X_train[col1].shape[1])]
vif['VIF']= round(vif["VIF"],2)
vif=vif.sort_values(by="VIF", ascending = False)
vif

# Model 9 is final Model as VIF values are low for all variables and P values is almost 0

# In[150]:


#Making predictions on Train data set to check
# Getting predttion values on train set

y_train_pred=result.predict(X_train_sm)
y_train_pred[:10]


# In[151]:


# changing to array
y_train_pred=y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[152]:


# creating datafram to predict probabilities 

y_train_pred_final=pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID']= y_train.index
y_train_pred_final.head()


# In[154]:


## Creating a new column 'predicted' with 1 if Converted_prob >0.5 else 0


y_train_pred_final['predicted']= y_train_pred_final.Converted_prob.map(lambda x:1 if x>0.5 else 0)
y_train_pred_final.head()


# In[156]:


# Creating confusion matrix

confusion_m=metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted)
print(confusion_m)


# In[ ]:


#predicted/Actual     not_converted      converted

#not_converted        3542               422
#converted            804.               1583


# In[160]:


#overall accuuracy
print('Accuracy :', metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[161]:


TP= confusion_m[1,1]
TN= confusion_m[0,0]
FP= confusion_m[0,1]
FN= confusion_m[1,0]


# In[162]:


# Sensitivity  of our final model
print("Sensitivity :", TP/float(TP+FN))


# In[163]:


# Specificity  of our final model
print("Specificity :", TN/float(TN+FP))


# In[165]:


# Check for false positive  rate

print("False Positive Rate :", FP/float(TN+FP) )


# In[166]:


# Check for Positive Predicitive Value

print("Positive Predicitive Value :", TP/float(TP+FP) )


# In[167]:


# Check for Negative Predicitive Value

print("Negative Predicitive Value :", TN/float(TN+FN) )


# In[ ]:


## Accuracy : 0.8069595339316643
##Sensitivity : 0.6631755341432761
## Specificity : 0.8935418768920282
## False Positive Rate : 0.10645812310797174
## Positive Predicitive Value : 0.7895261845386534
## Negative Predicitive Value : 0.8150023009664059


# In[ ]:


## As Accuracy of Model is 80.69% but Sensitivity is 66.31%

which need to be considered it 

The main reason behind low sensitivity because of the cut of 0.5 we choose . 
As cut off point need to be optimised using ROC curve


# # ROC Curve

# In[184]:


def draw_roc (actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate= False)
    
    auc_score= metrics.roc_auc_score (actual, probs)
    plt.figure (figsize= (5,4))
    plt.plot(fpr,tpr, label ='ROC curve (area= %0.2f)' %auc_score)
    plt.plot([0,1],[0,1],   'k--')
    plt.xlim([0.0, 1.05])
    plt.xlabel ('False Postive Rate or [1- True Negative Rate]')
    plt.ylabel ('True Positive Rate')
    plt.title ('ROC')
    plt.legend (loc= "lower right")
    plt.show()
    
    return None


# In[185]:


fpr, tpr, threshold = metrics.roc_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False)


# In[186]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[ ]:


## Observation -- Since we have higer (0.88) area under the ROC curve, therefore our Final model (Model-9) is 


# # Finding Optimal Cutoff point
# 

# In[ ]:


# Previously we consider 0.5 as our cut of Point but in order to get balance between senstivity and specificity


# In[190]:


numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x:1 if x> i else 0)
    y_train_pred_final.head()


# In[191]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])


# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[194]:


# lets plot accuracy senstivity and specificity for various probabilities 

cutoff_df.plot.line (x='prob', y= ['accuracy','sensi','speci'])
plt.show()

# From the curev above the optimum point is 0.35 as cutoff
# In[195]:


y_train_pred_final['final_predicted']= y_train_pred_final.Converted_prob.map(lambda x : 1 if x >0.35 else 0)
y_train_pred_final.head()


# # Try to lead score with trianing data 

# In[197]:


y_train_pred_final['Lead_Score']= y_train_pred_final.Converted_prob.map(lambda x : round(x*100))
y_train_pred_final.head()


# In[ ]:


### with uppdated cut off point


# In[198]:


# Let's check the overall accuracy.
print("Accuracy :",metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))


# In[199]:


# Confusion matrix
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[200]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[201]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity : ",TP / float(TP+FN))


# In[202]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[203]:


# Calculate false postive rate - predicting converted lead when the lead was actually not have converted
print("False Positive rate : ",FP/ float(TN+FP))


# In[204]:


# Positive predictive value 
print("Positive Predictive Value :",TP / float(TP+FP))


# In[205]:


# Negative predictive value
print("Negative Predictive Value : ",TN / float(TN+ FN))


# # Check for confusion matrix again

# In[206]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted)
confusion


# In[208]:


# Precision 
TP/TP+FP
print("Precision : ", confusion [1,1]/(confusion[0,1]+confusion[1,1]))


# In[209]:


## Recall
TP/TP+FN

print("Recall : ", confusion [1,1]/(confusion[1,0]+confusion[1,1]))


# In[212]:


# Using sklearn.metrics 


# In[213]:


print("Precision :", precision_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[214]:


print("Recall :", recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[222]:


p,r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
plt.plot(thresholds, p[:-1],"g-")
plt.plot(thresholds, r[:-1],"r-")
plt.show()


# In[ ]:





# In[ ]:





# # Making predictions for test set

# In[ ]:


# first we need to scale the test set


# In[224]:


X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]= scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])


# In[225]:


X_test =X_test[col1]
X_test.head()


# In[227]:


## Add a constant

X_test_sm =sm.add_constant(X_test)

y_test_pred =result.predict(X_test_sm)
y_test_pred 


# In[228]:


#convert Y-test to a dataFrame

y_pred_1=pd.DataFrame(y_test_pred)
y_pred_1


# In[229]:


y_test_df =pd.DataFrame(y_test)


# In[230]:


# Adding Prospect ID as well

y_test_df['Prospect ID']= y_test_df.index


# In[233]:


## append both data frame

y_pred_1.reset_index(drop=True, inplace =True)

y_test_df.reset_index(drop=True, inplace =True)

y_pred_final =pd.concat([y_test_df, y_pred_1], axis=1)


# In[234]:


y_pred_final


# In[235]:


# update the names and arrnage them properly 

y_pred_final=y_pred_final.rename(columns ={0: "Converted_prob"})
y_pred_final =y_pred_final .reindex (columns =["Prospect ID", 'Converted', 'Converted_prob'])
y_pred_final


# In[236]:


y_pred_final['final_predicted']=y_pred_final.Converted_prob.map(lambda x : 1 if x >0.35 else 0)
y_pred_final


# In[242]:


# checking the over all accuracy once again

accuracy = metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
print("Accuracy:", round(accuracy, 2))


# In[243]:


# checking confusion matrix once again

Confusion2=metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted)
Confusion2


# In[244]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# # Now finally check the senstivity and specificity finally 
# 
# 

# In[253]:



sensitivity = TP / float(TP + FN)
print("Sensitivity:", round(sensitivity, 5))

specificity = TN / float(TN + FP)
print("Specificity:", round(specificity, 5))


# In[255]:


## Assigining lead Score to the testing data

y_pred_final['Lead_Score']= y_pred_final.Converted_prob.map(lambda x : round(x*100))


# In[256]:


y_pred_final


# # Final Observations 
#                 Train data.       Test Data
# Accuracy    :    80%                  81%
# Sentivity   :    80%                  80%
# Specificity :    81%                  81%

# In[ ]:


# Finally our main objective to find out which leads are hot lead


# In[258]:


hot_leads = y_pred_final[y_pred_final["Lead_Score"] >= 85]
hot_leads


# In[ ]:


# 406 leads are hot leads, means have higher chances of conversion


# In[259]:


# Model 9 Important features

result.params.sort_values(ascending =False)


# # Recommendations
# 
# 1. **Lead Origin_Lead Add Form (3.250480)**: Prioritize leads generated from the "Lead Add Form" as they have a significantly higher conversion potential.
# 
# 2. **What is your current occupation_Working Professional (2.708639)**: Focus marketing efforts on working professionals, as they exhibit a high likelihood of conversion.
# 
# 3. **Lead Source_Welingak Website (2.508787)**: Allocate resources to leads originating from the "Welingak Website," as they show promise in conversion.
# 
# 4. **Last Activity_Others (2.219725)**: Investigate and engage with leads categorized under "Others" for potential conversion opportunities.
# 
# 5. **Last Activity_SMS Sent (1.697341)**: Continue sending SMS messages, as it has a positive impact on lead conversion.
# 
# 6. **Total Time Spent on Website (1.093062)**: Encourage leads to spend more time on your website, as it correlates positively with conversion.
# 
# 7. **Lead Source_Olark Chat (1.091834)**: Pay attention to leads coming from "Olark Chat" and engage in active chat conversations to boost conversions.
# 
# 8. **Last Activity_Email Opened (0.515497)**: Focus on crafting compelling email content that encourages lead engagement.
# 
# 9. **const (-0.808670)**: Consider adjusting your model's intercept or baseline, as it negatively affects the conversion prediction.
# 
# 10. **Lead Origin_Landing Page Submission (-1.078273)**: Optimize landing pages to improve conversion rates for leads from this source.
# 
# 11. **Specialization_others (-1.095234)**: Explore ways to engage leads with "Other" specializations to increase conversion rates.
# 
# 12. **Last Activity_Olark Chat Conversation (-1.168577)**: Proactively engage with leads in "Olark Chat Conversations" to address concerns and improve conversion chances.
# 
# 13. **Do Not Email (-1.2680)**: Be cautious when sending emails to leads who have opted not to receive emails, as it negatively impacts conversion rates.
# 
