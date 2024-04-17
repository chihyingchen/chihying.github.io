#!/usr/bin/env python
# coding: utf-8

# # "Exploring Insights from Social Media Consumer Buying Behavior Data"

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


import os 
for dirname, _, filenames in os.walk('social_ads.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load the dataset

# In[8]:


df= pd.read_csv("social_ads.csv")
df.head()


# In[9]:


df.info()


# In[10]:


#Check for missing values
df.isnull().sum()


# In[11]:


#Summery statistics
df.describe()


# # Data visualization

# In[12]:


sns.pairplot(df, hue='Purchased')
plt.show()


# In[18]:


#Boxplot to visualize age & estimated salary distribution
plt.figure(figsize=(8,6))
sns.boxplot(x='Purchased', y='Age', data=df)
plt.title('Age Distribution by Purchase')
plt.show()


# In[19]:


plt.figure(figsize=(8,6))
sns.boxplot(x='Purchased', y='EstimatedSalary', data=df)
plt.title('Estimated Salary Distribution by Purchase')
plt.show()


# In[22]:


#Correlation heatmap

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),annot=True,cmap='Greens')
plt.title('Correlation Heatmap')
plt.show()


# # Prediction of purchase based on age & estimated salary

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[27]:


x= df[['Age','EstimatedSalary']]
y= df['Purchased']


# In[29]:


X_train, X_test, y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=40)


# In[31]:


model=LogisticRegression()
model.fit(X_train, y_train)


# In[32]:


y_pred=model.predict(X_test)


# In[33]:


print(classification_report(y_test,y_pred))


# In[35]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[36]:


models={'Logistic Regression': LogisticRegression(),
       'Random Forest':RandomForestClassifier(),
       'Gradient Boosting': GradientBoostingClassifier()}


# In[38]:


params={'Logistic Regression':{'C':[0.1,1,10]},
        'Random Forest':{'n_estimators':[50,100,200],'max_depth':[3,5,7]},
        'Gradient Boosting':{'n_estimators':[50,100,200],
        'learning_rate':[0.01,0.1,0.5]}
}


# In[39]:


#GridSearchCV to find the best model and hyperparameters

best_models={}
for name, model in models.items():
    grid_search=GridSearchCV(model, params[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[name]=grid_search.best_estimator_
    print(f"Best parameters for {name}:{grid_search.best_params_}")


# In[40]:


#Evaluate the best models
for name, model in best_models.items():
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}:{accuracy}")


# In[ ]:




