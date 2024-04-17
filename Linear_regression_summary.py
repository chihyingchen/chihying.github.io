#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np


# In[13]:


score = [39,43,21,64,57,47,28,75,34,52]
grade = [65,78,52,82,92,89,73,98,56,75]


# In[14]:


data = {"SCORE" : score , "GRADE" : grade }


# In[15]:


#create DataFrame
df = pd.DataFrame(data)

#view first five rows of DataFrame
df


# In[16]:


X = df.drop("GRADE",axis =1 )
Y = df["GRADE"]


# In[17]:


#   y = a + b*X
from sklearn.linear_model import LinearRegression

#initiate linear regression model
model = LinearRegression()

#fit regression model
MODEL =model.fit(X, Y)
MODEL


# In[ ]:





# In[18]:


#display regression coefficients
print(MODEL.intercept_, MODEL.coef_)


# In[8]:


test_value = 35


# In[9]:


predicted = 40.78415521422798 + 0.76556184 * test_value
predicted


# In[10]:


predicted = MODEL.intercept_ + MODEL.coef_* test_value
predicted


# In[ ]:


### output for our example


# In[11]:


predicted1 = 10.9 + 0.23 * 73
predicted1


# In[ ]:


import pandas as pd


# In[ ]:


#create DataFrame
df = pd.DataFrame({'x1': [1, 2, 2, 4, 2, 1, 5, 4, 2, 4, 4],
                   'x2': [1, 3, 3, 5, 2, 2, 1, 1, 0, 3, 4],
                   'y': [76, 78, 85, 88, 72, 69, 94, 94, 88, 92, 90]})

#view first five rows of DataFrame
df.head()


# In[ ]:


#   y = a + b1*X1 + b2*X2
from sklearn.linear_model import LinearRegression

#initiate linear regression model
model = LinearRegression()

#define predictor and response variables

X  = df[['x1',"x2"]]
y = df["y"]
#fit regression model
model =model.fit(X, y)
model


# In[ ]:


#display regression coefficients and R-squared value of model
print(model.intercept_, model.coef_, model.score(X, y))


# In[ ]:


#   y = a + b*X   = 70.48 + 5.79 * x1 - 1.15*X2


# # Using this output, we can write the equation for the fitted regression model:
# 
# ## y = 70.48 + 5.79x1 – 1.16x2
# 
# #### We can also see that the R2 value of the model is 76.67.
# 
# ### This means that 76.67% of the variation in the response variable can be explained by the two predictor variables in the model.

# ## Method 2: Get Regression Model Summary from Statsmodels

# In[ ]:


import statsmodels.api as sm

#define response variable
y = df["y"]

#define predictor variables
x = df[["x1","x2"]]

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())


# In[ ]:





# In[ ]:




