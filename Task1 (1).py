#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#reading data from the given link
url= "http://bit.ly/w-data"


# In[6]:


s_data= pd.read_csv(url)


# In[7]:


s_data.head()


# In[8]:


s_data.describe()


# In[9]:


cdf=s_data[['Hours','Scores']]
cdf.head(9)


# In[10]:


cdf.hist()
plt.show()


# In[11]:


plt.scatter(cdf.Hours,cdf.Scores,color="red")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


# In[13]:


#training data
msk = np.random.rand(len(s_data)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[14]:


plt.scatter(train.Hours,train.Scores,color="blue")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


# In[15]:


#training algorithm
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Hours']])
train_y = np.asanyarray(train[['Scores']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[16]:


#plotting the regression line
plt.scatter(train.Hours, train.Scores,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Hours")
plt.ylabel("Scores")


# In[17]:


#evaluating the model
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['Hours']])
test_y = np.asanyarray(test[['Scores']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


# In[21]:


test_y_hat


# In[22]:


test_y


# In[23]:


test_x


# In[32]:


Hours=9.5
test_x = np.asanyarray([Hours])
test_x = test_x.reshape(-1, 1)
test_y_hat = regr.predict(test_x)


# In[33]:


test_y_hat


# In[ ]:




