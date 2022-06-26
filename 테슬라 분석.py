#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import numpy as np


# In[2]:


df = yf.download(['TSLA','CAT','AAPL'],start="2012-12-01",end="2022-06-25")


# In[3]:


df = np.log(1+ df['Adj Close'].pct_change())


# In[4]:


df


# In[5]:


weights = [0.5,0.5]


# In[6]:


weights[0]*df.TSLA.mean() + weights[1]*df.CAT.mean()


# In[7]:


def portfolioreturn(weights):
    return np.dot(df.mean(),weights)


# In[21]:


portfolioreturn(weights)


# In[9]:


df.cov()


# In[10]:


pv = weights[0]**2*df.cov().iloc[0,0]+weights[1]**2*df.cov().iloc[1,1]+weights[0]*weights[1]*df.cov().iloc[0,1]


# In[11]:


pv


# In[12]:


pv**(1/2)


# In[13]:


def portfoliostd(weights):
    return (np.dot(np.dot(df.cov(),weights),weights))**(1/2)*np.sqrt(250)


# In[22]:


portfoliostd(weights)


# In[15]:


def weightscreator(df):
    rand = np.random.random(len(df.columns))
    rand /= rand.sum()
    return rand


# In[16]:


weightscreator(df)


# In[17]:


returns = []
stds = []
w = []

for i in range(500):
    weights = weightscreator(df)
    returns.append(portfolioreturn(weights))
    stds.append(portfoliostd(weights))
    w.append(weights)


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


plt.scatter(stds,returns)
plt.scatter(min(stds), returns[stds.index(min(stds))], c='green')
plt.title("Efficient frontier")
plt.xlabel("portfoliostd")
plt.ylabel("portfolioreturn")
plt.show()


# In[20]:


returns[stds.index(min(stds))]


# In[ ]:




