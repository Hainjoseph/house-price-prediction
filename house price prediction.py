#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")                          #assigning the datasets
test = pd.read_csv("test.csv")

train = train.drop(["Unnamed: 0", "Id"], axis = 1)        #droping unwanted values
test = test.drop(["Unnamed: 0", "Id"], axis = 1)

train_data = train.values
Y = train_data[:, -1].reshape(train_data.shape[0], 1)     #in the dataset there is a feature named price and thats our 
X = train_data[:, :-1]                                    #prediction(Y) for that we are seperating X and Y


# In[3]:


test_data = test.values
Y_test = test_data[:, -1].reshape(test_data.shape[0], 1)  
X_test = test_data[:, :-1]


# In[6]:


X.shape                                                    


# In[7]:


Y.shape


# In[8]:


X_test.shape


# In[10]:


Y_test.shape


# In[11]:


X = np.vstack((np.ones((X.shape[0], )), X.T)).T                 #creating a coloumn of ones before the features of x 
X_test = np.vstack((np.ones((X_test.shape[0], )), X_test.T)).T  #for theta[0]  


# In[35]:


def model(X,Y,learnig_rate,iteration):
    m = Y.size
    theta = np.zeros((X.shape[1],1))
    cost_list = []
    for i in range(iteration):
        y_prediction = np.dot(X,theta)
        cost = (np.sum(np.square(y_prediction-Y)))/(2*m)
        d_theta = (np.dot(X.T,(y_prediction-Y)))/m
        theta = theta-(learning_rate*d_theta)
        cost_list.append(cost)
        
        if(i%500==0):
            print("cost is: ",cost)
            error = (1/X.shape[0])*np.sum(np.abs(y_prediction - Y))
            print("Train Accuracy :", (1- error)*100, "%")
            
    return theta,cost_list


# In[36]:


iteration = 10000
learning_rate = 0.000000005   
theta, cost_list = model(X, Y, learning_rate, iteration)  #let's train our model


# In[46]:


#let's plot a graph of cost function
rng = np.arange(0, iteration)
plt.plot(rng, cost_list)
plt.show()


# In[37]:


#now,lets test our model with test dataset and find out the accuracy

y_prediction = np.dot(X_test, theta)
error = (1/X_test.shape[0])*np.sum(np.abs(y_prediction - Y_test))
print("Test Accuracy is :", (1- error)*100, "%")


# In[ ]:




