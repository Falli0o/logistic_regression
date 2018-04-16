
# coding: utf-8

# In[1]:


from __future__ import division
get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


train_1 = np.loadtxt('Iris2D1_train.txt')
test_1 = np.loadtxt('Iris2D1_test.txt')
train_2 = np.loadtxt('Iris2D2_train.txt')
test_2 = np.loadtxt('Iris2D2_test.txt')


# In[3]:


fig = plt.figure()
plt.scatter(train_1[(train_1[:, 2] == 0)][:, 0], train_1[(train_1[:, 2] == 0)][:, 1],alpha = 1,marker = '^', c = 'green',label = 'Classes 0');
plt.scatter(train_1[(train_1[:, 2] == 1)][:, 0],train_1[(train_1[:, 2] == 1)][:, 1],alpha = 1,marker = 'v', c = 'pink',label = 'Classes 1');
plt.legend()
plt.title("Iris2D1_train");
plt.savefig("Iris2D1_train");
fig = plt.figure()
plt.scatter(train_2[(train_2[:, 2] == 0)][:, 0], train_2[(train_2[:, 2] == 0)][:, 1],alpha = 1,marker = '^',c = 'orange',label = 'Classes 0');
plt.scatter(train_2[(train_2[:, 2] == 1)][:, 0], train_2[(train_2[:, 2] == 1)][:, 1], alpha = 1,marker = 'v', c = 'blue',label = 'Classes 1');
plt.legend()
plt.title("Iris2D2_train");
plt.savefig("Iris2D2_train");
fig = plt.figure()
plt.scatter(test_1[(test_1[:, 2] == 0)][:, 0], test_1[(test_1[:, 2] == 0)][:, 1], alpha = 1, marker = '^', c = 'green', label = 'Classes 0');
plt.scatter(test_1[(test_1[:, 2] == 1)][:, 0], test_1[(test_1[:, 2] == 1)][:, 1], alpha = 1, marker = 'v', c = 'pink', label = 'Classes 1');
plt.legend()
plt.title("Iris2D1_test");
plt.savefig("Iris2D1_test");
fig = plt.figure()
plt.scatter(test_2[(test_2[:, 2] == 0)][:, 0], test_2[(test_2[:, 2] == 0)][:, 1], alpha = 1, marker = '^', c = 'orange',label = 'Classes 0');
plt.scatter(test_2[(test_2[:, 2] == 1)][:, 0], test_2[(test_2[:, 2] == 1)][:, 1], alpha = 1, marker = 'v', c = 'blue',label = 'Classes 1');
plt.legend()
plt.title("Iris2D2_test");
plt.savefig("Iris2D2_test");
#obeserve: the different classes have clusters and for train 2 has a clear decision boundary


# In[4]:


training_data_1 = train_1[:, :2]
training_label_1 = train_1[:, 2]
test_data_1 = test_1[:, :2]
test_label_1 = test_1[:, 2]

training_data_2 = train_2[:, :2]
training_label_2 = train_2[:, 2]
test_data_2 = test_2[:, :2]
test_label_2 = test_2[:, 2]


# In[5]:


"""
from sklearn import linear_model
logistic = linear_model.LogisticRegression()
logistic.fit(train_1[:, :2], train_1[:, 2]).get_params()
re = logistic.fit(train_1[:, :2], train_1[:, 2]).decision_function(test_1[:, :2])
logistic.fit(train_1[:, :2], train_1[:, 2]).score(test_1[:, :2], test_1[:, 2])
#logistic.fit(train_1[:, :2], train_1[:, 2]).predict(test_1[:, :2])
#logistic.fit(train_1[:, :2], train_1[:, 2]).get_params()
#logistic.fit(train_1[:, :2], train_1[:, 2])
#0.8333333333333334
"""


# In[6]:


"""
logistic = linear_model.LogisticRegression()
logistic.fit(train_2[:, :2], train_2[:, 2]).get_params()
re = logistic.fit(train_2[:, :2], train_2[:, 2]).decision_function(test_2[:, :2])
logistic.fit(train_2[:, :2], train_2[:, 2]).score(test_2[:, :2], test_2[:, 2])
#logistic.fit(train_2[:, :2], train_2[:, 2]).predict(test_2[:, :2])
#1.0
"""


# In[7]:


#theta: [w1, w2, w0] row wise
def calculate_gd(training_data, label, theta):
    theta = np.array(theta)
    new_label = label
    new_label[(label == 0)] = -1
    training_data = np.hstack((training_data, np.repeat(1, training_data.shape[0]).reshape(-1, 1)))
    length = training_data.shape[0]
    total = []
    for i in range(length):
        g = new_label[i]*training_data[i]/(1+np.exp(new_label[i]*np.dot(theta, training_data[i])))
        total.append(g)
    gd = - np.sum(total, axis = 0)/length
    return gd


# In[8]:


def lr_gd(training_data, label, test_data):
    theta = np.array([1,1,1])
    rate = 0.3
    iteration_time = 0
    while True:
        length = training_data.shape[0]
        gd = calculate_gd(training_data, label, theta)
        #print gd
        norm = np.linalg.norm(gd)
        #norm = np.sqrt(np.dot(gd.reshape(-1), gd.reshape(-1)))
        if norm < 10**(-10) or iteration_time > 10000:# or norm < 10**(-10):# or 
            print "gradient: " + str(gd)
            print "norm: " + str(norm)
            theta = theta.reshape(-1)
            print "the weight vector: " + str(theta)
            print ""
            Y = np.hstack((test_data, np.repeat(1, test_data.shape[0]).reshape(-1, 1)))
            #y = 1/(1+np.exp(-np.dot(theta, Y.T)))
            #print y
            z = np.dot(theta.T, Y.T)
            g = np.exp(z)/(1 + np.exp(z))#sigmoid function
            #print y
            #print g
            f = g
            f[(g > 0.5)] = 1
            f[(g < 0.5)] = 0
            print "iteration_time: " + str(iteration_time)
            print "predicted results for test data: "
            print f
            return f
            #return theta.reshape(-1)
            break
        else:
            theta = theta + rate*(-gd)
            iteration_time += 1


# In[9]:


#lr_gd(training_data_1, train_label_1, test_data_1)


# In[10]:


def zero_one_loss(training_data, label, test_data, ground_truth):
    predicted_value = lr_gd(training_data, label, test_data)
    ground_truth[(ground_truth == -1)] = 0
    print "0-1 loss: "
    return np.sum(ground_truth != predicted_value)


# In[11]:


training_label_1
#test_data_1.shape


# In[12]:


#for data 1
#training error
print zero_one_loss(training_data_1, training_label_1, training_data_1, training_label_1)
print ""
#test error
print zero_one_loss(training_data_1, training_label_1, test_data_1, test_label_1)


# In[13]:


#for data 1
#training error
print zero_one_loss(training_data_2, training_label_2, training_data_2, training_label_2)
print ""
#test error
print zero_one_loss(training_data_2, training_label_2, test_data_2, test_label_2)


# In[14]:


"""
from sklearn import linear_model
logistic = linear_model.LogisticRegression()
logistic.fit(train_1[:, :2], train_1[:, 2]).get_params()
re = logistic.fit(train_1[:, :2], train_1[:, 2]).decision_function(train_1[:, :2])
logistic.fit(train_1[:, :2], train_1[:, 2]).score(train_1[:, :2], train_1[:, 2])
#0.9285714285714286
"""


# In[15]:


"""
from sklearn import linear_model
logistic = linear_model.LogisticRegression()
logistic.fit(train_2[:, :2], train_2[:, 2]).get_params()
re = logistic.fit(train_2[:, :2], train_2[:, 2]).decision_function(train_2[:, :2])
logistic.fit(train_2[:, :2], train_2[:, 2]).score(train_2[:, :2], train_2[:, 2])
"""

