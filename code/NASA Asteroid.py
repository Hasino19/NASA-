#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" 

This notebook is meant to:
 1. load the NASA Asteroid dataset,
 
 2. do some data cleaning and preparation,
 
 3. build a machine learning model, 
 
 4. use the model to solve a binary classification problem,
    namely, to classify whether asteriods are hazardeous or not.
    
"""


# ### 0. Import necessary libraries

# In[1]:


import pandas as pd              # to handle data frames
import numpy as np               # to manipulate the data and perform any mathematical operations needed
import matplotlib.pyplot as plt  # to build visuals
import seaborn as sns            # to help plotting heatmaps


# In[ ]:





# ### 1. Load the dataset

# In[2]:


# path_to_data = 'path/to/your/dataset'
path_to_data = "C:/Users/Hussein/Desktop/nasa.csv"

data = pd.read_csv(path_to_data)


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[ ]:





# ### 2. Clean and prepare the dataset:

# #### 2.1 Drop unnecessary columns/features that aren't going to contribute to the learning process

# In[6]:


data = data.drop(['Neo Reference ID',
                  'Name',
                  'Orbit ID', 
                  'Close Approach Date',
                  'Epoch Date Close Approach',
                  'Orbit Determination Date', 
                  'Orbiting Body',
                  'Equinox'], axis= 1)

# print shape to confirm the success of column dropping
data.shape


# In[ ]:





# #### 2.2 Prepare the features: one-hot encoding of the labels

# In[7]:


hazardous_labels = pd.get_dummies(data['Hazardous'], dtype=int) 
# dtype=int to enforce get_dummies to return integers not boolean 
hazardous_labels


# In[8]:


data = pd.concat([data, hazardous_labels], axis = 1)
data.head()


# In[9]:


data = data.drop(['Hazardous'], axis = 1)
data.head()


# In[ ]:





# #### Plot a correlation heatmap to see the relationship between columns in the dataset

# In[10]:


plt.figure(figsize = (20,20))
sns.heatmap(data.corr(),annot = True)


# #### Some columns scored a correlation of 1. That means they are higly similar and therefore can be dropped

# In[11]:


data = data.drop(['Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)'
             ,'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 
             'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)', 
             'Miss Dist.(kilometers)', 'Miss Dist.(miles)', False], axis = 1)
data.shape


# In[12]:


data.head()


# In[ ]:





# ## 3. Build A Model

# In[13]:


### Split the label column from the other columns for features
features = data.drop([True], axis = 1)
labels = data[True].astype(int)


# In[14]:


### split the dataset into two partitions: train 70% and test 30%
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            random_state = 0 , test_size = 0.3)


# In[15]:


### make an instance from the XGBClassifier
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance

xbg_classifier = XGBClassifier()


# In[ ]:


""" 
The XGBoost classifier (XGBClassifier) is a powerful and efficient machine learning model based on the gradient boosting 
framework. It is widely used for classification tasks due to its ability to handle large datasets, and provide high accuracy.
"""


# ## 4.1 Training

# In[16]:


xbg_classifier.fit(features_train, labels_train)


# In[17]:


plot_importance(xbg_classifier, grid=False)
pyplot.show()


# In[ ]:





# ### 4.2 Test and report accuracy

# In[18]:


from sklearn.metrics import accuracy_score

predictions = xbg_classifier.predict(features_test)
acc = accuracy_score(labels_test, predictions)
print(str(np.round(acc*100, 2))+'%')


# In[ ]:




