#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dependencies
import pandas as pd
import numpy as np
import matplotlib as plt
from sqlalchemy import create_engine

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# import db password
from config import db_password


# In[3]:


# credentials for connecting to Postgres db
POSTGRES_ADDRESS = 'bootcamp-final-project.c8u2worjd1ui.us-east-1.rds.amazonaws.com'
POSTGRES_PORT = 5432
POSTGRES_USERNAME = 'peter_jennifer'
POSTGRES_PASSWORD = db_password
POSTGRES_DBNAME = 'us_gun_violence'


# In[4]:


# creat connection string and database engine
db_string = f'postgres://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_ADDRESS}:{POSTGRES_PORT}/{POSTGRES_DBNAME}'

engine = create_engine(db_string)


# In[5]:


# import transformed guns dataframe
guns_df = pd.read_sql_table('guns_ml_transformed', engine)
guns_df.head()


# In[6]:


# import incidents dataframe
incidents_df = pd.read_sql_table('incidents', engine)
incidents_df.head()


# # Predicting stolen/not-stolen gun acquisition
# 
# - Utilize a supervised learning method because dataset contains labels
# - Start by implementing a logistic regression algorithm
# - Explore a SVM algorithm
# - Explore a Random Forest Classifier 
# - Compare tradeoffs with each algorithm

# ## Preprocess data

# In[7]:


# explore data
guns_df.info()


# In[8]:


# extract rows that contain information about stolen status
guns_stolen_df = guns_df.loc[(guns_df['not_stolen'] == 1) | (guns_df['stolen'] == 1)]
guns_stolen_df.info()


# In[26]:


# merge stolen guns df with incidents
guns_incidents_df = guns_stolen_df.merge(incidents_df, how='left', on='incident_id')
guns_incidents_df.head()


# In[30]:


# drop unnecesary columns
guns_incidents_df = guns_incidents_df.drop(columns=['index', 'incident_id', 'not_stolen', 'date', 'state', 
                                                    'latitude', 'longitude', 'incident_characteristics', 'notes'])
guns_incidents_df = guns_incidents_df.dropna()
guns_incidents_df.head()


# In[31]:


# calculate IGR
Q1 = guns_incidents_df.quantile(0.25)
Q3 = guns_incidents_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[35]:


# explore district columns for potential outlier + to visualize value spread
guns_incidents_df.boxplot(column=['congressional_district', 'state_house_district', 'state_senate_district'])


# In[41]:


# separate features and target
y = guns_incidents_df['stolen']
X = guns_incidents_df.drop(columns=['stolen'])


# In[42]:


# split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
Counter(y_train)


# In[43]:


# implement random oversampling
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)


# In[44]:


# create scaler for disctric columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fit the scaler
X_scaler = scaler.fit(X_resampled)


# In[45]:


# scale the data
X_resampled_scaled = X_scaler.transform(X_resampled)
X_test_scaled = X_scaler.transform(X_test)


# ### Logistic Regression Algorithm

# In[46]:


# import logistic regression algorithm
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs',
                                max_iter=200,
                                random_state=1)


# In[47]:


# fit model using training data
classifier.fit(X_resampled_scaled, y_resampled)


# In[48]:


# make predictions
y_pred = classifier.predict(X_test_scaled)


# In[49]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[50]:


# print balanced accuracy score
print(balanced_accuracy_score(y_test, y_pred))


# In[51]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### SVM Algorithm

# In[52]:


# create linear SVM model
from sklearn.svm import SVC
model = SVC(kernel='linear', random_state=1)


# In[53]:


# fit the data
model.fit(X_resampled_scaled, y_resampled)


# In[58]:


# make predictions
y_pred = model.predict(X_test_scaled)


# In[59]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[60]:


# print balanced accuracy score
print(balanced_accuracy_score(y_test, y_pred))


# In[61]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### Random Forest Classifier

# In[62]:


# create random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=1)


# In[64]:


# fit the model
rf_model = rf_model.fit(X_resampled_scaled, y_resampled)


# In[66]:


# make predictions
y_pred = rf_model.predict(X_test_scaled)


# In[67]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[68]:


# print balanced accuracy score
print(balanced_accuracy_score(y_test, y_pred))


# In[69]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### Deep NN

# In[70]:


# set up model 
import tensorflow as tf
num_input_features = len(X_resampled.iloc[0])
hidden_nodes_layer1 = 10
hidden_nodes_layer2 = 6
nn = tf.keras.models.Sequential()


# In[71]:


# build model
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=num_input_features, activation='relu'))
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation='relu'))
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
nn.summary()


# In[72]:


# compile the model
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[76]:


# train model
fit_nn = nn.fit(X_resampled_scaled, y_resampled.to_numpy(), epochs=100)


# In[83]:


# evaluate model
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test.to_numpy(), verbose=2)
print(f'Loss: {model_loss}, Accuracy: {model_accuracy}')


# ## Algorithm Comparison

# # Predicting suspect outcome
# 
# - Again, have labeled dataset --> implement a supervised learning algorithm
# - Have multimple outcomes, start with SVM algorithm 
# - Explore Random Forest Classifier
# - Explore Neural Network
# - Compare tradeoffs & advantages/disadvantages of each algorithm

# ## Preprocess Data

# In[ ]:


# import transformed suspects dataframe
suspects_df = pd.read_sql_table('suspects_ml_transformed', engine)
suspects_df.head()


# In[ ]:


# add incident data 
suspects_incidents_df = suspects_df.merge(incidents_df, how='left', on='incident_id')
suspects_incidents_df.info()


# In[ ]:


# group into 3 statuses, Arrested, Killed, Other
suspects_incidents_df['status_Arrested'] = (suspects_incidents_df['status_Injured, Arrested'] +
                                            suspects_incidents_df['status_Unharmed, Arrested'] +
                                            suspects_incidents_df['status_Arrested'])
suspects_incidents_df['status_Other'] = (suspects_incidents_df['status_Injured'] +
                                        suspects_incidents_df['status_Unharmed'])
suspects_incidents_df.head()


# In[ ]:


# drop unnecessary columns
suspects_incidents_df = suspects_incidents_df.drop(columns=['index', 'Adult_18+', 'Child_0-11', 'Teen_12-17', 'status_Injured', 'status_Injured, Arrested',
                                                            'status_Unharmed, Arrested', 'status_Unharmed', 'date', 'state', 'latitude', 'longitude', 
                                                            'incident_characteristics', 'notes'])
suspects_incidents_df.head()


# In[ ]:


# drop NA columns
suspects_incidents_df = suspects_incidents_df.dropna()
suspects_incidents_df.info()


# In[ ]:


# separate features and target
y = suspects_incidents_df[['status_Arrested', 'status_Killed', 'status_Other']]
X = suspects_incidents_df.drop(columns=['status_Arrested', 'status_Killed', 'status_Other'])


# In[ ]:


for i, col in enumerate(y.columns.tolist(), 1):
    y.loc[:, col] *= i
y = y.sum(axis=1)


# In[ ]:


# split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
Counter(y_train)


# ### Random Forest Classifier

# In[ ]:


# define model
rf_model = RandomForestClassifier(n_estimators=200, random_state=1)


# In[ ]:


rf_model.fit(X_train, y_train)


# In[ ]:


# get predictions
y_pred = rf_model.predict(X_test)


# In[ ]:





# ### Neural Network

# ## Algorithm Comparison
