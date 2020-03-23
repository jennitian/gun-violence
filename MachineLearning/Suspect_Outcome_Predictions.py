#!/usr/bin/env python
# coding: utf-8

# In[26]:


# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from imblearn.over_sampling import RandomOverSampler
from collections import Counter, OrderedDict
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


# import transformed suspects dataframe
suspects_df = pd.read_sql_table('suspects_ml_transformed', engine)
suspects_df.head()


# In[6]:


# import incidents dataframe
incidents_df = pd.read_sql_table('incidents', engine)
incidents_df.head()


# # Predicting suspect outcome
# 
# - Again, have labeled dataset --> implement a supervised learning algorithm
# - Have multimple outcomes, start with SVM algorithm 
# - Explore Random Forest Classifier
# - Explore Neural Network
# - Compare tradeoffs & advantages/disadvantages of each algorithm

# ## Preprocess Data

# In[39]:


# combine suspect and incident data 
suspects_incidents_df = suspects_df.merge(incidents_df, how='left', on='incident_id')
suspects_incidents_df.info()


# In[40]:


# group into 3 statuses, Arrested, Killed, Other
suspects_incidents_df['status_Arrested'] = (suspects_incidents_df['status_Injured, Arrested'] +
                                            suspects_incidents_df['status_Unharmed, Arrested'] +
                                            suspects_incidents_df['status_Arrested'])
suspects_incidents_df['status_Other'] = (suspects_incidents_df['status_Injured'] +
                                        suspects_incidents_df['status_Unharmed'])
suspects_incidents_df.head()


# In[41]:


# define encoding meanings
status_defs = {'status_Arrested': 0, 'status_Killed': 1, 'status_Other': 2, 'status_Unknown': 3}


# In[42]:


# consolidate labels into one encoded column
suspects_incidents_df['status'] = [status_defs['status_Arrested'] if x == 1 
                                   else status_defs['status_Killed'] if y == 1 
                                   else status_defs['status_Other'] if z == 1 
                                   else status_defs['status_Unknown'] for (x, y, z) 
                                   in zip(suspects_incidents_df['status_Arrested'], suspects_incidents_df['status_Killed'], suspects_incidents_df['status_Other'])]
suspects_incidents_df.head(3)


# In[43]:


# inspect age groups
print('Child: \t', sorted(Counter(suspects_incidents_df['Child_0-11']).items()))
print('Teen: \t', sorted(Counter(suspects_incidents_df['Teen_12-17']).items()))
print('Adult: \t', sorted(Counter(suspects_incidents_df['Adult_18+']).items()))


# In[44]:


# drop child rows (only 578 entries)
suspects_incidents_df = suspects_incidents_df[suspects_incidents_df['Child_0-11'] != 1]


# In[45]:


# drop unnecessary columns
suspects_incidents_df = suspects_incidents_df.drop(columns=['incident_id', 'index', 'Adult_18+', 'Child_0-11', 'Teen_12-17', 'status_Injured', 
                                                            'status_Injured, Arrested', 'status_Unharmed, Arrested', 'status_Unharmed', 
                                                            'status_Arrested', 'status_Killed', 'status_Other', 'date', 
                                                            'state', 'latitude', 'longitude', 'incident_characteristics', 'notes'])
suspects_incidents_df.head()


# In[46]:


# inspect spread of targets
suspects_incidents_df['status'].value_counts()


# In[47]:


# drop columns with unknown status
suspects_incidents_df = suspects_incidents_df[suspects_incidents_df['status'] != status_defs['status_Unknown']]


# In[48]:


# drop NA columns
suspects_incidents_df = suspects_incidents_df.dropna()
suspects_incidents_df.info()


# In[49]:


# calculate IGR
Q1 = suspects_incidents_df.quantile(0.25)
Q3 = suspects_incidents_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[50]:


suspects_incidents_df['participant_age'].describe()


# In[51]:


# remove outlier ages
suspects_incidents_df = suspects_incidents_df[(suspects_incidents_df['participant_age'] < 100.0) & (suspects_incidents_df['participant_age'] > 11.0)]


# In[121]:


# separate features and target
y = suspects_incidents_df['status']
X = suspects_incidents_df.drop(columns=['status'])


# In[122]:


# plot target data
suspects_incidents_df.plot.scatter(x='female', y='participant_age', c='status', colormap='cool', figsize=(10,6))


# In[123]:


# split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
Counter(y_train)


# In[124]:


# implement random oversampling
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)


# In[125]:


# create scaler for disctric columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fit the scaler
X_scaler = scaler.fit(X_resampled)


# In[126]:


# scale the data
X_resampled_scaled = X_scaler.transform(X_resampled)
X_test_scaled = X_scaler.transform(X_test)


# ### Multinomial Logistic Regression Algorithm

# In[70]:


# import logistic regression algorithm
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=200, random_state=1)


# In[75]:


# fit model using training data
classifier.fit(X_resampled_scaled, y_resampled)


# In[76]:


# make predictions
y_pred = classifier.predict(X_test_scaled)


# In[77]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[78]:


# print balanced accuracy score
print(balanced_accuracy_score(y_test, y_pred))


# In[79]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### LinearSVC Model

# In[82]:


# create linear SVM model
from sklearn.svm import LinearSVC
model = LinearSVC(random_state=1)


# In[83]:


# fit the data
model.fit(X_resampled_scaled, y_resampled)


# In[84]:


# make predictions
y_pred = model.predict(X_test_scaled)


# In[85]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[86]:


# print balanced accuracy score
print(balanced_accuracy_score(y_test, y_pred))


# In[87]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### Random Forest Classifier

# In[88]:


# define model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=1)


# In[94]:


rf_model.fit(X_resampled, y_resampled)


# In[95]:


# get predictions
y_pred = rf_model.predict(X_test)


# In[96]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[97]:


# print balanced accuracy score
print(accuracy_score(y_test, y_pred))


# In[98]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### Deep NN

# In[127]:


# set up model 
import tensorflow as tf
num_input_features = len(X_train.iloc[0])
hidden_nodes_layer1 = 14
hidden_nodes_layer2 = 6
nn = tf.keras.models.Sequential()


# In[128]:


# build model
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=num_input_features, activation='relu'))
#nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation='relu'))
nn.add(tf.keras.layers.Dense(units=3, activation='softmax'))
nn.summary()


# In[129]:


# compile the model
nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[133]:


# train model
fit_nn = nn.fit(X_train, y_train, epochs=100)


# In[134]:


# evaluate model
model_loss, model_accuracy = nn.evaluate(X_test, y_test, verbose=2)
print(f'Loss: {model_loss}, Accuracy: {model_accuracy}')


# In[ ]:




