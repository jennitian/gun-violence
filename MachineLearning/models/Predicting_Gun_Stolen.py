#!/usr/bin/env python
# coding: utf-8

# # Predicting stolen/not-stolen gun acquisition

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


# ## Preprocess data

# In[7]:


# explore data
guns_df.info()


# In[8]:


# extract rows that contain information about stolen status
guns_stolen_df = guns_df.loc[(guns_df['not_stolen'] + guns_df['stolen']) == 1]
guns_stolen_df.info()


# In[9]:


# merge stolen guns df with incidents
guns_incidents_df = guns_stolen_df.merge(incidents_df, how='left', on='incident_id')
guns_incidents_df.head()


# In[10]:


# define encodings for states
states_dict = {
        'Alaska': 1,
        'Alabama': 2,
        'Arkansas': 3,
        'Arizona': 4,
        'California': 5,
        'Colorado': 6,
        'Connecticut': 7,
        'District of Columbia': 8,
        'Delaware': 9,
        'Florida': 10,
        'Georgia': 11,
        'Hawaii': 12,
        'Iowa': 13,
        'Idaho': 14,
        'Illinois': 15,
        'Indiana': 16,
        'Kansas': 17,
        'Kentucky': 18,
        'Louisiana': 19,
        'Massachusetts': 20,
        'Maryland': 21,
        'Maine': 22,
        'Michigan': 23,
        'Minnesota': 24,
        'Missouri': 25,
        'Mississippi': 26,
        'Montana': 27,
        'North Carolina': 28,
        'North Dakota': 29,
        'Nebraska': 30,
        'New Hampshire': 31,
        'New Jersey': 32,
        'New Mexico': 33,
        'Nevada': 34,
        'New York': 35,
        'Ohio': 36,
        'Oklahoma': 37,
        'Oregon': 38,
        'Pennsylvania': 39,
        'Rhode Island': 40,
        'South Carolina': 41,
        'South Dakota': 42,
        'Tennessee': 43,
        'Texas': 44,
        'Utah': 45,
        'Virginia': 46,
        'Vermont': 47,
        'Washington': 48,
        'Wisconsin': 49,
        'West Virginia': 50,
        'Wyoming': 51
}


# In[11]:


# use states dictionary to encode dataframe
guns_incidents_df['state_num'] = guns_incidents_df['state'].apply(lambda x: states_dict[x])
guns_incidents_df.head(3)


# In[12]:


# drop unnecesary columns
guns_incidents_df = guns_incidents_df.drop(columns=['index', 'incident_id', 'not_stolen', 'date', 'state', 
                                                    'latitude', 'longitude', 'incident_characteristics', 'notes'])
guns_incidents_df = guns_incidents_df.dropna()
guns_incidents_df.head()


# In[13]:


# calculate IGR
Q1 = guns_incidents_df.quantile(0.25)
Q3 = guns_incidents_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[14]:


# explore district columns for potential outlier + to visualize value spread
guns_incidents_df.boxplot(column=['congressional_district', 'state_house_district', 'state_senate_district'])


# In[15]:


# drop state_house_district, too large a spread, difficult to discern outliers
guns_incidents_df = guns_incidents_df.drop(columns=['state_house_district'])


# In[16]:


# separate features and target
y = guns_incidents_df['stolen']
X = guns_incidents_df.drop(columns=['stolen'])


# In[17]:


# split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
Counter(y_train)


# In[18]:


# implement random oversampling
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)


# In[19]:


# create scaler for disctric columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fit the scaler
X_scaler = scaler.fit(X_resampled)


# In[20]:


# scale the data
X_resampled_scaled = X_scaler.transform(X_resampled)
X_test_scaled = X_scaler.transform(X_test)


# ### Logistic Regression Algorithm

# In[21]:


# import logistic regression algorithm
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs',
                                max_iter=200,
                                random_state=1)


# In[22]:


# fit model using training data
classifier.fit(X_resampled_scaled, y_resampled)


# In[23]:


# make predictions
y_pred = classifier.predict(X_test_scaled)


# In[24]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[25]:


# print balanced accuracy score
print(balanced_accuracy_score(y_test, y_pred))


# In[26]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### SVM Algorithm

# In[27]:


# create linear SVM model
from sklearn.svm import SVC
model = SVC(kernel='linear', random_state=1)


# In[28]:


# fit the data
model.fit(X_resampled_scaled, y_resampled)


# In[29]:


# make predictions
y_pred = model.predict(X_test_scaled)


# In[30]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[31]:


# print balanced accuracy score
print(balanced_accuracy_score(y_test, y_pred))


# In[32]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### Random Forest Classifier

# In[33]:


# create random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=1)


# In[34]:


# fit the model
rf_model = rf_model.fit(X_resampled_scaled, y_resampled)


# In[35]:


# make predictions
y_pred = rf_model.predict(X_test_scaled)


# In[36]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[37]:


# print balanced accuracy score
print(balanced_accuracy_score(y_test, y_pred))


# In[38]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### Deep NN

# In[39]:


# set up model 
import tensorflow as tf
num_input_features = len(X_resampled.iloc[0])
hidden_nodes_layer1 = 20
hidden_nodes_layer2 = 5
nn = tf.keras.models.Sequential()


# In[40]:


# build model
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=num_input_features, activation='relu'))
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation='relu'))
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
nn.summary()


# In[41]:


# compile the model
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[42]:


# train model
fit_nn = nn.fit(X_train, y_train, epochs=100)


# In[43]:


# store predictions
y_pred = nn.predict(X_test)


# In[44]:


# evaluate model
model_loss, model_accuracy = nn.evaluate(X_test, y_test, verbose=2)
print(f'Loss: {model_loss}, Accuracy: {model_accuracy}')


# In[45]:


# print confusion matrix
print(tf.math.confusion_matrix(y_test, y_pred))


# In[46]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred > 0.5))


# In[ ]:




