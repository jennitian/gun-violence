#!/usr/bin/env python
# coding: utf-8

# # Predicting suspect outcome (arrested, injured, killed)

# In[1]:


# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# import transformed suspects dataframe
suspects_df = pd.read_sql_table('suspects_ml_transformed', engine)
suspects_df.head()


# In[6]:


# import incidents dataframe
incidents_df = pd.read_sql_table('incidents', engine)
incidents_df.head()


# ## Preprocess Data

# In[7]:


# combine suspect and incident data 
suspects_incidents_df = suspects_df.merge(incidents_df, how='left', on='incident_id')
suspects_incidents_df.info()


# In[8]:


# group into 3 statuses, Arrested, Killed, Other
suspects_incidents_df['status_Arrested'] = (suspects_incidents_df['status_Injured, Arrested'] +
                                            suspects_incidents_df['status_Unharmed, Arrested'] +
                                            suspects_incidents_df['status_Arrested'])
suspects_incidents_df['status_Other'] = (suspects_incidents_df['status_Injured'] +
                                        suspects_incidents_df['status_Unharmed'])
suspects_incidents_df.head()


# In[9]:


# define encoding meanings
status_defs = {'status_Arrested': 0, 'status_Killed': 1, 'status_Other': 2, 'status_Unknown': 3}


# In[10]:


# consolidate labels into one encoded column
suspects_incidents_df['status'] = [status_defs['status_Arrested'] if x == 1 
                                   else status_defs['status_Killed'] if y == 1 
                                   else status_defs['status_Other'] if z == 1 
                                   else status_defs['status_Unknown'] for (x, y, z) 
                                   in zip(suspects_incidents_df['status_Arrested'], suspects_incidents_df['status_Killed'], suspects_incidents_df['status_Other'])]
suspects_incidents_df.head(3)


# In[11]:


# inspect age groups
print('Child: \t', sorted(Counter(suspects_incidents_df['Child_0-11']).items()))
print('Teen: \t', sorted(Counter(suspects_incidents_df['Teen_12-17']).items()))
print('Adult: \t', sorted(Counter(suspects_incidents_df['Adult_18+']).items()))


# In[12]:


# drop child rows (only 578 entries)
suspects_incidents_df = suspects_incidents_df[suspects_incidents_df['Child_0-11'] != 1]


# In[13]:


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


# In[14]:


# use states dictionary to encode dataframe
suspects_incidents_df['state_num'] = suspects_incidents_df['state'].apply(lambda x: states_dict[x])
suspects_incidents_df.head(3)


# In[15]:


# drop unnecessary columns
suspects_incidents_df = suspects_incidents_df.drop(columns=['incident_id', 'index', 'Adult_18+', 'Child_0-11', 'Teen_12-17', 'status_Injured', 
                                                            'status_Injured, Arrested', 'status_Unharmed, Arrested', 'status_Unharmed', 
                                                            'status_Arrested', 'status_Killed', 'status_Other', 'date', 
                                                            'state', 'latitude', 'longitude', 'incident_characteristics', 'notes'])
suspects_incidents_df.head()


# In[16]:


# inspect spread of targets
suspects_incidents_df['status'].value_counts()


# In[17]:


# drop columns with unknown status
suspects_incidents_df = suspects_incidents_df[suspects_incidents_df['status'] != status_defs['status_Unknown']]


# In[18]:


# drop NA columns
suspects_incidents_df = suspects_incidents_df.dropna()
suspects_incidents_df.info()


# In[19]:


# calculate IGR
Q1 = suspects_incidents_df.quantile(0.25)
Q3 = suspects_incidents_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[20]:


# get more detailed description 
suspects_incidents_df.describe()


# In[21]:


# inspect age column
suspects_incidents_df.boxplot(column=['participant_age'])


# In[22]:


# remove outlier ages
suspects_incidents_df = suspects_incidents_df[(suspects_incidents_df['participant_age'] < 100.0) & (suspects_incidents_df['participant_age'] > 11.0)]


# In[23]:


# separate features and target
y = suspects_incidents_df['status']
X = suspects_incidents_df.drop(columns=['status'])


# In[24]:


# plot target data
suspects_incidents_df.plot.scatter(x='female', y='participant_age', c='status', colormap='cool', figsize=(10,6))


# In[25]:


# split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
Counter(y_train)


# In[26]:


# implement random oversampling
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)


# In[27]:


# create scaler for disctric columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fit the scaler
X_scaler = scaler.fit(X_resampled)


# In[28]:


# scale the data
X_resampled_scaled = X_scaler.transform(X_resampled)
X_test_scaled = X_scaler.transform(X_test)


# ### Multinomial Logistic Regression Algorithm

# In[29]:


# import logistic regression algorithm
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=200, random_state=1)


# In[30]:


# fit model using training data
classifier.fit(X_resampled_scaled, y_resampled)


# In[31]:


# make predictions
y_pred = classifier.predict(X_test_scaled)


# In[32]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[33]:


# print balanced accuracy score
print(f'Accuracy: {balanced_accuracy_score(y_test, y_pred)}')


# In[34]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### LinearSVC Model

# In[35]:


# create linear SVM model
from sklearn.svm import LinearSVC
model = LinearSVC(random_state=1)


# In[36]:


# fit the data
model.fit(X_resampled_scaled, y_resampled)


# In[37]:


# make predictions
y_pred = model.predict(X_test_scaled)


# In[38]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[39]:


# print balanced accuracy score
print(f'Accuracy: {balanced_accuracy_score(y_test, y_pred)}')


# In[40]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### Random Forest Classifier

# In[41]:


# define model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=1)


# In[42]:


rf_model.fit(X_resampled, y_resampled)


# In[43]:


# get predictions
y_pred = rf_model.predict(X_test)


# In[44]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[45]:


# print balanced accuracy score
print(f'Accuracy: {balanced_accuracy_score(y_test, y_pred)}')


# In[46]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# ### Deep NN

# In[47]:


# set up model 
import tensorflow as tf
num_input_features = len(X_train.iloc[0])
hidden_nodes_layer1 = 24
nn = tf.keras.models.Sequential()


# In[48]:


# build model
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=num_input_features, activation='relu'))
nn.add(tf.keras.layers.Dense(units=3, activation='softmax'))
nn.summary()


# In[49]:


# compile the model
nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[50]:


# train model
fit_nn = nn.fit(X_train, y_train, epochs=50)


# In[51]:


# store predictions (extract indices with largest probability prediction)
y_pred = nn.predict(X_test)
y_pred = [np.argmax(x) for x in y_pred]


# In[52]:


# evaluate model
model_loss, model_accuracy = nn.evaluate(X_test, y_test, verbose=2)
print(f'Loss: {model_loss}, Accuracy: {model_accuracy}')


# In[53]:


# print confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[54]:


# print classification report
print(classification_report_imbalanced(y_test, y_pred))


# In[ ]:




