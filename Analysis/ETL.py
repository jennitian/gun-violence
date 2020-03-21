#!/usr/bin/env python
# coding: utf-8

# In[1]:


#matplotlib inline command
get_ipython().run_line_magic('matplotlib', 'inline')

#importing dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


path = "Resources/gun-violence.csv"

df = pd.read_csv(path)

df.head()


# In[3]:


df.head()


# In[4]:


def split(df, column):
    return df[column].str.split("\d\:\:|\|\|\d{0,3}\:\:")


# In[5]:


columns = ['incident_id', 'date', 'state', 'latitude', 'longitude', 'n_killed', 'n_injured', 'n_guns_involved',
           'incident_characteristics', 'notes', 'congressional_district', 'state_house_district', 
           'state_senate_district']

exploded_columns = ['incident_id', 'participant_gender', 'participant_age', 'participant_age_group',
                       'participant_status', 'participant_type', 'gun_stolen', 
                        'gun_type']

shootings_df = df[columns]

exploded_df = df[exploded_columns]


# In[6]:


#participant type value counts
exploded_df.participant_type.value_counts()
#splitting participant_type
participant_type_split = split(exploded_df, 'participant_type')
#make new column with split list
exploded_df['participant'] = participant_type_split
#drop previous 
exploded_df = exploded_df.drop('participant_type', axis=1)


# In[7]:


#splitting participant_age
participant_age_split = split(exploded_df, 'participant_age')
exploded_df['age'] = participant_age_split
#drop previous 
exploded_df = exploded_df.drop('participant_age', axis=1)


# In[8]:


#splitting participant_gender
participant_gender_split = split(exploded_df, 'participant_gender')
exploded_df['genders'] = participant_age_split
#drop previous 
exploded_df = exploded_df.drop('participant_gender', axis=1)


# In[9]:


#splitting participant_status
participant_status_split = split(exploded_df, 'participant_status')
#make new column with split list
exploded_df['status'] = participant_status_split
#drop previous 
exploded_df = exploded_df.drop('participant_status', axis=1)


# In[10]:


#splitting gun_stolen
gun_stolen_split = split(exploded_df, 'gun_stolen')
#make new column with split list
exploded_df['gun_ownership'] = gun_stolen_split
#drop previous 
exploded_df = exploded_df.drop('gun_stolen', axis=1)


# In[11]:


#splitting gun_type
gun_type_split = split(exploded_df, 'gun_type')
#make new column with split list
exploded_df['type'] = gun_type_split
#drop previous 
exploded_df = exploded_df.drop('gun_type', axis=1)


# In[12]:


#splitting participant_age_group
age_split = split(exploded_df, 'participant_age_group')
#make new column with split list
exploded_df['age_group'] = age_split
#drop previous 
exploded_df = exploded_df.drop('participant_age_group', axis=1)


# In[13]:


exploded_df.head()


# In[14]:


exploded_df = exploded_df.set_index('incident_id').apply(lambda x: x.apply(pd.Series).stack().replace('', np.nan).dropna()).reset_index().drop('level_1', 1)


# In[15]:


suspects = exploded_df.loc[exploded_df['participant'] == 'Subject-Suspect']
suspects


# In[16]:


df = suspects.merge(shootings_df, how='left', on='incident_id')

df.columns


# In[17]:


df.to_csv('Resources/explode_test.csv')


# In[21]:


df


# In[20]:


#exploratory analysis
columns_na = ['incident_id', 'participant', 'age', 'genders', 'status',
       'gun_ownership', 'type', 'age_group', 'date', 'state', 'latitude',
       'longitude', 'n_killed', 'n_injured', 'n_guns_involved',
       'incident_characteristics', 'notes', 'congressional_district',
       'state_house_district', 'state_senate_district']
#counting null items
for item in columns_na:
    na = df[item].isnull().sum()
    print(f"{item} has {na} null items")   


# In[23]:


#
df.gun_ownership.count_values()


# In[ ]:




