#!/usr/bin/env python
# coding: utf-8

# In[26]:


# import dependencies
import pandas as pd
import numpy as np
import matplotlib as plt
from sqlalchemy import create_engine
import time

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


# # Encoding
# 
# ## Guns Table

# In[5]:


# import guns dataset from AWS
guns_df = pd.read_sql_table('guns', engine, columns=['incident_id', 'n_guns_involved', 'gun_stolen', 'gun_type'])
guns_df.head()


# In[6]:


# explore value counts of gun_type column
guns_df['gun_type'].value_counts()


# In[7]:


# define dictionary to be used to bin above values
gun_types = {'9mm': 'Handgun', '22 LR': 'Rifle', '40 SW': 'Handgun', '380 Auto': 'Handgun', 
            '45 Auto': 'Handgun', '38 Spl': 'Handgun', '223 Rem [AR-15]': 'Assault Rifle',
            '12 gauge': 'Shotgun', '7.62 [AK-47]': 'Assault Rifle', '357 Mag': 'Handgun',
            '25 Auto': 'Handgun', '32 Auto': 'Handgun', '20 gauge': 'Shotgun', '44 Mag': 'Handgun',
            '30-30 Win': 'Rifle', '410 gauge': 'Shotgun', '308 Win': 'Rifle', '30-06 Spr': 'Rifle',
            '10mm': 'Handgun', '16 gauge': 'Shotgun', '300 Win': 'Rifle', '28 gauge': 'Shotgun'}


# In[8]:


# map dictionary keys to dataframe
guns_df['category'] = guns_df['gun_type'].map(gun_types)
guns_df.head()


# In[9]:


guns_df['category'].value_counts()


# In[10]:


# discard previous gun type column and rename newly generated categories
guns_df.drop(columns=['gun_type'], inplace=True)
guns_df.rename(columns={'category': 'gun_type'}, inplace=True)

guns_df.head()


# In[11]:


# inspect gun_stolen value counts
guns_df['gun_stolen'].value_counts()


# In[12]:


# replace Unknown values with NaN
guns_df['gun_stolen'].replace({'Unknown': np.nan}, inplace=True)
guns_df.head()


# In[13]:


# encode gun_stolen and gun_type
guns_df_encoded = pd.get_dummies(guns_df, columns=['gun_stolen', 'gun_type'])
guns_df_encoded.head()


# In[14]:


# rename columns
guns_df_encoded.rename(columns={'gun_stolen_Not-stolen': 'not_stolen', 'gun_stolen_Stolen': 'stolen',
                               'gun_type_Assault Rifle': 'assault_rifle', 'gun_type_Handgun': 'handgun',
                               'gun_type_Rifle': 'rifle', 'gun_type_Shotgun': 'shotgun'}, inplace=True)
guns_df_encoded.head()


# ## Suspects Table

# In[15]:


# import suspects dataset from AWS
suspects_df = pd.read_sql_table('suspects', engine, columns=['incident_id', 'participant_gender', 
                                                            'participant_age', 'participant_age_group',
                                                            'participant_status'])
suspects_df.head()


# In[16]:


# inspect gender column
suspects_df['participant_gender'].value_counts()


# In[17]:


# inspect age_group column
suspects_df['participant_age_group'].value_counts()


# In[18]:


# inspect status column
suspects_df['participant_status'].value_counts()


# In[19]:


# Clean up bins, can't be killed and uninjured, assume the person died post-incident, report as killed
status_labels = {'Killed, Arrested': 'Killed', 'Injured, Unharmed, Arrested': 'Injured, Arrested',
                'Killed, Unharmed': 'Killed', 'Killed, Unharmed, Arrested': 'Killed', 'Injured, Unharmed': 
                'Injured', 'Killed, Injured': 'Killed'}


# In[20]:


# map dictionary keys to dataframe
suspects_df['status'] = suspects_df['participant_status'].map(status_labels).fillna(suspects_df['participant_status'])
suspects_df.drop(columns=['participant_status'], inplace=True)
suspects_df.head()


# In[21]:


suspects_df_encoded = pd.get_dummies(suspects_df, columns=['participant_gender', 'participant_age_group',
                                                          'status'])
suspects_df_encoded.head()


# In[28]:


suspects_df_encoded.rename(columns={'participant_gender_Female': 'female', 'participant_age_group_Adult 18+':
                                   'Adult_18+', 'participant_age_group_Child 0-11': 'Child_0-11', 'participant_age_group_Teen 12-17':
                                   'Teen_12-17'}, inplace=True)
suspects_df_encoded.drop(columns=['participant_gender_Male'], inplace=True)
suspects_df_encoded.head()


# ## Incidents Table

# In[23]:


# import incidents dataset from AWS
incidents_df = pd.read_sql_table('incidents', engine, columns=['date', 'state', 'latitude', 'longitude', 'n_killed',
                                                              'n_injured', 'incident_characteristics', 'notes', 'congressional_district',
                                                              'state_house_district', 'state_senate_district'])
incidents_df.head()


# # Load Data into AWS

# In[24]:


# method to divide dataframe into chunks
def get_chunks(df, size):
    return (df[pos:pos+size] for pos in range(0, len(df), size))

# method to import dataframe into database
def import_to_db(df, table_name, con_engine, chunksize):
    start_time = time.time()
    for i, chunk in enumerate(get_chunks(df, chunksize)):
        # print status update of rows being processed
        print(f'importing rows {i*chunksize} to {len(chunk) + i*chunksize}...', end='')
        
        if_exists_op = 'replace' if i == 0 else 'append'
        chunk.to_sql(name=table_name, con=con_engine, if_exists=if_exists_op, method='multi')

        # add elapsed time to print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')


# In[29]:


# import transformed gun data in chunks
import_to_db(guns_df_encoded, 'guns_ml_transformed', engine, 10000)


# In[30]:


# import transformed gun data in chunks
import_to_db(suspects_df_encoded, 'suspects_ml_transformed', engine, 10000)


# In[ ]:




