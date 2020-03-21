#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


# import dataset from https://www.kaggle.com/jameslko/gun-violence-data
shootings_df = pd.read_csv('~/Downloads/gun-violence-data_01-2013_03-2018.csv')
shootings_df.head()


# In[3]:


# display summary of raw dataset
shootings_df.info()


# # Create/organize dataframe structures

# In[4]:


# declare column structures for each dataframe to be made
total_columns = ['incident_id', 'date', 'state', 'latitude', 'longitude', 'n_killed', 'n_injured',
           'gun_type', 'gun_stolen', 'n_guns_involved', 'incident_characteristics', 
           'participant_age', 'participant_age_group', 'participant_gender', 'participant_status',
           'participant_type', 'notes', 'congressional_district', 'state_house_district',
           'state_senate_district']

incident_columns = ['incident_id', 'date', 'state', 'latitude', 'longitude', 'n_killed', 'n_injured',
           'incident_characteristics', 'notes', 'congressional_district', 'state_house_district',
           'state_senate_district']

participant_columns = ['incident_id', 'participant_gender', 'participant_age', 'participant_age_group', 
                    'participant_status', 'participant_type']

gun_columns = ['incident_id', 'gun_stolen', 'gun_type', 'n_guns_involved']


# In[5]:


# drop all irrelevant columns
shootings_df = shootings_df[total_columns]


# ### Create Incidents Dataframe

# In[6]:


# create incidents dataframe
incidents_df = shootings_df[incident_columns].copy()
incidents_df.head()


# In[7]:


# set index as incident_id
incidents_df = incidents_df.set_index('incident_id')
incidents_df.head()


# In[8]:


# convert date to datetime
incidents_df.loc[:,'date'] = pd.to_datetime(incidents_df['date'])


# In[9]:


# inspect dataframe
incidents_df.info()


# In[10]:


# export sample to csv
incidents_df[0:1000].to_csv('./sample_transformations/sample_incidents.csv')


# ### Explode Function

# In[11]:


## Explode columns function
## Takes in the DataFrame, columns containing multiple datapoints, and 
## name of sequential index that will be created
## Performs additional cleaning by casting datatype of incident_id

def explode_columns(df, columns, index_name):
    # store original DataFrame without exploded columns
    # serves as an "aggregator" for all the newly created DataFrames
    aggregate_df = df.drop(columns=columns)
    aggregate_df['temp_index'] = 0
    
    # declare MultiIndex structure in preparation for merge with
    # exploded DataFrames
    index = pd.MultiIndex.from_arrays([aggregate_df.index.values, aggregate_df['temp_index'].values], 
                                  names=['incident_index', 'temp_index'])
    aggregate_df = aggregate_df.set_index(index).drop(columns=['temp_index'])
    
    # loop through "to-be exploded" columns
    for col in columns:
        # extract column and split on '|' delimiter
        temp_series = df[col].str.split('\\|').apply(pd.Series, 1).stack().replace('', np.nan).dropna()
        temp_series.index = temp_series.index.droplevel(-1)
        temp_series.name = col
        
        # expand column on ':' delimiter, retain provided index to be used when merging
        temp_df = temp_series.str.split(':',expand=True).drop(columns=1)
        temp_df[0] = pd.to_numeric(temp_df[0])
        temp_df.rename(columns = {0: 'temp_index', 2:col}, inplace=True)
        
        # create multiIndex 
        index = pd.MultiIndex.from_arrays([temp_df.index.values, temp_df['temp_index'].values], 
                                  names=['incident_index', 'temp_index'])
        temp_df = temp_df.set_index(index).drop(columns=['temp_index'])
        
        # combine new DataFrame with aggregator DataFrame
        aggregate_df = aggregate_df.join(temp_df, on=['incident_index', 'temp_index'], how='outer').sort_index()
        
        # print status update
        print(f'Finished exploding and merging {col} column...')
    
    # fill non-exploded columns with repeat values within subindex
    aggregate_df[aggregate_df.columns.difference(columns)] = aggregate_df[aggregate_df.columns.difference(columns)].groupby(level=0).fillna(method='ffill')
    
    # drop indices used for merges
    aggregate_df = aggregate_df.reset_index(drop=True)

    # confirm incident_id is int
    aggregate_df.loc[:,'incident_id'] = pd.to_numeric(aggregate_df['incident_id'], downcast='integer')
    
    # set index name
    aggregate_df.index.name = index_name
    
    print('Done')

    return aggregate_df
        


# ### Create Suspects Dataframe

# In[12]:


# create df for participants only
participants_df = shootings_df[participant_columns].copy()


# In[13]:


# explode the columns with multiple data points into separate rows
exploded_participants_df = explode_columns(participants_df, participant_columns[1:], 'suspect_index')


# In[14]:


# select suspects only and drop participant_type
suspects_df = exploded_participants_df.loc[exploded_participants_df['participant_type'] == 'Subject-Suspect']
suspects_df = suspects_df.drop(columns='participant_type')


# In[15]:


# drop rows where all participant information is NaN
suspects_df = suspects_df.dropna(thresh=2)


# In[16]:


# reset index and name
suspects_df.index = suspects_df.reset_index(drop=True).index.rename('suspect_index')
suspects_df.head()


# In[17]:


# convert age to numeric
suspects_df.loc[:,'participant_age'] = pd.to_numeric(suspects_df['participant_age'])


# In[18]:


# get summary of df
suspects_df.info()


# In[19]:


# export sample to csv
suspects_df[0:1000].to_csv('./sample_transformations/sample_suspects.csv')


# ### Create Guns Dataframe

# In[20]:


# create separate guns dataframe
guns_df = shootings_df[gun_columns].dropna(thresh=2).copy()


# In[21]:


# explode gun columns
guns_df = explode_columns(guns_df, gun_columns[1:3], 'gun_index')
guns_df.head()


# In[22]:


# convert datatype of n_guns_involved
guns_df.loc[:,'n_guns_involved'] = pd.to_numeric(guns_df['n_guns_involved'], downcast='integer')


# In[23]:


# inspect exploded df
guns_df.info()


# In[24]:


# export sample to csv
guns_df[0:1000].to_csv('./sample_transformations/sample_guns.csv')


# ## Load Data into AWS

# In[25]:


# import dependencies
import psycopg2
from sqlalchemy import create_engine
import time

# import db password
from config import db_password


# In[26]:


# credentials for connecting to Postgres db
POSTGRES_ADDRESS = 'bootcamp-final-project.c8u2worjd1ui.us-east-1.rds.amazonaws.com'
POSTGRES_PORT = 5432
POSTGRES_USERNAME = 'peter_jennifer'
POSTGRES_PASSWORD = db_password
POSTGRES_DBNAME = 'us_gun_violence'


# In[27]:


# creat connection string and database engine
db_string = f'postgres://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_ADDRESS}:{POSTGRES_PORT}/{POSTGRES_DBNAME}'

engine = create_engine(db_string)


# In[28]:


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


# import gun data in chunks
import_to_db(guns_df, 'guns', engine, 10000)


# In[30]:


# import suspect data in chunks
import_to_db(suspects_df, 'suspects', engine, 10000)


# In[31]:


# import incident data in chunks
import_to_db(incidents_df, 'incidents', engine, 10000)


# In[ ]:




