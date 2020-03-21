#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

#import dependencies
import pandas as pd, json
import psycopg2
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import session
from sqlalchemy import create_engine, func
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[2]:


host_address = 'bootcamp-final-project.c8u2worjd1ui.us-east-1.rds.amazonaws.com'
port = '5432'
username = 'peter_jennifer'
password = 'Puhj6k2%pbW'
db = 'us_gun_violence'


# In[3]:


# A long string that contains the necessary Postgres login information
postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(
                username=username,
                password=password,
                ipaddress=host_address,
                port=port,
                dbname=db))
# Create the connection
engine = create_engine(postgres_str)


# In[4]:


suspects_df = pd.read_sql_query('''SELECT * FROM suspects;''', engine)
incidents_df = pd.read_sql_query('''SELECT * FROM incidents;''', engine)
guns_df = pd.read_sql_query('''SELECT * FROM guns;''', engine)


# In[5]:


guns_df_dup = guns_df.drop_duplicates(subset='incident_id')
guns_df_dup


# In[6]:


#male vs female
count = suspects_df.participant_gender.value_counts()
counts = [167708, 11746]
labels = ['Males', 'Females']
colors = ['indigo', 'darkorchid']

fig_gender = plt.pie(counts, labels=labels, colors=colors, shadow=True)

plt.title('Gender in Gun Violence')

plt.show()

fig_gender


# In[7]:


#shootings by politician/district/state
states = pd.DataFrame(incidents_df.state.value_counts())
state = states.index
shootings = states.state.tolist()

#graph
fig_states = plt.bar(state[:10], shootings[:10], color='yellowgreen')
plt.xticks(state[:10], rotation=45, ha='right')
plt.title('Shootings by State')

fig_states


# In[8]:


#shootings by politician/district
district = pd.DataFrame(incidents_df.state_senate_district.value_counts())
senate = district.index
shootings = district.state_senate_district.tolist()

#graph
fig_senate = plt.bar(senate[:10], shootings[:10], color='teal')
plt.xticks(senate[:10], rotation=45, ha='right')
plt.title('Shootings by Senate District')

fig_senate


# In[9]:


#shootings by politician/district
guns = pd.DataFrame(guns_df.gun_type.value_counts())
labels = guns.index
number = guns.gun_type.tolist()
total = sum(number)
percentages = []
for n in number[1:27]:
    percent = (n/total ) * 100
    percentages.append(percent)
    
fig_guns = plt.barh(labels[1:11], number[1:11], align='center', color='y')
plt.gca().invert_yaxis()

plt.title('Gun Types Used in Shootings')

fig_guns


# In[10]:


age_groups = pd.DataFrame(suspects_df.participant_age_group.value_counts())

groups = age_groups.index
number = age_groups.participant_age_group.tolist()

fig_age_group = plt.bar(groups, number, color='firebrick')
plt.xticks(groups)

fig_age_group


# In[11]:


age = suspects_df.participant_age
incidents = suspects_df.participant_status

fig_ages = plt.hist(age, bins=50, color='lightblue')
plt.xlim(0,100)
plt.xlabel('Age of Shooter')
fig_ages


# In[12]:


guns_df.gun_stolen.value_counts()


# In[13]:


guns_df.gun_stolen.value_counts()
gun_ownership = guns_df.loc[(guns_df['gun_stolen'] == 'Stolen') | (guns_df['gun_stolen'] == 'Not-stolen')] 

gun_owners_df = gun_ownership.merge(incidents_df, on= 'incident_id', how='left')

gun_owners_df


# In[14]:


il_guns = gun_owners_df.loc[gun_owners_df['state'] == 'Illinois']
test = il_guns.gun_stolen.value_counts()

test


# In[15]:


top_5 = ['Illinois', 'California', 'Florida', 'Texas', 'Ohio']
def state_guns(df, column1, column2, top5):
    not_stolen = []
    stolen = []
    for state in top5:
        guns = df.loc[df[column1] == state]
        s_ns = guns[column2].value_counts().tolist()
        not_stolen.append(s_ns[1])
        stolen.append(s_ns[0])
    return not_stolen, stolen

not_stolen, stolen = state_guns(gun_owners_df, 'state', 'gun_stolen', top_5)

Illinois: [232, 16]
California: [274, 63]
Florida: [630, 165]
Texas: [354, 159]
Ohio: [204, 41]
    


# In[16]:


width = 0.25 
fig_stolen = plt.bar(top_5, stolen, width, color='#F78F1E')
fig_notstolen = plt.bar(top_5, not_stolen, width, color='#FFC222')

plt.xticks(top_5)
plt.xlabel('top 5 states with the most shootings')
plt.ylabel('number of guns')
plt.legend(['Stolen', 'Not Stolen'], loc='upper left')           


# In[17]:


incidents_df

new_df = incidents_df.dropna()


# In[23]:


map_df = new_df[['date', 'latitude', 'longitude', 'n_killed', 'notes']]


# In[22]:


map_df.to_csv('Resources/map.csv')
map_df


# In[38]:


col = ['date', 'n_killed', 'notes']
sample_df = map_df.sort_values('n_killed', ascending=False).head(50)

sample_df


# In[26]:


def df_to_geojson(df, properties, longitude='longitude', latitude='latitude'):
    geojson = {"type":"FeatureCollection", "features":[]}
    for _, row in df.iterrows():
        feature = {"type": "Feature",
                   "geometry": {"type": "Point",
                               "coordinates": []},
                   "properties": {}}
        feature["geometry"]["coordinates"] = [row[longitude], row[latitude]]
        for prop in properties:
            feature["properties"][prop] = row[prop]
        geojson["features"].append(feature)
    return geojson


# In[39]:


geojson_dict = df_to_geojson(sample_df, col)

geojson_str = json.dumps(geojson_dict, indent=2, sort_keys=True, default=str)


# In[40]:


# save the geojson result to a file
output_filename = 'static/js/incident.json'
with open(output_filename, 'w') as output_file:
    output_file.write('var dataset = {};'.format(geojson_str))
    
# how many features did we save to the geojson file?
print('{} geotagged features saved to file'.format(len(geojson_dict['features'])))


# In[ ]:


output_filename


# In[ ]:




