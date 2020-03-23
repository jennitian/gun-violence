#!/usr/bin/env python
# coding: utf-8

# # Perform NLP on *notes*
# 
# - Extract 'notes' column
# - Perform NLP to determine "hot"/"key" words
# - Determine if there's any correlation between presence of key words and incident stats such as n_killed, suspect outcome, or gun type used

# In[1]:


# import dependencies
import findspark
# import db password
from config import db_password


# In[2]:


# start a SparkSession
findspark.init()


# In[3]:


# start a spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("NLP_Notes").config("spark.driver.extraClassPath","content/postgresql-42.2.9.jar").getOrCreate()


# In[4]:


# credentials for connecting to Postgres db
POSTGRES_ADDRESS = 'bootcamp-final-project.c8u2worjd1ui.us-east-1.rds.amazonaws.com'
POSTGRES_PORT = 5432
POSTGRES_USERNAME = 'peter_jennifer'
POSTGRES_PASSWORD = db_password
POSTGRES_DBNAME = 'us_gun_violence'


# In[5]:


# set up connection URL and db properties
db_url = f'jdbc:postgresql://{POSTGRES_ADDRESS}:{POSTGRES_PORT}/{POSTGRES_DBNAME}'
db_properties = {'user': POSTGRES_USERNAME, 'password': POSTGRES_PASSWORD}


# In[6]:


# import data
incidents_df = spark.read.jdbc(url=db_url, table='incidents', properties=db_properties)


# In[7]:


# display dateframe
incidents_df.limit(5).toPandas().head()


# In[8]:


# import functions
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.sql.functions import length, col


# In[10]:


# drop unnecessary columns and null rows
notes_df = incidents_df.drop('date', 'state', 'longitude', 'latitude', 'incident_characteristics', 'congressional_district',
                             'state_house_district', 'state_senate_district').filter(incidents_df['notes'].isNotNull())


# In[27]:


# add columns for lengths of text
notes_length_df = notes_df.withColumn('notes_length', length(notes_df['notes']))
notes_length_df.show()


# ### Predicting *n_killed*

# In[28]:


# rename label column 
notes_length_df = notes_length_df.withColumnRenamed('n_killed', 'label')
notes_length_df.printSchema()


# In[13]:


# show dataframe summary
notes_length_df.describe().show()


# In[14]:


# create features
tokenizer = Tokenizer(inputCol="notes", outputCol="token_notes")
stopremove = StopWordsRemover(inputCol='token_notes',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="token_notes", outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')


# In[15]:


# Create feature vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
clean_up = VectorAssembler(inputCols=['idf_token', 'notes_length'], outputCol='features')


# In[16]:


# Create and run a data processing Pipeline
from pyspark.ml import Pipeline
data_prep_pipeline = Pipeline(stages=[tokenizer, stopremove, hashingTF, idf, clean_up])


# In[17]:


# Fit and transform the pipeline
cleaner = data_prep_pipeline.fit(notes_length_df)
cleaned = cleaner.transform(notes_length_df)


# In[18]:


# Show label and resulting features
cleaned.select(['label', 'features']).show()


# In[19]:


# Break data down into a training set and a testing set
training, testing = cleaned.randomSplit([0.7, 0.3])


# In[21]:


from pyspark.ml.classification import NaiveBayes
# Create a Naive Bayes model and fit training data
nb = NaiveBayes()


# In[23]:


# make predictions
predictor = nb.fit(training)


# In[24]:


# Transform the model with testing data
test_results = predictor.transform(testing)
test_results.limit(5).toPandas().head()


# In[25]:


# Use the Class Evalueator for a cleaner description
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting n_killed was: %f" % acc)


# ### Predict n_injured

# In[30]:


# drop previous label and rename n_injured column 
notes_length_df = notes_length_df.drop('label').withColumnRenamed('n_injured', 'label')
notes_length_df.printSchema()


# In[31]:


# show dataframe summary
notes_length_df.describe().show()


# In[32]:


# create features
tokenizer = Tokenizer(inputCol="notes", outputCol="token_notes")
stopremove = StopWordsRemover(inputCol='token_notes',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="token_notes", outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')


# In[33]:


# Create feature vectors
clean_up = VectorAssembler(inputCols=['idf_token', 'notes_length'], outputCol='features')


# In[34]:


# Create and run a data processing Pipeline
data_prep_pipeline = Pipeline(stages=[tokenizer, stopremove, hashingTF, idf, clean_up])


# In[35]:


# Fit and transform the pipeline
cleaner = data_prep_pipeline.fit(notes_length_df)
cleaned = cleaner.transform(notes_length_df)


# In[36]:


# Show label and resulting features
cleaned.select(['label', 'features']).show()


# In[37]:


# Break data down into a training set and a testing set
training, testing = cleaned.randomSplit([0.7, 0.3])


# In[39]:


# create new Naive Bayes model
nb = NaiveBayes()


# In[40]:


# fit the model with training data
predictor = nb.fit(training)


# In[41]:


# Transform the model with testing data
test_results = predictor.transform(testing)
test_results.limit(5).toPandas().head()


# In[42]:


# evaluate model performance
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting n_injured was: %f" % acc)


# In[ ]:




