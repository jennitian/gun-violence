#!/usr/bin/env python
# coding: utf-8

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
spark = SparkSession.builder.appName("BigDataHW").config("spark.driver.extraClassPath","content/postgresql-42.2.9.jar").getOrCreate()


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
incidents_df.show()


# In[27]:


# import functions
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.sql.functions import length, col


# In[31]:


# drop unnecessary columns and null rows
notes_df = incidents_df.drop('date', 'state', 'longitude', 'latitude', 'incident_characteristics', 'congressional_district',
                             'state_house_district', 'state_senate_district').filter(notes_df['notes'].isNotNull())


# In[32]:


# add columns for lengths of text
notes_length_df = notes_df.withColumn('notes_length', length(notes_df['notes']))
notes_length_df.show()


# In[60]:


# rename label column
notes_length_df = notes_length_df.withColumnRenamed('n_killed', 'label')
notes_length_df.printSchema()


# In[61]:


# create features
tokenizer = Tokenizer(inputCol="notes", outputCol="token_notes")
stopremove = StopWordsRemover(inputCol='token_notes',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="token_notes", outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')


# In[62]:


# Create feature vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
clean_up = VectorAssembler(inputCols=['idf_token', 'notes_length'], outputCol='features')


# In[63]:


# Create and run a data processing Pipeline
from pyspark.ml import Pipeline
data_prep_pipeline = Pipeline(stages=[tokenizer, stopremove, hashingTF, idf, clean_up])


# In[64]:


# Fit and transform the pipeline
cleaner = data_prep_pipeline.fit(notes_length_df)
cleaned = cleaner.transform(notes_length_df)


# In[65]:


# Show label and resulting features
cleaned.select(['label', 'features']).show()


# In[66]:


# Break data down into a training set and a testing set
training, testing = cleaned.randomSplit([0.7, 0.3])


# In[67]:


from pyspark.ml.classification import NaiveBayes
# Create a Naive Bayes model and fit training data
nb = NaiveBayes()
predictor = nb.fit(training)


# In[69]:


# Transform the model with testing data
test_results = predictor.transform(testing)
test_results.limit(5).toPandas().head()


# In[70]:


# Use the Class Evalueator for a cleaner description
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting reviews was: %f" % acc)


# In[ ]:




