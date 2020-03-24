#!/usr/bin/env python
# coding: utf-8

# # Perform NLP on *incident_characteristics*
# 
# - Extract 'incident_characteristics' column
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
spark = SparkSession.builder.appName("NLP_Characteristics").config("spark.driver.extraClassPath","content/postgresql-42.2.9.jar").getOrCreate()


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


# In[9]:


# drop unnecessary columns and null rows
characteristics_df = incidents_df.drop('date', 'state', 'longitude', 'latitude', 'notes', 'congressional_district',
                             'state_house_district', 'state_senate_district').filter(incidents_df['incident_characteristics'].isNotNull())


# In[12]:


# add columns for lengths of text
characteristics_length_df = characteristics_df.withColumn('characteristics_length', length(characteristics_df['incident_characteristics']))
characteristics_length_df.show()


# ### Predicting *n_killed*

# In[13]:


# rename label column 
characteristics_length_df = characteristics_length_df.withColumnRenamed('n_killed', 'label')
characteristics_length_df.printSchema()


# In[14]:


# show dataframe summary
characteristics_length_df.describe().show()


# In[20]:


# create features
tokenizer = Tokenizer(inputCol="incident_characteristics", outputCol="token_characteristics")
stopremove = StopWordsRemover(inputCol='token_characteristics',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="token_characteristics", outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')


# In[21]:


# Create feature vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
clean_up = VectorAssembler(inputCols=['idf_token', 'characteristics_length'], outputCol='features')


# In[22]:


# Create and run a data processing Pipeline
from pyspark.ml import Pipeline
data_prep_pipeline = Pipeline(stages=[tokenizer, stopremove, hashingTF, idf, clean_up])


# In[23]:


# Fit and transform the pipeline
cleaner = data_prep_pipeline.fit(characteristics_length_df)
cleaned = cleaner.transform(characteristics_length_df)


# In[24]:


# Show label and resulting features
cleaned.select(['label', 'features']).show()


# In[25]:


# Break data down into a training set and a testing set
training, testing = cleaned.randomSplit([0.7, 0.3])


# In[26]:


from pyspark.ml.classification import NaiveBayes
# Create a Naive Bayes model and fit training data
nb = NaiveBayes()


# In[27]:


# fit model using training data
predictor = nb.fit(training)


# In[28]:


# Transform the model with testing data
test_results = predictor.transform(testing)
test_results.limit(5).toPandas().head()


# In[29]:


# Use the Class Evalueator for a cleaner description
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting n_killed was: %f" % acc)


# ### Predicting *n_injured*

# In[30]:


# drop previous label column and rename n_injured 
characteristics_length_df = characteristics_length_df.drop('label').withColumnRenamed('n_injured', 'label')
characteristics_length_df.printSchema()


# In[31]:


# create features
tokenizer = Tokenizer(inputCol="incident_characteristics", outputCol="token_characteristics")
stopremove = StopWordsRemover(inputCol='token_characteristics',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="token_characteristics", outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')


# In[32]:


# Create feature vectors
clean_up = VectorAssembler(inputCols=['idf_token', 'characteristics_length'], outputCol='features')


# In[33]:


# Create and run a data processing Pipeline
data_prep_pipeline = Pipeline(stages=[tokenizer, stopremove, hashingTF, idf, clean_up])


# In[34]:


# Fit and transform the pipeline
cleaner = data_prep_pipeline.fit(characteristics_length_df)
cleaned = cleaner.transform(characteristics_length_df)


# In[35]:


# Show label and resulting features
cleaned.select(['label', 'features']).show()


# In[36]:


# Break data down into a training set and a testing set
training, testing = cleaned.randomSplit([0.7, 0.3])


# In[37]:


# Create new Naive Bayes model
nb = NaiveBayes()


# In[38]:


# Transform the model with testing data
test_results = predictor.transform(testing)
test_results.limit(5).toPandas().head()


# In[39]:


# evaluate model performance
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting n_injured was: %f" % acc)


# In[ ]:




