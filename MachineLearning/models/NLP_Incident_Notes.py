#!/usr/bin/env python
# coding: utf-8

# # Perform NLP on *notes*

# In[1]:


# import Spark dependencies
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


# import NLP dependencies
import numpy as np
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.sql.functions import length, col

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced


# In[9]:


# drop unnecessary columns and null rows
notes_df = incidents_df.drop('date', 'state', 'longitude', 'latitude', 'incident_characteristics', 'congressional_district',
                             'state_house_district', 'state_senate_district').filter(incidents_df['notes'].isNotNull())


# In[10]:


# add columns for lengths of text
notes_length_df = notes_df.withColumn('notes_length', length(notes_df['notes']))
notes_length_df.show()


# ### Predicting *n_killed*

# In[11]:


# rename label column 
notes_length_df = notes_length_df.withColumnRenamed('n_killed', 'label')
notes_length_df.printSchema()


# In[12]:


# show dataframe summary
notes_length_df.describe().show()


# In[13]:


# get spread of n_killed
notes_length_df.groupBy('label').count().show()


# In[14]:


# filter rows where n_killed > 2
notes_length_df = notes_length_df.filter(notes_length_df.label <= 2)


# In[15]:


# create features
tokenizer = Tokenizer(inputCol="notes", outputCol="token_notes")
stopremove = StopWordsRemover(inputCol='token_notes',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="token_notes", outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')


# In[16]:


# Create feature vectors
clean_up = VectorAssembler(inputCols=['idf_token', 'notes_length'], outputCol='features')


# In[17]:


# Create and run a data processing Pipeline
data_prep_pipeline = Pipeline(stages=[tokenizer, stopremove, hashingTF, idf, clean_up])


# In[18]:


# Fit and transform the pipeline
cleaner = data_prep_pipeline.fit(notes_length_df)
cleaned = cleaner.transform(notes_length_df)


# In[19]:


# Show label and resulting features
cleaned.select(['label', 'features']).show()


# In[20]:


# Break data down into a training set and a testing set
training, testing = cleaned.randomSplit([0.7, 0.3])


# In[21]:


# Create a Naive Bayes model
nb = NaiveBayes()


# In[22]:


# fit model with training data
predictor = nb.fit(training)


# In[23]:


# Transform the model with testing data
test_results = predictor.transform(testing)
test_results.limit(5).toPandas().head()


# In[24]:


# Use the Class Evalueator for a cleaner description
acc_eval = MulticlassClassificationEvaluator().setMetricName("accuracy")
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting n_killed was: %f" % acc)


# In[25]:


# store np arrays of labels & predictions for evaluation
y_labels = np.array(testing.select('label').collect())
y_pred = np.array(test_results.select('prediction').collect())


# In[26]:


# print confusion matrix
print(confusion_matrix(y_labels, y_pred))


# In[27]:


# print classification report
print(classification_report_imbalanced(y_labels, y_pred))


# ### Predict *n_injured*

# In[28]:


# drop previous label and rename n_injured column 
notes_length_df = notes_length_df.drop('label').withColumnRenamed('n_injured', 'label')
notes_length_df.printSchema()


# In[29]:


# show dataframe summary
notes_length_df.describe().show()


# In[30]:


# get spread of n_injured
notes_length_df.groupBy('label').count().show()


# In[31]:


# filter rows where n_injured > 4
notes_length_df = notes_length_df.filter(notes_length_df.label <= 2)


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


# In[38]:


# create new Naive Bayes model
nb = NaiveBayes()


# In[39]:


# fit the model with training data
predictor = nb.fit(training)


# In[40]:


# Transform the model with testing data
test_results = predictor.transform(testing)
test_results.limit(5).toPandas().head()


# In[41]:


# evaluate model performance
acc_eval = MulticlassClassificationEvaluator().setMetricName("accuracy")
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting n_injured was: %f" % acc)


# In[42]:


# store np arrays of labels & predictions for evaluation
y_labels = np.array(testing.select('label').collect())
y_pred = np.array(test_results.select('prediction').collect())


# In[43]:


# print confusion matrix
print(confusion_matrix(y_labels, y_pred))


# In[44]:


# print classification report
print(classification_report_imbalanced(y_labels, y_pred))


# In[ ]:




