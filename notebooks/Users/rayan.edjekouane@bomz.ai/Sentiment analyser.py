# Databricks notebook source
# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Spark Context
# MAGIC 
# MAGIC We iniated a spark context to work on our data

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %run "./Includes/Utility-Methods"

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Reading Data as a Spark Dataframe

# COMMAND ----------

data_spark = spark.read.table("tweet_csv")

data_spark.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Pandas Dataframe
# MAGIC 
# MAGIC To manipulate and have a good data we convert our spark Dataframe to an Pandas one

# COMMAND ----------

import pandas as pd
from sklearn.utils import shuffle

# COMMAND ----------

data_pandas = data_spark.toPandas()

# COMMAND ----------

data_pandas.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Preprocessing
# MAGIC 
# MAGIC  #### a. Remove non ascii characters
# MAGIC   
# MAGIC  #### b. Remove punctuation(punctuation might be useful in some cases!)
# MAGIC   
# MAGIC  #### c. Remove texts shorter than a specific length
# MAGIC   
# MAGIC  #### d. Correct spelling (you can use sparkNLP ‘check_spelling_dl’ for context preserving spell correction)

# COMMAND ----------

data_pandas = data_pandas.drop(['id_tweet','query_tweet', 'username_tweet'], axis=1) # removing useless columns
data_pandas.rename(columns={'sentiment': 'real_sent','text_tweet': 'text'}, inplace=True) #change the name of the text column to fit in the Pretrained Pipeline
data_pandas.head(5)

# COMMAND ----------

data_pandas.info()

# COMMAND ----------

data_pandas['text'] = pd.Series(data_pandas['text'], dtype="string")
data_pandas['real_sent'] = pd.Series(data_pandas['real_sent'], dtype="int")
data_pandas['date_tweet'] = pd.Series(data_pandas['date_tweet'], dtype="string")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC  #### We took just 100 000 entries to be faster

# COMMAND ----------

data_pandas = shuffle(data_pandas)
data_pandas = data_pandas.head(1000000)
data_pandas.info()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC  ### Removing non ASCII characters

# COMMAND ----------

def remove_non_ascii(text): 
    return ''.join(i for i in text if ord(i)<128) 
 

data_pandas['text'] = data_pandas['text'].apply(remove_non_ascii) 

# COMMAND ----------

data_pandas.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) New Dataset

# COMMAND ----------

df = spark.createDataFrame(data_pandas)
df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Spark NLP

# COMMAND ----------

import sparknlp

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

from pyspark.ml import Pipeline

from sparknlp.pretrained import PretrainedPipeline

# COMMAND ----------

pipeline = PretrainedPipeline("analyze_sentimentdl_use_twitter", lang="en")

# COMMAND ----------

df_result = pipeline.transform(df)

# COMMAND ----------

display(df_result)

# COMMAND ----------

df_result = df_result.selectExpr("text","real_sent","sentiment.result[0] result")

# COMMAND ----------

display(df_result)

# COMMAND ----------

df_result.printSchema()

# COMMAND ----------

from pyspark.sql.types import StringType

df_result = df_result.withColumn("real_sent",df_result["real_sent"].cast(StringType()))
df_result.printSchema()

# COMMAND ----------

from pyspark.sql.functions import regexp_replace

df_result = df_result.withColumn("real_sent", regexp_replace("real_sent", "0", "negative"))
df_result = df_result.withColumn("real_sent", regexp_replace("real_sent", "4", "positive"))
df_result = df_result.withColumn("real_sent", regexp_replace("real_sent", "2", "neutral"))

# COMMAND ----------

display(df_result)

# COMMAND ----------

import matplotlib.pyplot as plt

df_plot = df_result.toPandas()
print(df_plot.real_sent.value_counts())
df_plot.real_sent.value_counts().plot(kind='bar')
plt.show()

# COMMAND ----------

print(df_plot.result.value_counts())
df_plot.result.value_counts().plot(kind='bar')
plt.show()

# COMMAND ----------

import numpy as np               
import matplotlib.pyplot as plt   

# Create dictionaries from lists with this format: 'letter':count
dict1 = dict(zip(*np.unique(df_plot.real_sent, return_counts=True)))
dict2 = dict(zip(*np.unique(df_plot.result, return_counts=True)))

# Add missing letters with count=0 to each dictionary so that keys in
# each dictionary are identical
only_in_set1 = set(dict1)-set(dict2)
only_in_set2 = set(dict2)-set(dict1)
dict1.update(dict(zip(only_in_set2, [0]*len(only_in_set2))))
dict2.update(dict(zip(only_in_set1, [0]*len(only_in_set1))))

# Sort dictionaries alphabetically
dict1 = dict(sorted(dict1.items()))
dict2 = dict(sorted(dict2.items()))

# Create grouped bar chart
xticks = np.arange(len(dict1))
bar_width = 0.3
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(xticks-bar_width/2, dict1.values(), bar_width,
       color='blue', alpha=0.5, label='Dataset Prediction Base')
ax.bar(xticks+bar_width/2, dict2.values(), bar_width,
       color='red', alpha=0.5, label='SparkNLP Prediction')

# Set annotations, x-axis ticks and tick labels
ax.set_ylabel('Counts')
ax.set_title('Comparison of Sentiment anlaysis between SparkNLP and the anlysed given by the Dataset')
ax.set_xticks(xticks)
ax.set_xticklabels(dict1.keys())
ax.legend(frameon=False)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC  # Reste a faire
# MAGIC  
# MAGIC  #### montrer les avantages de Databricks (notebook en cascade, git, etc ...)
# MAGIC  #### faire une note pour expliquer le projet

# COMMAND ----------

