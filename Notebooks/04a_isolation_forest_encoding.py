# Databricks notebook source
# MAGIC %pip install kneed

# COMMAND ----------

from pyspark.sql import functions as sf
from pyspark.sql import types as st
import Levenshtein
from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType, StringType, StructType, StructField
from pyspark.sql.utils import AnalysisException
from dqa.attributes.address import COUNTRIES
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import math
from sklearn.ensemble import IsolationForest
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array
from pyspark.sql.window import Window
from dqa.modelling.clustering import KMeansPostProcessed
from dqa.modelling.encoding import SimilarityBasedEncoder
from dqa import logging, schema
logger = logging.make_logger(__name__)

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "false")

# COMMAND ----------

dbutils.widgets.dropdown(
    name="country_code",
    defaultValue=str(list(COUNTRIES.keys())[0]),
    choices=list(COUNTRIES.keys()),
)
dbutils.widgets.dropdown(
    name="source_schema",
    defaultValue=schema.DELTA_SCHEMAS[0],
    choices=schema.DELTA_SCHEMAS,
)
dbutils.widgets.dropdown(
    name="target_schema",
    defaultValue=schema.DELTA_SCHEMAS[1],
    choices=schema.DELTA_SCHEMAS,
)

dbutils.widgets.text("tag", "")

# COMMAND ----------

try:
    SOURCE_SCHEMA = dbutils.widgets.get("source_schema")
except ValueError as e:
    raise e
else:
    assert (
        SOURCE_SCHEMA in schema.DELTA_SCHEMAS
    ), logger.warning("Source schema must be in the list of values {schema.DELTA_SCHEMAS}")

try:
    TARGET_SCHEMA = dbutils.widgets.get("target_schema")
except ValueError as e:
    raise e
else:
    assert (
        TARGET_SCHEMA in schema.DELTA_SCHEMAS
    ), logger.warning("Target schema must be in the list of values {schema.DELTA_SCHEMAS}")

COUNTRY_CODE = dbutils.widgets.get("country_code")
RUNTAG = dbutils.widgets.get("tag")

# COMMAND ----------

CMD_SOURCE_TABLE = (
    f"{schema.CATALOGS[0]}.{SOURCE_SCHEMA}.tbl_cleansing_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_cmd"
)
COLS_CMD = [
    "id__cmd",
    "city__cmd",
    "lat__cmd",
    "long__cmd",

]

# COMMAND ----------

try:
    cmd_df = spark.read.table(CMD_SOURCE_TABLE)
except AnalysisException as e:
    # Handle the case where the table is not found or is not a Delta table
    cmd_df = None

cmd_df = cmd_df.select(COLS_CMD)

# COMMAND ----------

logger.warning(
    f"""Running with the following conf:
    country code: {COUNTRY_CODE}
    source schema: {SOURCE_SCHEMA}
    target schema: {TARGET_SCHEMA}
    Reading from {CMD_SOURCE_TABLE}
    """
)

# COMMAND ----------

STD_CITY_SOURCE_TABLE = (
    f"{schema.CATALOGS[0]}.dqa_gold.tbl_fuzzy_st_stacked_{COUNTRY_CODE.lower()}{RUNTAG.lower()}"
)

try:
    city_df = spark.read.table(STD_CITY_SOURCE_TABLE)
except AnalysisException as e:
    # Handle the case where the table is not found or is not a Delta table
    city_df = None

# COMMAND ----------

# Create a new column based on the condition
city_df = city_df.withColumn(
    "standardized_city",
    sf.when(
        sf.col("proposed_city") == "to be manually selected",
        sf.concat_ws(
            " ", sf.first("city__cmd").over(Window.partitionBy("cluster")), sf.lit("*")
        ),
    ).otherwise(sf.col("proposed_city")),
)


# COMMAND ----------

# Step 4: Join the original DataFrame with the reordered DataFrame
cmd_df = cmd_df.join(city_df, cmd_df.city__cmd == city_df.city__cmd, "left").select(
    cmd_df["*"], city_df["standardized_city"]
)

cmd_df = cmd_df.withColumn(
    "standardized_city",
    sf.when(
        sf.col("standardized_city").isNotNull(), sf.col("standardized_city")
    ).otherwise(sf.col("city__cmd")),
)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Creating KMeans cluster

# COMMAND ----------

#clustering
input_list = ["lat__cmd", "long__cmd"]
model = KMeansPostProcessed(geo_cols=input_list)
clustered_df = model.fit(cmd_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating encoding

# COMMAND ----------

#encoding
model = SimilarityBasedEncoder()
encoded_df = model.fit(clustered_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save results

# COMMAND ----------

OUTPUT_TABLE_NAME = f"{schema.CATALOGS[0]}.{TARGET_SCHEMA}.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_encoding_cmd"

logger.warning(
    f"""Saving the results
    """
)

(
    encoded_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE_NAME)
)
