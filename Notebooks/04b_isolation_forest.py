# Databricks notebook source
from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType, StringType, StructType, StructField
from pyspark.sql.utils import AnalysisException
from pyspark.ml.feature import VectorAssembler, StandardScaler
from dqa.attributes.address import COUNTRIES
import numpy as np
import pandas as pd
from dqa.scoring import DistanceMatching
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array
from dqa.modelling.isolationforest import IsolationForestWithPreprocessing
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
    defaultValue=schema.DELTA_SCHEMAS[1],
    choices=schema.DELTA_SCHEMAS,
)
dbutils.widgets.dropdown(
    name="target_schema",
    defaultValue=schema.DELTA_SCHEMAS[3],
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
    f"{schema.CATALOGS[0]}.{SOURCE_SCHEMA}.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_encoding_cmd"
)

# COMMAND ----------

try:
    cmd_df = spark.read.table(CMD_SOURCE_TABLE)
except AnalysisException as e:
    # Handle the case where the table is not found or is not a Delta table
    cmd_df = None

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

# MAGIC %md
# MAGIC ## Anomalies in coords - Isolation Forest

# COMMAND ----------

logger.warning(
    f"""Running isolation forest with lat and lon
    """
)

input_list = ["lat__cmd", "long__cmd"]  # Specify the input columns
output_name = "S_Score_coordinates"  # Specify the output column name

model = IsolationForestWithPreprocessing(partition_cols = None, input_cols=input_list, output_col=output_name)
cmd_df = model.fit(cmd_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Isolation Forest with City (methodology B - full db)

# COMMAND ----------

logger.warning(
    f"""Running isolation forest with lat and lon and index
    """
)

input_list = ["lat__cmd", "long__cmd", "index"]  # Specify the input columns
output_name = "S_Score_city"  # Specify the output column name

model = IsolationForestWithPreprocessing(partition_cols = None, input_cols=input_list, output_col=output_name)
cmd_df = model.fit(cmd_df)

# COMMAND ----------


logger.warning(
    f"""Running isolation forest with lat and lon and index (partitionBy Cluster)
    """
)

input_list = ["lat__cmd", "long__cmd", "index"]  # Specify the input columns
output_name = "S_Score_city_cluster"  # Specify the output column name

model = IsolationForestWithPreprocessing(partition_cols = "most_common_cluster", input_cols=input_list, output_col=output_name)
cmd_df = model.fit(cmd_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save results

# COMMAND ----------

OUTPUT_TABLE_NAME = f"{schema.CATALOGS[0]}.{TARGET_SCHEMA}.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_cmd"

logger.warning(
    f"""Saving the results
    """
)

(
    cmd_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE_NAME)
)
