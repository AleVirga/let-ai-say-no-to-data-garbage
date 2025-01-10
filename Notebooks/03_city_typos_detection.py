# Databricks notebook source
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructType,
    StructField,
    IntegerType,
    ArrayType
)
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import pandas_udf, PandasUDFType, udf
from fuzzywuzzy import process, fuzz
import numpy as np
import pandas as pd
import jellyfish
from dqa.attributes.address import COUNTRIES
from dqa.configurators import APP
from dqa import logging, schema
import networkx as nx
from pyspark.sql.window import Window
import re
import Levenshtein

logger = logging.make_logger(__name__)


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
    defaultValue=schema.DELTA_SCHEMAS[3],
    choices=schema.DELTA_SCHEMAS,
)
dbutils.widgets.text("tag", "")
dbutils.widgets.text("similarity_threshold_fuzzy", "92")
dbutils.widgets.text("similarity_threshold_jellyfish", "85")

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
SIMILARITY_THRESHOLD_FUZZY = float(dbutils.widgets.get("similarity_threshold_fuzzy"))
SIMILARITY_THRESHOLD_JELLYFISH= float(dbutils.widgets.get("similarity_threshold_jellyfish"))

# COMMAND ----------

CMD_SOURCE_TABLE = (
    f"{schema.CATALOGS[0]}.{SOURCE_SCHEMA}.tbl_cleansing_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_cmd"
)
COLS_CMD = [
    "id__cmd",
    "city__cmd",
    "post_code__cmd",
]
SUMMARY_TABLE_NAME = f"{schema.CATALOGS[0]}.dqa_gold.tbl_fuzzy_st_summary_{COUNTRY_CODE.lower()}{RUNTAG.lower()}"
STACKED_TABLE_NAME = f"{schema.CATALOGS[0]}.dqa_gold.tbl_fuzzy_st_stacked_{COUNTRY_CODE.lower()}{RUNTAG.lower()}"
DETAIL_TABLE_NAME = f"{schema.CATALOGS[0]}.dqa_gold.tbl_fuzzy_st_details_{COUNTRY_CODE.lower()}{RUNTAG.lower()}"

# COMMAND ----------

try:
    cmd_df = spark.read.table(CMD_SOURCE_TABLE)
except AnalysisException as e:
    # Handle the case where the table is not found or is not a Delta table
    cmd_df = None

cmd_df = cmd_df.select(COLS_CMD)

# COMMAND ----------

# Get the distinct values of the city__cmd column
unique_cities_df = cmd_df.select("city__cmd").distinct()

# Collect the results as a list of unique values
unique_names = [row.city__cmd for row in unique_cities_df.collect()]

# Filter out NA values early
unique_names = [name for name in unique_names if not pd.isna(name)]

# Dictionary for the most common name for each group of similar names
name_mapping = {}
name_mapping = {name: name for name in unique_names}
changes = []  # To store the changes made

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check city standardization

# COMMAND ----------

for name in unique_names:
    if pd.isna(name):
        continue
    if not name_mapping:  # Initialize with the first name if empty
        name_mapping[name] = name
        continue
    # Fuzzy matching to find the closest match in the mapping
    selected_keys = [key for key in name_mapping.keys() if key != name]
    match = process.extractOne(name, selected_keys, scorer=fuzz.WRatio)
    if match:
        matched_name, similarity = match
        if similarity >= SIMILARITY_THRESHOLD_FUZZY:
            name_mapping[name] = name_mapping[matched_name]
            if name != name_mapping[matched_name]:
                changes.append(
                    {
                        "city__cmd": name,
                        "matched_city": name_mapping[matched_name],
                        "similarity_score_fuzzy": similarity,
                    }
                )
        else:
            name_mapping[name] = name
    else:
        name_mapping[name] = name


# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Jaro Winkler Similarity to deal with short string

# COMMAND ----------

unique_cities_pdf = pd.DataFrame(changes)

# COMMAND ----------

unique_cities_pdf["similarity_score_jellyfish"] = unique_cities_pdf.apply(
    lambda row: jellyfish.jaro_winkler_similarity(row["city__cmd"], row["matched_city"])
    * 100,
    axis=1,
)
unique_cities_pdf = unique_cities_pdf[
    unique_cities_pdf["similarity_score_jellyfish"] >= SIMILARITY_THRESHOLD_JELLYFISH
]


# COMMAND ----------

# MAGIC %md
# MAGIC ##Creating the clusters

# COMMAND ----------

# Create an empty graph
G = nx.Graph()

# Add edges to the graph
edges = list(zip(unique_cities_pdf['city__cmd'], unique_cities_pdf['matched_city']))
G.add_edges_from(edges)

# Find connected components
connected_components = list(nx.connected_components(G))

# Create a mapping from city to cluster id
city_to_cluster = {}
for cluster_id, component in enumerate(connected_components):
    for city in component:
        city_to_cluster[city] = cluster_id

# Map the clusters back to the DataFrame
unique_cities_pdf['cluster'] = unique_cities_pdf['city__cmd'].map(city_to_cluster)

# COMMAND ----------

result_df = spark.createDataFrame(unique_cities_pdf)

# COMMAND ----------

result_df.write.mode("overwrite").format("delta").option(
    "delta.columnMapping.mode", "name"
).option("overwriteSchema", "true").saveAsTable(SUMMARY_TABLE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare stacked version

# COMMAND ----------

sel_df_1 = result_df.select(["city__cmd", "cluster"])
sel_df_2 = result_df.select(["matched_city", "cluster"]).withColumnRenamed(
    "matched_city", "city__cmd"
)
stacked_df = sel_df_1.union(sel_df_2).dropDuplicates()
city_df = cmd_df.groupBy(["city__cmd"]).agg(sf.count("city__cmd").alias("city_count"))
final_stacked_df = stacked_df.join(city_df, "city__cmd", "inner").orderBy("cluster")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Suggest the majority as correction

# COMMAND ----------

# Define a window specification to rank cities by city_count within each postal_code
cityWindowSpec = Window.partitionBy("cluster").orderBy(sf.desc("city_count"))

# Rank cities by city_count within each postal_code
df_ranked = final_stacked_df.withColumn("rank", sf.row_number().over(cityWindowSpec))

# Define another window specification to count city_count occurrences within each postal_code
countWindowSpec = Window.partitionBy("cluster", "city_count")

# Add a column to count occurrences of each city_count within each postal_code
df_counted = df_ranked.withColumn("count", sf.count("city_count").over(countWindowSpec))

# Set the rank to None if there is a tie in city_count for rank 1
df_city_max = (
    df_counted.withColumn(
        "final_rank",
        sf.when((sf.col("rank") == 1) & (sf.col("count") > 1), None).otherwise(
            sf.col("rank")
        ),
    )
    .filter(sf.col("final_rank") == 1)
    .drop("rank", "count")
    .withColumnRenamed("final_rank", "rank")
)


# COMMAND ----------

MSG_MANUAL_SEL = "to be manually selected"
# Select only the columns of interest
df_city_max = df_city_max.select(
    "cluster", sf.col("city__cmd").alias("proposed_city")
)

# Join the aggregated data back to the original DataFrame
final_stacked_df = final_stacked_df.join(df_city_max, on="cluster", how="left")

# Add message in case of no proposed city
final_stacked_df = final_stacked_df.withColumn(
    "proposed_city",
    sf.when(sf.col("proposed_city").isNull(), sf.lit(MSG_MANUAL_SEL)).otherwise(
        sf.col("proposed_city")
    ),
)

# COMMAND ----------

# Check if proposed city is valid, otherwise replace with MSG_MANUAL_SEL
def is_invalid(city):
    return 1 if city is None or bool(re.match(r".*\?", city)) else 0

# Create the UDF
city_invalid_udf = udf(is_invalid, IntegerType())


final_stacked_df = final_stacked_df.withColumn(
    "is_invalid_proposed_city", city_invalid_udf(sf.col("proposed_city"))
)

final_stacked_df = (
    final_stacked_df.withColumn(
        "proposed_city",
        sf.when(
            sf.col("is_invalid_proposed_city") == 1, sf.lit(MSG_MANUAL_SEL)
        ).otherwise(sf.col("proposed_city")),
    )
    .orderBy("cluster")
    .drop("is_invalid_proposed_city")
    .cache()
)


# COMMAND ----------

final_stacked_df.write.mode("overwrite").format("delta").option(
    "delta.columnMapping.mode", "name"
).option("overwriteSchema", "true").saveAsTable(STACKED_TABLE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Details customer

# COMMAND ----------

detail_df = cmd_df.join(final_stacked_df, on = "city__cmd", how = "inner")
detail_df = detail_df.withColumn(
    "apply_changes",
    sf.when(
        (sf.col("city__cmd") != sf.col("proposed_city"))
        & (sf.col("proposed_city") != MSG_MANUAL_SEL),
        1,
    ).otherwise(sf.when(sf.col("proposed_city") == MSG_MANUAL_SEL, None).otherwise(0)),
)

# COMMAND ----------

detail_df.write.mode("overwrite").format("delta").option(
    "delta.columnMapping.mode", "name"
).option("overwriteSchema", "true").saveAsTable(DETAIL_TABLE_NAME)
