# Databricks notebook source
from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType, StringType, StructType
from pyspark.sql.utils import AnalysisException
from dqa.attributes.address import COUNTRIES
from pyspark.sql.window import Window
import numpy as np
import pandas as pd
from dqa import logging, schema
import folium
import branca.colormap as cm
logger = logging.make_logger(__name__)

# COMMAND ----------

dbutils.widgets.dropdown(
    name="country_code",
    defaultValue=str(list(COUNTRIES.keys())[0]),
    choices=list(COUNTRIES.keys()),
)
dbutils.widgets.dropdown(
    name="source_schema",
    defaultValue=schema.DELTA_SCHEMAS[3],
    choices=schema.DELTA_SCHEMAS,
)
dbutils.widgets.dropdown(
    name="target_schema",
    defaultValue=schema.DELTA_SCHEMAS[3],
    choices=schema.DELTA_SCHEMAS,
)
dbutils.widgets.text("tag", "")
dbutils.widgets.text("select_top_scores", "True")
dbutils.widgets.text("percentage_of_geocoords_anomalies", "0.5")
dbutils.widgets.text("percentage_of_cities_anomalies", "5")
dbutils.widgets.text("top_k_scores_city", "1000")
dbutils.widgets.text("top_k_scores_coordinates", "5")

# COMMAND ----------

try:
    SOURCE_SCHEMA = dbutils.widgets.get("source_schema")
except ValueError as e:
    raise e
else:
    assert SOURCE_SCHEMA in schema.DELTA_SCHEMAS, logger.warning(
        "Source schema must be in the list of values {schema.DELTA_SCHEMAS}"
    )

try:
    TARGET_SCHEMA = dbutils.widgets.get("target_schema")
except ValueError as e:
    raise e
else:
    assert TARGET_SCHEMA in schema.DELTA_SCHEMAS, logger.warning(
        "Target schema must be in the list of values {schema.DELTA_SCHEMAS}"
    )

COUNTRY_CODE = dbutils.widgets.get("country_code")
RUNTAG = dbutils.widgets.get("tag")
SELECT_TOP_SCORES = dbutils.widgets.get("select_top_scores")


# COMMAND ----------

CMD_SOURCE_TABLE = f"{schema.CATALOGS[0]}.{SOURCE_SCHEMA}.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_cmd"

OUTPUT_TABLE_NAME = f"{schema.CATALOGS[0]}.{TARGET_SCHEMA}.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_results_cmd"


# COMMAND ----------

try:
    cmd_df = spark.read.table(CMD_SOURCE_TABLE)
except AnalysisException as e:
    # Handle the case where the table is not found or is not a Delta table
    cmd_df = None

# COMMAND ----------

ISOLATED_SOURCE_TABLE = f"{schema.CATALOGS[0]}.dqa_silver.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_isolated_cmd"
try:
    df = spark.read.table(ISOLATED_SOURCE_TABLE).select(
        [
            "id__cmd",
            "is_isolated",
            "neighbors_count",
            "closest_city__cmd",
            "closest_distance_m",
            "closest_id__cmd",
        ]
    )
except AnalysisException as e:
    # Handle the case where the table is not found or is not a Delta table
    df = None

cmd_df = cmd_df.join(df, on="id__cmd", how="left").drop(
    *[col for col in cmd_df.columns if col.startswith("st_attr")]
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Results

# COMMAND ----------

if SELECT_TOP_SCORES == "True":
    TOP_K_COORDS = int(dbutils.widgets.get("top_k_scores_coordinates"))
    TOP_K_CITY = int(dbutils.widgets.get("top_k_scores_city"))

    # Define a window specification to rank S_Score_coordinates
    window_spec = Window.orderBy("S_Score_coordinates")

    # Add a rank column based on S_Score_coordinates
    cmd_df = cmd_df.withColumn("rank", sf.rank().over(window_spec))

    # Add the is_anomaly_coords column based on the TOP_K_COORDS
    cmd_df = cmd_df.withColumn(
        "is_anomaly_coords", sf.col("rank") <= TOP_K_COORDS
    )

    # Define a window specification to rank S_Score_city_cluster
    window_spec = Window.orderBy("S_Score_city_cluster")

    # Add a rank column based on S_Score_city
    cmd_df = cmd_df.withColumn("rank", sf.rank().over(window_spec))

    # Add the is_anomaly_coords column based on the TOP_K_COORDS
    cmd_df = cmd_df.withColumn(
        "is_anomaly_city_cluster", (sf.col("rank") <= TOP_K_CITY) & (sf.col("is_isolated") != 1),
    )

    # Define a window specification to rank TOP_K_CITY
    window_spec = Window.orderBy("S_Score_city")

    # Add a rank column based on S_Score_city
    cmd_df = cmd_df.withColumn("rank", sf.rank().over(window_spec))

    # Add the is_anomaly_coords column based on the TOP_K_CITY
    cmd_df = cmd_df.withColumn(
        "is_anomaly_city", (sf.col("rank") <= TOP_K_CITY) & (sf.col("is_isolated") != 1),
    )
    cmd_df = cmd_df.drop("rank")

else:
    # Calculate the 1st percentile threshold
    THRESHOLD = float(dbutils.widgets.get("percentage_of_geocoords_anomalies"))/100

    percentile_threshold = cmd_df.approxQuantile("S_Score_coordinates", [THRESHOLD], 0.0)[0]
    logger.warning(
        f"Threshold for geocoords Score {THRESHOLD} percentile: {percentile_threshold}"
    )
    # Add the is_anomaly_coords column based on the threshold
    cmd_df = cmd_df.withColumn(
        "is_anomaly_coords", sf.col("S_Score_coordinates") <= percentile_threshold
    )
    THRESHOLD = float(dbutils.widgets.get("percentage_of_cities_anomalies"))/100
    # Filter out rows where is_isolated = 1
    filtered_df = cmd_df.filter(sf.col("is_isolated") != 1)

    # Calculate the 1st percentile threshold on the filtered data
    percentile_threshold = filtered_df.approxQuantile("S_Score_city", [THRESHOLD], 0.0)[0]
    logger.warning(
        f"Threshold for S_Score_city {THRESHOLD} percentile: {percentile_threshold}"
    )

    # Add the is_anomaly_coords column based on the threshold in the original DataFrame
    cmd_df = cmd_df.withColumn(
        "is_anomaly_city",
        (sf.col("S_Score_city") <= percentile_threshold) & (sf.col("is_isolated") != 1),
    )

    # Calculate the 1st percentile threshold
    percentile_threshold = filtered_df.approxQuantile(
        "S_Score_city_cluster", [THRESHOLD], 0.0
    )[0]
    logger.warning(
        f"Threshold for S_Score_city_cluster {THRESHOLD} percentile: {percentile_threshold}"
    )

    # Add the is_anomaly_coords column based on the threshold
    cmd_df = cmd_df.withColumn(
        "is_anomaly_city_cluster",
        (sf.col("S_Score_city_cluster") <= percentile_threshold)
        & (sf.col("is_isolated") != 1),
    )


# COMMAND ----------

cmd_df = cmd_df.withColumn(
    "tag_anomalies",
    sf.when(
        (sf.col("city__cmd") != sf.col("standardized_city")) & (sf.col("is_anomaly_city_cluster") == 0),
        "case 1 - city to be standardized"
    ).when(
        (sf.col("city__cmd") != sf.col("standardized_city")) & (sf.col("is_anomaly_city_cluster") == 1),
        "case 2 - both city standardization and coords to be checked"
    ).when(
        (sf.col("city__cmd") == sf.col("standardized_city")) & (sf.col("is_anomaly_city_cluster") == 1),
        "case 3 - inconsistency between city name and coords"
    ).when(
        (sf.col("is_anomaly_coords") == 1),
        "case 4 - out of Country"
    )
)

# COMMAND ----------

#Coords Isolation forest
col_score = "S_Score_coordinates"
col_anomaly = "is_anomaly_coords"
db = cmd_df.toPandas()
db = db.set_index("id__cmd")
db = db.sort_values(by = col_score)
db.hist(column = col_score, bins= 20)

# COMMAND ----------

test = db.loc[db[col_anomaly] == True]
# Create a colormap
colormap = cm.linear.YlOrRd_09.scale(test[col_score].min(), test[col_score].max())
colormap.caption = col_score


# Initialize the map centered around the approximate center of the points
m = folium.Map(location=[20, 0], zoom_start=2)


# Function to convert S_Score to radius
def get_radius(score):
    return abs(score) * 15  # Adjust the multiplier as needed


# Add the points to the map
for idx, row in test.iterrows():
    radius = get_radius(row[col_score])
    color = colormap(row[col_score])

    # Create a popup with information about the point
    popup_text = f"ID: {idx} <br> City CMD:{row['city__cmd']} <br>{row['lat__cmd']}<br>{row['long__cmd']}<br>S_Score: {row[col_score]}<br>is_isolated: {row['is_isolated']}<br>n_neighbours: {row['neighbors_count']}"
    popup = folium.Popup(popup_text, max_width=300)

    # Add the circle marker with the popup
    folium.CircleMarker(
        location=[row["lat__cmd"], row["long__cmd"]],
        radius=radius,
        color="black",  # Border color
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        weight=1,
        popup=popup,  # Add the popup here
    ).add_to(m)

# Add colormap to map
colormap.add_to(m)

# Display the map in a Jupyter Notebook (if running in a notebook)
m


# COMMAND ----------

col_score = "S_Score_city_cluster"
col_anomaly = "is_anomaly_city_cluster"
db = db.sort_values(by = col_score)
db[db["is_isolated"] == 0].hist(column = col_score, bins= 20)

# COMMAND ----------

test = db.loc[db[col_anomaly] == 1]
# Create a colormap
colormap = cm.linear.YlOrRd_09.scale(test[col_score].min(), test[col_score].max())
colormap.caption = col_score


# Initialize the map centered around the approximate center of the points
m = folium.Map(location=[20, 0], zoom_start=2)


# Function to convert S_Score to radius
def get_radius(score):
    return abs(score) * 15  # Adjust the multiplier as needed


# Add the points to the map
for idx, row in test.iterrows():
    radius = get_radius(row[col_score])
    color = colormap(row[col_score])

    # Create a popup with information about the point
    popup_text = f"ID: {idx} <br> City CMD:{row['city__cmd']}<br>S_Score: {row[col_score]}<br>is_isolated: {row['is_isolated']}<br>n_neighbours: {row['neighbors_count']}<br>cluster: {row['most_common_cluster']}<br>City of closest cust: {row['closest_city__cmd']}<br>Distance m of closest cust: {row['closest_distance_m']}"
    popup = folium.Popup(popup_text, max_width=300)

    # Add the circle marker with the popup
    folium.CircleMarker(
        location=[row["lat__cmd"], row["long__cmd"]],
        radius=radius,
        color="black",  # Border color
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        weight=1,
        popup=popup,  # Add the popup here
    ).add_to(m)

# Add colormap to map
colormap.add_to(m)

# Display the map in a Jupyter Notebook (if running in a notebook)
m


# COMMAND ----------

# MAGIC %md
# MAGIC ##Final Cases

# COMMAND ----------

test = db.loc[(db["tag_anomalies"].notnull())]
col_score = "S_Score_city_cluster"
# Create a colormap
colormap = cm.linear.YlOrRd_09.scale(test[col_score].min(), test[col_score].max())
colormap.caption = col_score

# Initialize the map centered around the approximate center of the points
m = folium.Map(location=[20, 0], zoom_start=2)

# Function to convert S_Score to radius
def get_radius(score):
    return abs(score) * 15  # Adjust the multiplier as needed

# Add the points to the map
for idx, row in test.iterrows():
    radius = get_radius(row[col_score])
    color = colormap(row[col_score])
    
    # Create a popup with information about the point
    popup_text = f"ID: {idx} <br> City CMD:{row['city__cmd']}<br> Proposed City:{row['standardized_city']}<br>S_Score: {row[col_score]}<br>is_isolated: {row['is_isolated']}<br>n_neighbours: {row['neighbors_count']}<br>cluster: {row['most_common_cluster']}<br>City of closest cust: {row['closest_city__cmd']}<br>Distance m of closest cust: {row['closest_distance_m']}<br>Tag: {row['tag_anomalies']}"
    popup = folium.Popup(popup_text, max_width=300)
    
    # Add the circle marker with the popup
    folium.CircleMarker(
        location=[row['lat__cmd'], row['long__cmd']],
        radius=radius,
        color='black',  # Border color
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        weight=1,
        popup=popup  # Add the popup here
    ).add_to(m)

# Add colormap to map
colormap.add_to(m)

# Save the map as an HTML file
m


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create final tag and save the results

# COMMAND ----------

final_df = cmd_df.filter(sf.col("tag_anomalies").isNotNull())

(
    final_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE_NAME)
)
