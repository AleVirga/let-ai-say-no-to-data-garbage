# Databricks notebook source
# MAGIC %md
# MAGIC ## Visualization of errors in data

# COMMAND ----------

# MAGIC %pip install contextily

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType, StringType, StructType
from pyspark.sql.utils import AnalysisException
from dqa.attributes.address import COUNTRIES
from pyspark.sql.window import Window
import numpy as np
import pandas as pd
from dqa import logging, schema
import folium
import contextily as ctx
import numpy as np
from matplotlib import colors
import matplotlib.gridspec as gs
from sklearn.metrics import precision_recall_curve, f1_score
import scipy.special as sss
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
import os
from os.path import abspath
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
from collections import deque
from itertools import product
import sys
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

# COMMAND ----------

ERROR_SOURCE_TABLE = f"{schema.CATALOGS[0]}.dqa_bronze.tbl_cleansing_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_cmd"
err_df = spark.read.table(ERROR_SOURCE_TABLE)
from functools import reduce
# List of columns to check
columns_to_check = [
    "shuffle_postcode",
    "shuffle_lat",
    "shuffle_long",
    "swapped_geo",
    "typo_postcode"
]

# Create the 'is_error' column
err_df = err_df.withColumn(
    "is_error",
    sf.when(
        reduce(
            lambda x, y: x | y, 
            [sf.col(col_name) for col_name in columns_to_check]
        ),
        True
    ).otherwise(False)
)

# Create the `index_error` column
err_df = err_df.withColumn(
    "index_error",
    sf.when(sf.col(columns_to_check[0]), 1)  # Check the first column
    .when(sf.col(columns_to_check[1]), 2)  # Check the second column
    .when(sf.col(columns_to_check[2]), 3)  # Check the third column
    .when(sf.col(columns_to_check[3]), 4)  # Check the fourth column
    .when(sf.col(columns_to_check[4]), 5)  # Check the fifth column
    .otherwise(None)  # If none are true, return null
)

is_error_df = err_df.filter(sf.col("is_error") == True)

# COMMAND ----------

grouped_df = err_df.groupBy(columns_to_check).count()

# COMMAND ----------

import pandas as pd
import plotly.express as px
df = is_error_df.toPandas()

# Create a map with Plotly
fig = px.scatter_mapbox(
    df,
    lat="lat__cmd",
    lon="long__cmd",
    color="index_error",
    color_discrete_sequence=px.colors.qualitative.Set2,
    zoom=5,
    mapbox_style="carto-positron",
    title="Data points with errors"
)

# Adjust the figure layout for a vertical map
fig.update_layout(
    width=1000,  # Narrow width for a vertical look
    height=1000  # Increase height for more vertical space
)

# Show the map
fig.show()

# COMMAND ----------

CMD_SOURCE_TABLE = f"{schema.CATALOGS[0]}.{SOURCE_SCHEMA}.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_cmd"

OUTPUT_TABLE_NAME = f"{schema.CATALOGS[0]}.{TARGET_SCHEMA}.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_results_cmd"

# COMMAND ----------

try:
    cmd_df = spark.read.table(CMD_SOURCE_TABLE)
except AnalysisException as e:
    # Handle the case where the table is not found or is not a Delta table
    cmd_df = None

cmd_df = cmd_df.join(err_df.select(["id__cmd", "is_error"]), on="id__cmd", how = "left")

# COMMAND ----------

display(cmd_df.groupBy("most_common_cluster").count())

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

df = df.withColumn(
        "is_isolated", sf.when((sf.col("neighbors_count").isNotNull()) & (sf.col("neighbors_count") > 4), False).otherwise(True))
        
cmd_df = cmd_df.join(df, on="id__cmd", how="left").drop(
    *[col for col in cmd_df.columns if col.startswith("st_attr")]
)


# COMMAND ----------

db = cmd_df.toPandas()
db = db.set_index("id__cmd")
db["value_score"] = abs(db['S_Score_city_cluster'])
Q1 = db['value_score'].quantile(0.25)
Q3 = db['value_score'].quantile(0.75)
IQR = Q3 - Q1

light_tail_start = -(Q3 + 1.5 * IQR)
logger.warning(f"The light tail starts at: {light_tail_start}")

from scipy.stats import zscore

db['z_score'] = zscore(db['value_score'])
light_tail_start_2 = -abs(db.loc[db['z_score'] > 1.5, 'value_score'].min())
logger.warning(f"The light tail starts at: {light_tail_start_2}")

estimated_threshold = round(min(light_tail_start, light_tail_start_2), 2)
logger.warning(f"Final threshold: {estimated_threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure isolation Forest

# COMMAND ----------

# Define a range of thresholds from -0.9 to -0.4
thresholds = np.linspace(-0.9, -0.4, 101)

# Store results
f1_scores = []
precision_scores = []
recall_scores = []

# Loop through thresholds to compute F1 scores
for threshold in thresholds:
    db["predicted_error"] = db["S_Score_city_cluster"] < threshold  # Predict errors
    precision, recall, thresholds_pr = precision_recall_curve(
        db["is_error"], db["S_Score_city_cluster"]
    )
    closest_idx = np.argmin(np.abs(thresholds_pr - threshold))
    precision_scores.append(precision[closest_idx])
    recall_scores.append(recall[closest_idx])
    f1 = f1_score(db["is_error"], db["predicted_error"])
    f1_scores.append(f1)

# Determine the best threshold
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = np.max(f1_scores)

# Fancy Plot: F1 Score vs Threshold
plt.figure(figsize=(10, 6))

# Plot the F1 scores
plt.plot(
    thresholds, f1_scores, color="teal", linestyle="--", linewidth=2, label="F1 Score"
)

# Highlight the best threshold with a vertical line
plt.axvline(
    x=best_threshold,
    color="red",
    linestyle="-",
    linewidth=2,
    label=f"Max F1 Threshold = {best_threshold:.2f}",
)

# Highlight the best threshold (IQR method) with a vertical line
iqr_threshold = estimated_threshold
thresholds_closer_to_light_tail_start_index = np.abs(
    thresholds - iqr_threshold
).argmin()
f1_score_iqr = f1_scores[thresholds_closer_to_light_tail_start_index]
plt.axvline(
    x=iqr_threshold,
    color="blue",
    linestyle="-",
    linewidth=2,
    label=f"IQR Threshold = {iqr_threshold:.2f}",
)

# Annotate the best F1 score
plt.scatter(
    [best_threshold], [best_f1], color="red", zorder=5
)  # Add a marker at the best point
plt.scatter(
    [iqr_threshold], [f1_score_iqr], color="blue", zorder=5
)  # Add a marker at the best point

# Add title and labels
# plt.title('F1 Score vs Threshold', fontsize=16, fontweight='bold', color='darkblue')
plt.xlabel("Threshold", fontsize=14, fontweight="bold", color="black")
plt.ylabel("Density (S-Score)", fontsize=14, fontweight="bold", color="black")

# Customize ticks and gridlines
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Add legend
plt.legend(fontsize=12, loc="best", shadow=True)

# Set limits for F1 score
plt.ylim(0, 1)

# Display the plot
plt.tight_layout()
plt.show()



# COMMAND ----------

# Fancy Plot: F1 Score vs Threshold
plt.figure(figsize=(10, 6))

# Plot the F1 scores
db["S_Score_city_cluster"].hist(
    bins=20, alpha=0.7, color="gray", label="S-Score Distribution"
)

# Highlight the best threshold with a vertical line
plt.axvline(
    x=best_threshold,
    color="red",
    linestyle="-",
    linewidth=2,
    label=f"Max F1 Threshold = {best_threshold:.2f}",
)

# Highlight the best threshold (IQR method) with a vertical line
iqr_threshold = estimated_threshold
thresholds_closer_to_light_tail_start_index = np.abs(
    thresholds - estimated_threshold
).argmin()
f1_score_iqr = f1_scores[thresholds_closer_to_light_tail_start_index]
plt.axvline(
    x=iqr_threshold,
    color="blue",
    linestyle="-",
    linewidth=2,
    label=f"IQR Threshold = {iqr_threshold:.2f}",
)

# Add title and labels
plt.xlabel("S-Score", fontsize=14, fontweight="bold", color="black")
plt.ylabel("Counts", fontsize=14, fontweight="bold", color="black")

# Customize ticks and gridlines
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Add legend
plt.legend(fontsize=12, loc="best", shadow=True)

# Set limits for F1 score
# plt.ylim(0, 1)

# Display the plot
plt.tight_layout()
plt.show()


# COMMAND ----------

# Create the figure and subplots
fig, axes = plt.subplots(
    2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
)

# Plot 1: F1 Score vs Threshold
axes[0].plot(
    thresholds, f1_scores, color="teal", linestyle="--", linewidth=2, label="F1 Score"
)
axes[0].axvline(
    x=best_threshold,
    color="red",
    linestyle="-",
    linewidth=2,
    label=f"Max F1 Threshold = {best_threshold:.2f}",
)
axes[0].axvline(
    x=estimated_threshold,
    color="blue",
    linestyle="-",
    linewidth=2,
    label=f"Estimated Threshold = {estimated_threshold:.2f}",
)
axes[0].scatter([best_threshold], [best_f1], color="red", zorder=5)
axes[0].scatter(
    [estimated_threshold],
    [f1_scores[np.abs(thresholds - estimated_threshold).argmin()]],
    color="blue",
    zorder=5,
)

# Add title and labels for the first plot
# axes[0].set_title('F1 Score vs Threshold', fontsize=16, fontweight='bold', color='darkblue')
axes[0].set_ylabel("F1 Score", fontsize=14, fontweight="bold", color="black")
axes[0].legend(fontsize=12, loc="best", shadow=True)
axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
axes[0].set_ylim(0, 1)  # Ensure F1 score is within [0, 1]

# Plot 2: Histogram of S-Scores
db["S_Score_city_cluster"].hist(
    ax=axes[1], bins=20, alpha=0.7, color="gray", label="S-Score Distribution"
)
axes[1].axvline(x=best_threshold, color="red", linestyle="-", linewidth=2)
axes[1].axvline(x=estimated_threshold, color="blue", linestyle="-", linewidth=2)

# Add labels for the second plot
axes[1].set_xlabel("S-score", fontsize=14, fontweight="bold", color="black")
axes[1].set_ylabel("Counts", fontsize=14, fontweight="bold", color="black")
axes[1].legend(fontsize=12, loc="best", shadow=True)
axes[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Figure Classifier

# COMMAND ----------

if COUNTRY_CODE == "GB":
#pre
    TP, TN, FP, FN = 1905, 46298, 397, 2890

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1_score = (
    (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
)

# Display results
logger.warning(f"Accuracy: {accuracy:.4f}")
logger.warning(f"Precision: {precision:.4f}")
logger.warning(f"Recall: {recall:.4f}")
logger.warning(f"F1-Score: {f1_score:.4f}")

# Create confusion matrix as a 2x2 matrix
conf_matrix = [[TN, FP], [FN, TP]]

# Calculate confusion matrices for selected thresholds

# Enhanced Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    ax=axes[0],
    annot_kws={"size": 12, "weight": "bold"},
)
# Format annotations for the first heatmap
for text in axes[0].texts:  # Update text annotations in the heatmap
    value = int(text.get_text())
    text.set_text(f"{value:,}")  

axes[0].set_title(f"Isolation Forest", fontsize=14, weight="bold", color="black")
axes[0].set_xlabel("Predicted", fontsize=12, weight="bold", color="black")
axes[0].set_ylabel("Actual", fontsize=12, weight="bold", color="black")
axes[0].tick_params(axis="x", rotation=45, labelsize=10)
axes[0].tick_params(axis="y", labelsize=10)

if COUNTRY_CODE == "GB":
#post
    TP, TN, FP, FN = 1796, 46676, 19, 2999

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1_score = (
    (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
)

# Display results
logger.warning(f"Accuracy: {accuracy:.4f}")
logger.warning(f"Precision: {precision:.4f}")
logger.warning(f"Recall: {recall:.4f}")
logger.warning(f"F1-Score: {f1_score:.4f}")

# Create confusion matrix as a 2x2 matrix
conf_matrix = [[TN, FP], [FN, TP]]

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    ax=axes[1],
    annot_kws={"size": 12, "weight": "bold"},
)

# Format annotations for the second heatmap
for text in axes[1].texts:  # Update text annotations in the heatmap
    value = int(text.get_text())
    text.set_text(f"{value:,}")  # Use dot as the thousands separator
axes[1].set_title(f"Overall Framework", fontsize=14, weight="bold", color="black")
axes[1].set_xlabel("Predicted", fontsize=12, weight="bold", color="black")
axes[1].set_ylabel("Actual", fontsize=12, weight="bold", color="black")
axes[1].tick_params(axis="x", rotation=45, labelsize=10)
axes[1].tick_params(axis="y", labelsize=10)

# plt.suptitle("Confusion Matrices at Selected Thresholds", fontsize=16, weight="bold", color="navy")
plt.show()


# COMMAND ----------

# Select thresholds for confusion matrices
selected_thresholds = [-0.56, -0.53]

# Store confusion matrices
confusion_matrices = {}

# Calculate confusion matrices for selected thresholds
for threshold in selected_thresholds:
    db["predicted_error"] = db["S_Score_city_cluster"] < threshold
    cm = confusion_matrix(db["is_error"], db["predicted_error"])
    confusion_matrices[threshold] = cm

# Enhanced Plot
fig, axes = plt.subplots(
    1, len(selected_thresholds), figsize=(10, 6), constrained_layout=True
)

for i, threshold in enumerate(selected_thresholds):
    cm = confusion_matrices[threshold]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=axes[i],
        annot_kws={"size": 12, "weight": "bold"},
    )
    axes[i].set_title(
        f"Threshold: {threshold:.2f}", fontsize=14, weight="bold", color="darkblue"
    )
    axes[i].set_xlabel("Predicted", fontsize=12, weight="bold", color="black")
    axes[i].set_ylabel("Actual", fontsize=12, weight="bold", color="black")
    axes[i].tick_params(axis="x", rotation=45, labelsize=10)
    axes[i].tick_params(axis="y", labelsize=10)

plt.suptitle(
    "Confusion Matrices at Selected Thresholds",
    fontsize=16,
    weight="bold",
    color="navy",
)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Figure clustering

# COMMAND ----------

# Create a figure with 1 row and 2 columns of subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
db["postprocessed cluster"] = db["most_common_cluster"]
db["KMeans cluster"] = db["cluster"]
# Plot the first subplot
sns.scatterplot(
    x="long__cmd",
    y="lat__cmd",
    hue="KMeans cluster",
    sizes=(50, 200),
    data=db,
    palette="viridis",
    ax=axes[0],
)

# Add basemap using Contextily for the first subplot
ctx.add_basemap(axes[0], crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)

# Set title and labels for the first subplot
axes[0].set_title("KMeans Clustering", fontsize=14, weight="bold", color="black")
axes[0].set_xlabel("Longitude", fontsize=12, weight="bold", color="black")
axes[0].set_ylabel("Latitude", fontsize=12, weight="bold", color="black")

# Plot the second subplot (same as the first one)
sns.scatterplot(
    x="long__cmd",
    y="lat__cmd",
    hue="postprocessed cluster",  # Renamed from 'most_common_cluster' to 'postprocessed_cluster'
    sizes=(50, 200),
    data=db,
    palette="viridis",
    ax=axes[1],
    alpha=0.7,  # Added transparency
)

# Add basemap using Contextily for the second subplot
ctx.add_basemap(axes[1], crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)

# Set title and labels for the second subplot
axes[1].set_title(
    "Post-processed Clustering", fontsize=14, weight="bold", color="black"
)
axes[1].set_xlabel("Longitude", fontsize=12, weight="bold", color="black")
axes[1].set_ylabel("Latitude", fontsize=12, weight="bold", color="black")

# Display the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.show()



