# Databricks notebook source
from pyspark.sql import functions as sf
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_sk
from pyspark.sql.types import DoubleType, StringType, StructType, StructField
from pyspark.sql.functions import col, sum as spark_sum, when
from pyspark.sql.utils import AnalysisException
from dqa.attributes.address import COUNTRIES
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from dqa.modelling.supervisedclassifier import SupervisedAnomalyClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array
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
    defaultValue=schema.DELTA_SCHEMAS[3],
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

FEEDBACK_TABLE_NAME = f"{schema.CATALOGS[0]}.{TARGET_SCHEMA}.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_feedback_cmd"
OUTPUT_TABLE_NAME = f"{schema.CATALOGS[0]}.{TARGET_SCHEMA}.tbl_isolationforest_{COUNTRY_CODE.lower()}{RUNTAG.lower()}_updated_results_cmd"

logger.warning(
    f"""Running with the following conf:
    country code: {COUNTRY_CODE}
    source schema: {SOURCE_SCHEMA}
    target schema: {TARGET_SCHEMA}
    Reading from {FEEDBACK_TABLE_NAME}
    """
)

response = "is_anomaly"
input_columns = [
            "lat__cmd",
            "long__cmd",
            "S_Score_coordinates",
            "S_Score_city_cluster",
            "neighbors_count",
        ]

# COMMAND ----------

try:
    cmd_df = spark.read.table(FEEDBACK_TABLE_NAME)
except AnalysisException as e:
    # Handle the case where the table is not found or is not a Delta table
    cmd_df = None

cmd_df = cmd_df.withColumn(
    "train_or_test",
    sf.when(sf.col(response).isNotNull(), "train").otherwise("test")
)
train_df = cmd_df.filter(sf.col(response).isNotNull())
logger.warning(f"Training data: {train_df.count()}")
test_df = cmd_df.filter(sf.col(response).isNull())
logger.warning(f"Test data: {test_df.count()}")

# COMMAND ----------

# Calculate confusion matrix components
confusion_matrix_df = cmd_df.select(
    when(col("is_error") == 1, 1).otherwise(0).alias("actual_positive"),
    when(col("is_error") == 0, 1).otherwise(0).alias("actual_negative"),
    when(col("id__cmd").isNotNull(), 1).otherwise(0).alias("predicted_positive"),
)

# Aggregate counts for TP, TN, FP, FN
metrics = confusion_matrix_df.agg(
    spark_sum(col("actual_positive") * col("predicted_positive")).alias("TP"),
    spark_sum(col("actual_negative") * col("predicted_positive")).alias("FP"),
).collect()[0]

TP, FP = metrics["TP"], metrics["FP"]

# Calculate metrics
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

# Display results

logger.warning(f"Precision: {precision:.4f}")



# COMMAND ----------

model = SupervisedAnomalyClassifier(class_thresholds = [0.5, 0.5])
model.fit(train_df)
predictions = model.predict(context=None, model_df=train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Feature Importance

# COMMAND ----------

# Access the RandomForest model from the pipeline
rf_model = model.classifier_model.stages[-1]  # Assuming the RandomForestClassifier is the last stage

# Extract feature importances
importances = rf_model.featureImportances.toArray()

# Map the feature importances to the feature names
feature_importance = list(zip(input_columns, importances))

# Sort features by importance
sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

# Display feature importance
logger.warning("Feature Importances:")
for feature, importance in sorted_importance:
    logger.warning(f"Feature: {feature}, Importance: {round(importance, 4)}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpretability

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlation

# COMMAND ----------

##Linear Correlation
def calculate_correlation(df, cols):
    """
    Calculate pairwise correlations between columns in a PySpark DataFrame.

    :param df: PySpark DataFrame
    :param cols: List of column names to compute correlations
    :return: Correlation matrix as a dictionary
    """
    correlation_matrix = {}

    for i in range(len(cols)):
        for j in range(i, len(cols)):  # Avoid redundant calculations
            col1, col2 = cols[i], cols[j]
            if col1 == col2:
                correlation_matrix[(col1, col2)] = 1.0  # Self-correlation is always 1
            else:
                # Compute correlation
                correlation = df.stat.corr(col1, col2)
                correlation_matrix[(col1, col2)] = correlation
                correlation_matrix[(col2, col1)] = correlation  # Symmetry

    return correlation_matrix


def calculate_kendall_correlation(df, cols):
    """
    Calculate pairwise Kendall correlation for the given columns.

    :param df: PySpark or pandas DataFrame
    :param cols: List of column names
    :return: DataFrame with pairwise Kendall correlation coefficients
    """
    # Convert PySpark DataFrame to pandas
    pandas_df = df.select(cols).toPandas() if hasattr(df, "toPandas") else df

    # Initialize an empty DataFrame for correlation matrix
    kendall_matrix = pd.DataFrame(index=cols, columns=cols)

    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                kendall_matrix.loc[col1, col2] = 1.0  # Self-correlation is 1
            else:
                # Compute Kendall Tau correlation
                tau = pandas_df[[col1, col2]].corr(method="kendall").iloc[0, 1]
                kendall_matrix.loc[col1, col2] = tau
    return kendall_matrix.astype(float)


def calculate_normalized_mutual_information(df, cols):
    """
    Calculate pairwise normalized mutual information for the given columns.

    :param df: PySpark or pandas DataFrame
    :param cols: List of column names
    :return: DataFrame with pairwise normalized mutual information
    """
    # Convert PySpark DataFrame to pandas
    pandas_df = df.select(cols).toPandas()

    # Normalize columns to [0, 1] using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(pandas_df), columns=cols)

    # Initialize empty matrix
    nmi_matrix = pd.DataFrame(index=cols, columns=cols)

    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                nmi_matrix.loc[col1, col2] = 1.0  # Self-correlation is 1
            else:
                # Compute mutual information
                mi = mutual_info_regression(
                    normalized_data[[col1]], normalized_data[col2]
                )[0]
                # Normalize mutual information
                h1 = np.log2(len(normalized_data[col1].unique()))  # Entropy of col1
                h2 = np.log2(len(normalized_data[col2].unique()))  # Entropy of col2
                nmi = mi / max(h1, h2)  # Normalized mutual information
                nmi_matrix.loc[col1, col2] = nmi

    return nmi_matrix.astype(float)


nmi_matrix = calculate_normalized_mutual_information(train_df, input_columns)
kendall_matrix = calculate_kendall_correlation(train_df, input_columns)
correlation_matrix = calculate_correlation(train_df, input_columns)


# COMMAND ----------

def plot_combined_heatmaps(correlation_matrix, kendall_matrix, nmi_matrix, cols):
    """
    Plots a figure with three subplots for Kendall correlation, NMI, and linear combination.
    """
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Kendall Correlation Heatmap
    sns.heatmap(
        kendall_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
        cbar=True,
        ax=axes[0],
    )
    axes[0].set_title("Kendall Correlation Heatmap")

    # Normalized Mutual Information Heatmap
    sns.heatmap(
        nmi_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
        cbar=True,
        ax=axes[1],
    )
    axes[1].set_title("NMI Heatmap")

    # Combined Heatmap (Average of Kendall and NMI)
    # Convert the correlation matrix dictionary into a pandas DataFrame
    data = [[correlation_matrix.get((c1, c2), 0) for c2 in cols] for c1 in cols]
    corr_df = pd.DataFrame(data, index=cols, columns=cols)
    sns.heatmap(
        corr_df,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
        cbar=True,
        ax=axes[2],
    )
    axes[2].set_title("Linear correlation Matrix")

    # Adjust layout
    plt.tight_layout()
    plt.show()


plot_combined_heatmaps(correlation_matrix, kendall_matrix, nmi_matrix, input_columns)


# COMMAND ----------

# MAGIC %md
# MAGIC ####PDP

# COMMAND ----------


def create_partial_dependence_plot(pipeline_model, train_df, cols, feature):
    """
    Generate a Partial Dependence Plot (PDP) for a feature, along with the exposure and average of "is_anomaly".

    :param pipeline_model: Trained PySpark PipelineModel
    :param train_df: PySpark DataFrame used for training
    :param cols: List of input feature column names
    :param feature: Feature for which PDP is generated
    """
    # Step 1: Extract the PySpark RandomForestClassifier from the pipeline
    rf_model = pipeline_model.stages[-1]  # Assuming RF is the last stage

    # Step 2: Convert PySpark DataFrame to pandas
    pandas_df = train_df.select(cols + ["is_anomaly"]).toPandas()
    features = pandas_df[cols]
    labels = pandas_df["is_anomaly"]

    # Step 3: Fix the seed value to fit within scikit-learn's range
    seed = rf_model.getOrDefault("seed") % (
        2**32
    )  # Ensure the seed is in the valid range
    n_estimators = rf_model.getNumTrees
    # Step 4: Recreate the RandomForestClassifier in scikit-learn
    rf_classifier = RandomForestClassifier_sk(
        n_estimators=n_estimators,
        max_depth=rf_model.getOrDefault("maxDepth"),
        random_state=seed,
    )

    # Train scikit-learn RandomForestClassifier on the same data
    rf_classifier.fit(features, labels)

    # Step 5: Generate PDP using `PartialDependenceDisplay.from_estimator`
    fig, ax1 = plt.subplots(figsize=(10, 6))
    display = PartialDependenceDisplay.from_estimator(
        rf_classifier,
        features,
        features=[feature],  # Feature name here, not index
        ax=ax1,
    )

    # Extract PDP values from the display
    pd_x = display.pd_results[0]["values"]
    pd_y = display.pd_results[0]["average"].tolist()

    # Ensure `pd_x` and `pd_y` dimensions match
    if len(pd_x) != len(pd_y):
        pd_x = np.linspace(min(pd_x), max(pd_x), len(pd_y))

    # Step 6: Compute exposure and average anomaly for feature values
    exposure = pandas_df.groupby(feature)["is_anomaly"].count()
    avg_anomaly = pandas_df.groupby(feature)["is_anomaly"].mean()

    # Plot PDP on the left y-axis
    ax1.plot(pd_x, pd_y, label="PDP", color="blue", lw=2)
    ax1.set_xlabel(f"{feature}")
    ax1.set_ylabel("Partial Dependence", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot exposure and average anomaly on the right y-axis
    ax2 = ax1.twinx()
    # ax2.bar(exposure.index, exposure.values, alpha=0.4, label="Exposure", color="gray")
    ax2.scatter(
        avg_anomaly.index,
        avg_anomaly.values,
        label="Avg. Is_Anomaly",
        color="red",
        marker="o",
    )
    ax2.set_ylabel("Exposure / Avg. Is_Anomaly", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Add title and show plot
    plt.title(f"Partial Dependence and Exposure for Feature: {feature}")
    plt.show()


# Loop over input columns to generate Partial Dependence Plots for each feature
for feature in input_columns:
    print(f"Generating Partial Dependence Plot for feature: {feature}")
    create_partial_dependence_plot(
        pipeline_model=model, train_df=train_df, cols=input_columns, feature=feature
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluating performances on test set

# COMMAND ----------

# Apply the trained model to test data
test_predictions = model.predict(context = None, model_df=cmd_df)

# Ensure no nulls in required columns
test_predictions = test_predictions.fillna(0, subset=[response, "prediction"])

# COMMAND ----------

# Calculate confusion matrix components
confusion_matrix_df = test_predictions.select(
    when(col("is_error") == 1, 1).otherwise(0).alias("actual_positive"),
    when(col("is_error") == 0, 1).otherwise(0).alias("actual_negative"),
    when(col("prediction") == 1, 1).otherwise(0).alias("predicted_positive"),
    when(col("prediction") == 0, 1).otherwise(0).alias("predicted_negative"),
)

# Aggregate counts for TP, TN, FP, FN
metrics = confusion_matrix_df.agg(
    spark_sum(col("actual_positive") * col("predicted_positive")).alias("TP"),
    spark_sum(col("actual_negative") * col("predicted_negative")).alias("TN"),
    spark_sum(col("actual_negative") * col("predicted_positive")).alias("FP"),
    spark_sum(col("actual_positive") * col("predicted_negative")).alias("FN"),
).collect()[0]

TP, TN, FP, FN = metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]

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

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Negative", "Predicted Positive"],
    yticklabels=["Actual Negative", "Actual Positive"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ####Adjust tag_anomalies

# COMMAND ----------

result = test_predictions.groupBy("tag_anomalies", "is_error","train_or_test","prediction").count()
display(result)

# COMMAND ----------

test = test_predictions.toPandas()
test = test[test['city__cmd'] == 'SAN DORLIGO DELLA VALLE - DOLINA']
# Create a colormap
import folium


# Initialize the map centered around the approximate center of the points
m = folium.Map(location=[20, 0], zoom_start=2)


# Function to convert S_Score to radius
def get_classification_color(row):
    if row['is_error'] == 1 and row['prediction'] == 1:
        return 'green'  # TP
    elif row['is_error'] == 0 and row['prediction'] == 1:
        return 'orange'  # FP
    elif row['is_error'] == 0 and row['prediction'] == 0:
        return 'blue'  # TN
    elif row['is_error'] == 1 and row['prediction'] == 0:
        return 'red'  # FN


# Add the points to the map
for idx, row in test.iterrows():
    color = get_classification_color(row)

 # Add the marker with the city name as a label
    border_color = 'red' if (row['train_or_test'])=="train" else 'black'
    # Create a popup with information about the point
    popup_text = f"ID: {row['id__cmd']} <br> City CMD:{row['city__cmd']}<br>latitude: {row['lat__cmd']}<br>longitude: {row['long__cmd']}"
    popup = folium.Popup(popup_text, max_width=300)
    # Add the circle marker with the popup
    folium.CircleMarker(
        location=[row["lat__cmd"], row["long__cmd"]],
        radius=5,
        color=border_color,  # Border color
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        weight=1,
        popup = popup
    ).add_to(m)

# Add colormap to map
m

# Display the map in a Jupyter Notebook (if running in a notebook)
m


# COMMAND ----------

test_predictions = test_predictions.withColumn(
    "tag_anomalies",
    when((col("prediction") == 0) & (col("tag_anomalies").isNotNull()), None).otherwise(
        col("tag_anomalies")
    ),
).drop("features", "rawPrediction", "probability")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Save results

# COMMAND ----------

final_df = test_predictions.filter(sf.col("tag_anomalies").isNotNull())
logger.warning(
    f"""Saving the results
    """
)

(
    final_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE_NAME)
)
