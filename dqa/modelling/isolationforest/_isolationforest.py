
import mlflow
import json
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import pandas as pd


class IsolationForestWithPreprocessing(mlflow.pyfunc.PythonModel):
    """
    A custom PySpark model that classifies anomalies based on input features.
    The model uses an Isolation Forest for anomaly detection and is designed to
    preprocess data before fitting and predicting.
    """

    def __init__(
        self, params=None, input_cols=None, partition_cols=None, output_col=None, geo_cols=None
    ):
        """
        Initialize the model with parameters, input columns, and partition columns.

        Args:
            params (dict): Parameters for the Isolation Forest model.
            input_cols (list): List of input column names to be used as features.
            partition_cols (list): List of column names for grouping during processing.
            output_col (str): Name of the output column to save anomaly scores.
        """
        # Default parameters for Isolation Forest
        default_params = {
            "n_estimators": 50,
            "max_samples": "auto",
            "contamination": "auto",
            "max_features": 1.0,
            "random_state": 0,
        }

        # Default input columns if not specified
        default_input_cols = ["lat__cmd", "long__cmd", "index"]
        default_geo_cols = ["lat__cmd", "long__cmd"]
        self.encoding_cols = "index"
        self.params = (
            params if params else default_params
        )  # Use provided or default parameters
        self.input_cols = (
            input_cols if input_cols else default_input_cols
        )  # Input feature columns
        self.geo_cols = (
            geo_cols if geo_cols else default_geo_cols
        )  # Input feature columns
        self.partition_cols = (
            partition_cols if partition_cols else []
        )  # Grouping columns
        self.output_col = (
            output_col if output_col else "anomaly_score"
        )  # Output column for scores
        self.config = None  # Placeholder for configuration settings (if needed)
        self.model_input_cols = []  # Std and scaled input cols for isolation forest

    def load_context(self, context=None, config_path=None):
        """
        Load the context or configuration for the model.

        This function is automatically called when the model is deployed or loaded
        using `mlflow.pyfunc.load_model()`. It can also be manually triggered in a notebook for testing.

        Args:
            context (mlflow.pyfunc.PythonModelContext): Context object containing model artifacts.
            config_path (str): Path to the configuration file (if provided manually).

        Sets:
            self.config (dict): Loaded configuration as a dictionary.
        """
        if context:  # Executed during deployment or model loading
            config_path = context.artifacts["config_path"]
        elif config_path:  # Executed during local testing with a provided path
            pass

        # Load the configuration file
        self.config = json.load(open(config_path))


    def _preprocess_model(self, model_df: DataFrame) -> DataFrame:
        """
        Preprocess the input DataFrame for the model.

        This includes:

        - Filling null values most_common_cluster to starting cluster.
        - Casting null values encoding cols to maximum value + 1.

        Args:
            model_df (DataFrame): Input PySpark DataFrame to preprocess.

        Returns:
            DataFrame: Preprocessed PySpark DataFrame.
        """
        from pyspark.sql import functions as sf

        max_index = model_df.agg(sf.max(self.encoding_cols)).first()[0]
        processed_input = model_df.withColumn(
            "most_common_cluster",
            sf.when(sf.col(self.encoding_cols).isNull(), sf.col("cluster")).otherwise(
                sf.col("most_common_cluster")
            ),
        )
        processed_input = processed_input.withColumn(
            self.encoding_cols,
            sf.when(sf.col(self.encoding_cols).isNull(), max_index + 1).otherwise(
                sf.col(self.encoding_cols)
            ),
        )

        return processed_input

    def isolation_forest(self, pdf: pd.DataFrame, *args) -> pd.DataFrame:
        """
        Apply Isolation Forest to a pandas DataFrame for anomaly detection.

        Args:
            pdf (pd.DataFrame): Input pandas DataFrame to process.
            *args: Additional arguments passed by applyInPandas (e.g., grouping keys)

        Returns:
            pd.DataFrame: DataFrame with anomaly scores added.
        """
        from sklearn.ensemble import IsolationForest

        # Filter input columns to remove constant columns
        model_input_cols = self.model_input_cols
        model_input_cols = [col for col in model_input_cols if pdf[col].nunique() > 1]
        data = pdf[model_input_cols]

        # Initialize and fit the Isolation Forest model
        model = IsolationForest()
        model.set_params(**self.params)  # Use the default or customized parameters
        model.fit(data)  # Fit the model on the selected data

        # Add the anomaly scores to the original DataFrame
        pdf[self.output_col] = model.score_samples(data)
        return pdf

    def fit(self, train_df: DataFrame) -> DataFrame:
        """
        Fit the Isolation Forest model using the preprocessed training data.

        This includes:
        - Preprocessing the training data.
        - Applying Isolation Forest in pandas grouped by partition columns.

        Args:
            train_df (DataFrame): Training PySpark DataFrame.

        Returns:
            DataFrame: PySpark DataFrame with anomaly scores added.
        """
        from pyspark.sql.types import StructType, StructField, DoubleType, StringType
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.ml.functions import vector_to_array

        # Running preprocessing
        preprocessed_train_df = self._preprocess_model(train_df)
        # Assembling the features
        assembler = VectorAssembler(
            inputCols=self.input_cols, outputCol="assembled_features"
        )
        preprocessed_train_df = assembler.transform(preprocessed_train_df)

        # Step 2: Apply StandardScaler to the 'features' column
        scaler = StandardScaler(
            inputCol="assembled_features",
            outputCol="scaled_features",
            withMean=True,
            withStd=True,
        )
        scaler_model = scaler.fit(preprocessed_train_df)
        preprocessed_train_df = scaler_model.transform(preprocessed_train_df)

        preprocessed_train_df = preprocessed_train_df.withColumn(
            "features_array", vector_to_array(sf.col("scaled_features"))
        )

        # Get the length of the array (vector)
        array_length = preprocessed_train_df.select(
            sf.size(sf.col("features_array"))
        ).first()[0]

        column_names = [f"st_attr_{i}" for i in range(array_length)]
        preprocessed_train_df = preprocessed_train_df.select(
            "*",
            *[
                sf.col("features_array")[i].alias(column_name)
                for i, column_name in enumerate(column_names)
            ],
        ).drop(
            "assembled_features", "scaled_features", "features_array", "features_new"
        )
        self.model_input_cols = column_names

        # Define the schema for the output, adding the anomaly score column
        schema_output = StructType(
            preprocessed_train_df.schema.fields
            + [StructField(self.output_col, DoubleType(), True)]
        )
        # Apply Isolation Forest using `applyInPandas` for grouped processing
        if self.partition_cols == []:
            output_df = preprocessed_train_df.groupby().applyInPandas(
                lambda pdf: self.isolation_forest(pdf),
                schema=schema_output,
            )
        else:
            output_df = preprocessed_train_df.groupby(self.partition_cols).applyInPandas(
                lambda pdf: self.isolation_forest(pdf),
                schema=schema_output,
            )

        output_df = output_df.drop(*self.model_input_cols)

        return output_df
