
import mlflow
from pyspark.sql import DataFrame
import json
import os
from pyspark.sql.functions import col

class SupervisedAnomalyClassifier(mlflow.pyfunc.PythonModel):
    """
    A custom PySpark model that classifies anomalies based on input features.
    The model uses a RandomForestClassifier within a PySpark ML pipeline and
    is designed to preprocess data before fitting and predicting.
    """

    def __init__(self, params=None, input_columns=None, response=None, class_thresholds=[0.5, 0.5]):
        """
        Initialize the model with input columns and the target response column.

        Args:
            input_columns (list): List of input column names to be used as features.
            response (str): Name of the target column (label).
        """
        default_params = {
            "numTrees": 20,
            "maxDepth": 5,
            "impurity": "gini",
            "featureSubsetStrategy": "auto",
            "minInstancesPerNode": 1,
            "minInfoGain": 0.0,
            "seed": None,
        }
        default_input_cols =  [
            "lat__cmd",
            "long__cmd",
            "S_Score_coordinates",
            "S_Score_city_cluster",
            "neighbors_count",
        ]
        default_response = "is_anomaly"
        self.classifier_model = None  # Placeholder for the trained model pipeline
        self.params = params if params else default_params #Parameters for the model
        self.config = None  # Placeholder for configuration settings (if needed)
        self.input_columns = input_columns if input_columns else default_input_cols # Feature column names
        self.response = response if response else default_response  # Target column name
        self.class_thresholds = class_thresholds

    def load_context(self, context=None, config_path=None):
        """
        Load the context or configuration for the model.

        This function is automatically called when the model is deployed or loaded 
        using `mlflow.pyfunc.load_model()`. It can also be manually triggered 
        in a notebook for testing.

        Args:
            context (mlflow.pyfunc.PythonModelContext): Context object containing 
                model artifacts (e.g., config_path).
            config_path (str): Path to the configuration file (if provided manually).

        Sets:
            self.config (dict): Loaded configuration as a dictionary.
        """
        if context:  # Executed during deployment or model loading
            config_path = context.artifacts["config_path"]
        else:  # Executed during local testing
            pass

        # Load the configuration file
        self.config = json.load(open(config_path))

    def preprocess_input(self, model_df: DataFrame) -> DataFrame:
        """
        Preprocess the input DataFrame for the model.

        This includes:
        - Casting all input columns to double type.
        - Filling null values in the input columns with 0.

        Args:
            model_df (DataFrame): Input PySpark DataFrame to preprocess.

        Returns:
            DataFrame: Preprocessed PySpark DataFrame.
        """
        processed_input = model_df
        for column in self.input_columns:
            # Cast each input column to double type
            processed_input = processed_input.withColumn(column, col(column).cast("double"))
            # Fill null values with 0
            processed_input = processed_input.fillna({column: 0})

        return processed_input

    def fit(self, train_df: DataFrame):
        """
        Fit the RandomForestClassifier model using the preprocessed training data.

        This includes:
        - Preprocessing the training data.
        - Creating a pipeline with a VectorAssembler and RandomForestClassifier.
        - Fitting the pipeline.

        Args:
            train_df (DataFrame): Training PySpark DataFrame.
        """
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml import Pipeline

        # Preprocess the training DataFrame
        processed_model_input = self.preprocess_input(train_df)

        # Assemble feature columns into a single "features" column
        assembler = VectorAssembler(inputCols=self.input_columns, outputCol="features")

        # Initialize the RandomForestClassifier
        classifier = RandomForestClassifier(
            featuresCol="features",  # Column containing the feature vector
            labelCol=self.response,  # Target column
            predictionCol="prediction",  # Column to store predictions
            **self.params  # Use the default or customized parameters
        )
        classifier = classifier.setThresholds(self.class_thresholds)
        # Create a pipeline with the assembler and classifier
        classifier_model = Pipeline(stages=[assembler, classifier])

        # Fit the pipeline to the preprocessed training data
        self.classifier_model = classifier_model.fit(processed_model_input)

    def predict(self, context, model_df: DataFrame) -> DataFrame:
        """
        Make predictions using the trained model.

        This includes:
        - Preprocessing the input data.
        - Transforming the data using the trained pipeline to get predictions.

        Args:
            context: Context object (not used in this function).
            model_df (DataFrame): Input PySpark DataFrame for prediction.

        Returns:
            DataFrame: PySpark DataFrame with predictions.
        """
        # Preprocess the input DataFrame
        processed_model_input = self.preprocess_input(model_df)

        # Use the trained pipeline to transform the data and generate predictions
        return self.classifier_model.transform(processed_model_input)
