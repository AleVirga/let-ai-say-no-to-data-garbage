
import mlflow
import json
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import pandas as pd
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from pyspark.sql import SparkSession


class SimilarityBasedEncoder(mlflow.pyfunc.PythonModel):
    """
    A custom PySpark model designed to optimize the ordering of categorical values 
    within clusters by maximizing the entropy of their arrangement, based on 
    Levenshtein distance.

    Attributes:
        output_col (str): The name of the output column in the resulting DataFrame. Defaults to "index".
        cat_col (str): The name of the categorical column to process. Defaults to "standardized_city".
        partition_col (str): The name of the column used for partitioning the data into clusters. Defaults to "most_common_cluster".
    """

    def __init__(self, output_col=None, cat_col=None, partition_col=None):
        """
        Initializes the encoder with default or custom column names.

        Args:
            output_col (str, optional): The name of the output column. Defaults to "index".
            cat_col (str, optional): The name of the categorical column to process. Defaults to "standardized_city".
            partition_col (str, optional): The name of the partition column. Defaults to "most_common_cluster".
        """
        default_cat_col = "standardized_city"
        default_partition_col = "most_common_cluster"
        self.output_col = output_col if output_col else "index"
        self.partition_col = partition_col if partition_col else default_partition_col
        self.cat_col = cat_col if cat_col else default_cat_col

    # UDF to compute Levenshtein distance between two strings
    @staticmethod
    @sf.udf(st.IntegerType())
    def _levenshtein_udf(str1, str2):
        """
        User-defined function (UDF) to compute the Levenshtein distance between two strings.

        Args:
            str1 (str): The first string.
            str2 (str): The second string.

        Returns:
            int: The Levenshtein distance between the two strings.
        """
        import Levenshtein
        return Levenshtein.distance(str1, str2)
    @staticmethod
    def _greedy_maximize_entropy(word_list, distances):
        """
        Rearrange the words in word_list such that similar words are placed as far apart as possible.

        Args:
            word_list (list): A list of unique words.
            distances (dict): A dictionary with keys as (word1, word2) tuples and values as the distance between them.

        Returns:
            list: A list of words arranged to maximize the distance between similar words.
        """
        # Initialize the list of words that will store the final order
        optimized_order = []
        
        # Start with the first word from the word list
        current_word = word_list[0]
        optimized_order.append(current_word)
        
        # Keep track of used words
        used_words = {current_word}
        
        # Greedily pick the next word to maximize the distance from the last selected word
        while len(optimized_order) < len(word_list):
            max_distance = -1
            next_word = None
            
            for word in word_list:
                if word not in used_words:
                    # Calculate the distance from the current word to this candidate word
                    distance = distances.get((current_word, word), distances.get((word, current_word), 0))
                    
                    # Update if this is the farthest word we've seen so far
                    if distance > max_distance:
                        max_distance = distance
                        next_word = word
            
            # Add the next word to the optimized order and mark it as used
            optimized_order.append(next_word)
            used_words.add(next_word)
            current_word = next_word  # Move to the next word
        
        return optimized_order
    
    def fit(self, df: DataFrame) -> DataFrame:
        """
        Processes the input PySpark DataFrame by clustering categorical values and 
        optimizing their order within each cluster.

        Args:
            df (DataFrame): Input PySpark DataFrame to process.

        Returns:
            DataFrame: A PySpark DataFrame containing optimized orders of the categorical column
                       with an index column.
        """
        # Create a DataFrame to hold the optimized orders across clusters
        spark = SparkSession.builder.appName("SimilarityEncoder").getOrCreate()
        optimized_orders_df = spark.createDataFrame([], schema=f"{self.output_col} INT, word STRING")

        # Assuming clustered_df contains a partitiona column
        clusters = df.select(self.partition_col).distinct().collect()

        # Process each batch independently
        for cluster_row in clusters:
            cluster = cluster_row[self.partition_col]

            # Filter cities for the current cluster
            cat_df2 = df.filter(sf.col(self.partition_col) == cluster).select(self.cat_col).dropDuplicates()

            # Generate a list of renamed columns for df2 with '_2' suffix
            renamed_df2_columns = [
                sf.col(f"df2.{col}").alias(f"{col}_2") for col in cat_df2.columns
            ]

            # Select all columns from df1 and the renamed columns from df2
            df_cross = cat_df2.alias("df1").crossJoin(cat_df2.alias("df2")).select(
                [sf.col(f"df1.{col}") for col in cat_df2.columns] + renamed_df2_columns
            )
            
            # Calculate the pairwise Levenshtein distance for each pair
            df_with_distance = df_cross.withColumn(
                "distance",
                self._levenshtein_udf(sf.col(f"df1.{self.cat_col}"), sf.col(f"{self.cat_col}_2"))
            ).filter(
                sf.col(f"df1.{self.cat_col}") != sf.col(f"{self.cat_col}_2")
            )
            
            # Example: Collecting distances into a matrix and applying a similar algorithm
            word_pairs = df_with_distance.select(
                self.cat_col, f"{self.cat_col}_2", "distance"
            ).collect()

            # Extract words and distances from the collected result
            word_list = [
                row[self.cat_col]
                for row in cat_df2.select(self.cat_col).distinct().collect()
            ]
            distances = {
                (row[self.cat_col], row[f"{self.cat_col}_2"]): row["distance"]
                for row in word_pairs
            }

            # Output the optimized order
            optimized_order = self._greedy_maximize_entropy(word_list, distances)
            # Create a list of tuples with index starting from the last index of the previous cluster
            start_index = optimized_orders_df.count() + 1  # Start index from the maximum of the previous cluster
            indexed_words = [(i + start_index, word) for i, word in enumerate(optimized_order)]

            # Create a PySpark DataFrame from the indexed list for the current cluster
            current_cluster_df = spark.createDataFrame(indexed_words, schema=[f"{self.output_col}", "word"])

            # Append current cluster results to the optimized orders DataFrame
            optimized_orders_df = optimized_orders_df.union(current_cluster_df)
        
        encoded_df = df.join(optimized_orders_df, df[self.cat_col] == optimized_orders_df.word, how = "left")
        return encoded_df