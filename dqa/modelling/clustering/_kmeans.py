
import mlflow
import json
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import pandas as pd
from dqa.logging import make_logger

logger = make_logger(__name__)


class KMeansPostProcessed(mlflow.pyfunc.PythonModel):
    """
    A custom PySpark model designed to perform post-processing on KMeans clustering results.
    This class provides functionality for clustering, refining clusters, and merging small clusters
    based on geospatial and categorical data.

    Attributes:
        kmeans_params (dict): Parameters for configuring the KMeans clustering algorithm.
        geo_cols (list): List of column names representing geospatial features (e.g., latitude, longitude).
        cat_col (str): Column name for categorical data used to group records.
        output_col (str): Name of the column to store the resulting cluster assignments.

    Methods:
        KMeans(df): Performs KMeans clustering on the input DataFrame, determining the optimal number
            of clusters (k) using the elbow method and assigning cluster labels.
        _unique_cluster_cat(df): Identifies the most common cluster for each categorical group and
            refines cluster assignments accordingly.
        _merge_small_clusters(df): Merges small clusters (below a size threshold) with the nearest
            larger cluster based on geospatial proximity.
        fit(df): applies the prevous functions and returns a Datataframe with final clusters.
    """

    def __init__(
        self, kmeans_params=None, output_col=None, geo_cols=None, cat_col=None
    ):
        """
        Initializes the KMeansPostProcessed class with optional parameters for clustering and data input.

        Args:
            kmeans_params (dict): Parameters for KMeans clustering. Default parameters are used if None is provided.
            geo_cols (list): List of geospatial columns (e.g., latitude, longitude). Defaults to ["lat__cmd", "long__cmd"].
            output_col (str): Name of the output column to store cluster labels. Defaults to "cluster".
            cat_col (str): Name of the categorical column used for grouping. Defaults to "standardized_city".
        """
        # Default parameters for KMeans
        default_kMeans_params = {
            "size_threshold": 100,
            "max_n_clusters": 11,
            "seed": 1,
            "default_cluster": 99,
        }

        default_geo_cols = ["lat__cmd", "long__cmd"]
        default_cat_col = "standardized_city"
        self.output_col = output_col if output_col else "cluster"
        self.kmeans_params = kmeans_params if kmeans_params else default_kMeans_params
        self.geo_cols = geo_cols if geo_cols else default_geo_cols
        self.cat_col = cat_col if cat_col else default_cat_col

    def KMeans(self, df: DataFrame) -> DataFrame:
        """
        Performs KMeans clustering on the input DataFrame, determining the optimal number of clusters using the elbow method.

        Args:
            df (DataFrame): PySpark DataFrame containing the data to be clustered.

        Returns:
            DataFrame: PySpark DataFrame with cluster labels assigned in the specified output column.
        """
        from pyspark.ml.clustering import KMeans
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql import functions as sf
        import matplotlib.pyplot as plt

        # Step 1: Remove rows with NULL values in geospatial columns
        input_df = df.dropna(subset=self.geo_cols)

        # Step 2: Assemble feature vector for clustering
        assembler = VectorAssembler(
            inputCols=self.geo_cols, outputCol="assembled_features"
        )
        input_df = assembler.transform(input_df)

        # Step 3: Use the elbow method to determine the optimal number of clusters (k)
        wcss = []  # List to store Within-Cluster Sum of Squared Errors for each k
        k_values = range(2, 11)  # Testing k values from 2 to 10
        for k in k_values:
            kmeans = KMeans(
                k=k,
                seed=1,
                featuresCol="assembled_features",
                predictionCol="prediction",
            )
            model = kmeans.fit(input_df)
            wcss.append(model.summary.trainingCost)

        # Identify the optimal k using the elbow point
        from kneed import KneeLocator

        kneedle = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
        optimal_k = kneedle.elbow

        plt.figure(figsize=(8, 5))
        plt.plot(k_values, wcss, marker="o", linestyle="-", label="WCSS")
        plt.axvline(
            optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}"
        )
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
        plt.title("Elbow Method with KneeLocator")
        plt.legend()
        plt.grid(True)
        plt.show()

        logger.info(f"Optimal number of clusters (k): {optimal_k}")

        # Step 4: Apply KMeans with the optimal k
        kmeans = KMeans(
            k=optimal_k,
            seed=1,
            featuresCol="assembled_features",
            predictionCol=self.output_col,
        )
        model = kmeans.fit(input_df)
        clustered_df = model.transform(input_df).drop("assembled_features")

        # Step 5: Handle rows with NULL values by assigning them to a default cluster (e.g., 99)
        null_condition = sf.lit(False)
        for column in self.geo_cols:
            null_condition |= sf.col(column).isNull()
        default_cluster_df = df.filter(null_condition).withColumn(
            self.output_col, sf.lit(99)
        )

        # Combine clustered and default cluster DataFrames
        final_df = default_cluster_df.unionByName(
            clustered_df, allowMissingColumns=True
        )
        return final_df

    def _unique_cluster_cat(self, df: DataFrame) -> DataFrame:
        """
        Refines cluster assignments by identifying the most common cluster for each categorical group.

        Args:
            df (DataFrame): PySpark DataFrame with initial cluster assignments.

        Returns:
            DataFrame: PySpark DataFrame with refined cluster assignments. A new column named `most_common_<output_col>` is added,
                       which stores the most common cluster for each categorical group.
        """
        from pyspark.sql.window import Window
        import pyspark.sql.functions as sf

        logger.info(f"Refining clusters according to catgorical attribute")
        # Step 1: Count the number of records in each (category, cluster) pair
        cat_cluster_counts = df.groupBy(self.cat_col, self.output_col).agg(
            sf.count("*").alias("count")
        )

        # Step 2: Rank clusters within each category by count
        window = Window.partitionBy(self.cat_col).orderBy(sf.desc("count"))
        ranked_clusters = cat_cluster_counts.withColumn(
            "rank", sf.row_number().over(window)
        )

        # Step 3: Keep only the most common cluster for each category
        most_common_clusters = ranked_clusters.filter(sf.col("rank") == 1).select(
            self.cat_col, self.output_col
        )
        most_common_clusters = most_common_clusters.withColumnRenamed(
            self.output_col, f"most_common_{self.output_col}"
        )

        # Step 4: Assign the most common cluster back to the original DataFrame
        df = df.join(most_common_clusters, on=self.cat_col, how="left").withColumn(
            f"most_common_{self.output_col}",
            sf.when(
                sf.col(f"most_common_{self.output_col}").isNull(),
                sf.col(self.output_col),
            ).otherwise(sf.col(f"most_common_{self.output_col}")),
        )
        return df

    def _merge_small_clusters(self, df: DataFrame) -> DataFrame:
        """
        Merges small clusters (below a size threshold) with the nearest large cluster based on geospatial proximity.

        Args:
            df (DataFrame): PySpark DataFrame with cluster assignments.

        Returns:
            DataFrame: PySpark DataFrame with updated cluster assignments after merging small clusters. The final column
                       remains as `most_common_<output_col>`, with small clusters merged into larger ones.
        """
        from pyspark.sql import functions as sf
        from pyspark.sql.types import DoubleType
        from pyspark.sql.window import Window
        from dqa.scoring import DistanceMatching

        logger.info(f"Merging small clusters")
        latitude = self.geo_cols[0]
        longitude = self.geo_cols[1]
        size_threshold = 100  # Minimum size for clusters to be considered "large"
        distance_udf = sf.udf(DistanceMatching().distance_match, DoubleType())

        while True:
            # Step 1: Compute cluster centroids and sizes
            centroids = df.groupBy(f"most_common_{self.output_col}").agg(
                sf.avg(latitude).alias("centroid_lat"),
                sf.avg(longitude).alias("centroid_long"),
                sf.count("*").alias("cluster_size"),
            )

            # Step 2: Identify small and large clusters
            small_clusters = centroids.filter(sf.col("cluster_size") < size_threshold)
            large_clusters = centroids.filter(sf.col("cluster_size") >= size_threshold)

            if small_clusters.count() == 0:
                break  # Exit loop if no small clusters remain

            # Step 3: Calculate distances between small and large clusters
            small_clusters = small_clusters.select(
                sf.col(f"most_common_{self.output_col}").alias("small_cluster_id"),
                sf.col("centroid_lat").alias("small_centroid_lat"),
                sf.col("centroid_long").alias("small_centroid_long"),
            )

            large_clusters = large_clusters.select(
                sf.col(f"most_common_{self.output_col}").alias("large_cluster_id"),
                sf.col("centroid_lat").alias("large_centroid_lat"),
                sf.col("centroid_long").alias("large_centroid_long"),
            )

            small_large_distances = small_clusters.crossJoin(large_clusters).withColumn(
                "distance",
                distance_udf(
                    sf.col("small_centroid_lat"),
                    sf.col("small_centroid_long"),
                    sf.col("large_centroid_lat"),
                    sf.col("large_centroid_long"),
                ),
            )

            # Step 4: Map each small cluster to its nearest large cluster
            window_spec = Window.partitionBy("small_cluster_id").orderBy("distance")
            small_to_large_mapping = (
                small_large_distances.withColumn(
                    "nearest_large_cluster",
                    sf.first("large_cluster_id").over(window_spec),
                )
                .select("small_cluster_id", "nearest_large_cluster")
                .distinct()
            )

            # Step 5: Update cluster assignments in the DataFrame
            df = df.withColumnRenamed(
                f"most_common_{self.output_col}", "original_cluster_id"
            )
            df = (
                df.join(
                    small_to_large_mapping,
                    on=(
                        df["original_cluster_id"]
                        == small_to_large_mapping["small_cluster_id"]
                    ),
                    how="left",
                )
                .withColumn(
                    f"most_common_{self.output_col}",
                    sf.coalesce(
                        sf.col("nearest_large_cluster"), sf.col("original_cluster_id")
                    ),
                )
                .drop(
                    "small_cluster_id", "nearest_large_cluster", "original_cluster_id"
                )
            )

        return df

    def fit(self, df: DataFrame) -> DataFrame:
        """
        Executes the full workflow: clustering, refining, and merging small clusters.

        Args:
            df (DataFrame): Input PySpark DataFrame to process.

        Returns:
            DataFrame: Final PySpark DataFrame with refined and merged cluster assignments.
        """
        df_output = self.KMeans(df)
        df_output = self._unique_cluster_cat(df_output)
        df_output = self._merge_small_clusters(df_output)
        return df_output
