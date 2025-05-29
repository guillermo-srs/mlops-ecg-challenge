# File: src/data_handling.py

import os
import time
import numpy as np
from typing import List, Any, Tuple, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col as spark_col, rand as spark_rand


def _generate_and_save_data(
    spark: SparkSession,
    path: str,
    num_rows: int,
    num_cols: int,
    id_column_name: str,
    target_num_partitions: int,
) -> DataFrame:
    """
    Helper function to generate, save, and return a DataFrame.
    """
    feature_col_expressions = [(spark_rand(seed=i).alias(f"col_{i}")) for i in range(num_cols)]

    if num_cols > 0:
        df = spark.range(0, num_rows, 1, target_num_partitions) \
                  .withColumnRenamed("id", id_column_name) \
                  .select(spark_col(id_column_name), *feature_col_expressions)
    else:
        df = spark.range(0, num_rows, 1, target_num_partitions) \
                  .withColumnRenamed("id", id_column_name)

    df.write.mode("overwrite").parquet(path)
    return df


def get_or_create_parquet_dataframe(
        spark: SparkSession,
        path: str,
        num_rows: int,
        num_cols: int,
        id_column_name: str,
        target_num_partitions: int,
        force_regenerate: bool = False
) -> DataFrame:
    """
    Loads a DataFrame from Parquet if it exists.
    Otherwise, generates, saves, and returns it.
    """
    operation_start_time = time.time()
    operation_description = ""

    if not force_regenerate and os.path.exists(path):
        print(f"Attempting to load data from: {path}")
        df = spark.read.parquet(path)
        # Basic check for id column; more robust checks could be added if necessary
        if id_column_name not in df.columns or (num_cols > 0 and f"col_{num_cols-1}" not in df.columns):
            print(f"Schema mismatch or missing columns in existing data at {path}. Regenerating.")
            df = _generate_and_save_data(spark, path, num_rows, num_cols,
                                         id_column_name, target_num_partitions)
            operation_description = f"Data regenerated and saved to '{path}'"
        else:
            operation_description = f"Data loaded from '{path}'"
    else:
        if force_regenerate:
            print(f"Forcing data regeneration for: {path}")
        else:
            print(f"Data not found at '{path}'. Generating new data.")
        df = _generate_and_save_data(spark, path, num_rows, num_cols, id_column_name,
                                     target_num_partitions)
        operation_description = f"Data generated and saved to '{path}'"

    operation_time = time.time() - operation_start_time
    print(f"{operation_description} in {operation_time:.2f} seconds. DataFrame has {df.rdd.getNumPartitions()} partitions.")
    return df


def create_query_vector(num_features: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Creates a 1D NumPy query vector.
    If a seed is provided, the vector will be deterministic.
    """
    if num_features <= 0:
        return np.array([], dtype=float)
    
    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.rand(num_features).astype(float)
    else:
        # Default to a common pattern if no seed, e.g., vector of 0.5
        # This was the previous behavior of create_reference_vector
        return np.array([0.5] * num_features, dtype=float)
