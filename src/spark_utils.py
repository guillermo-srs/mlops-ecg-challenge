# File: src/spark_utils.py

import os
import time # time was already imported, good.
from typing import Callable, Iterator, Any, List, Tuple

from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.rdd import RDD


def init_spark_session(
    app_name: str,
    master_url: str,
    driver_memory_str: str,
    log_level: str = "WARN", # New parameter for log level
    arrow_batch_size: int = 10000,
    spark_local_dir: str = './tmp/',
    max_partition_bytes: str = "300m"
) -> SparkSession:
    """
    Initializes and returns a SparkSession.
    Sets the PySpark log level.
    """
    # Spark event log configuration (local)
    # This is for Spark's own event logging, not application logs via log4j directly
    spark_log_path_local = "./spark-events"
    absolute_event_log_dir_path = os.path.abspath(spark_log_path_local)
    # No print needed for directory creation, it's a setup detail
    os.makedirs(absolute_event_log_dir_path, exist_ok=True)

    event_log_uri = f"file://{absolute_event_log_dir_path}"

    builder = SparkSession.builder \
        .appName(app_name) \
        .master(master_url) \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", event_log_uri) \
        .config("spark.history.fs.logDirectory", event_log_uri) \
        .config("spark.ui.showConsoleProgress", "true") \
        .config("spark.driver.memory", driver_memory_str) \
        .config("spark.local.dir", spark_local_dir) \
        .config("spark.sql.files.maxPartitionBytes", max_partition_bytes) \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_batch_size))

    # This print is a high-level phase, so it's okay.
    print(f"Initializing Spark session: {app_name} with master {master_url}, driver memory {driver_memory_str}.")
    spark = builder.getOrCreate()

    # Set the PySpark log level
    # Valid levels include: "ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"
    if spark.sparkContext:
        spark.sparkContext.setLogLevel(log_level)
        print(f"PySpark log level set to: {log_level}")
    else:
        # This case should be rare with getOrCreate()
        print("Warning: SparkContext not available, could not set log level.")

    return spark


def repartition_dataframe(
    input_df: DataFrame,
    num_target_partitions: int
) -> DataFrame:
    """
    Repartitions the DataFrame if the current number of partitions
    does not match the target number and num_target_partitions > 0.
    """
    current_partitions = input_df.rdd.getNumPartitions()
    if current_partitions != num_target_partitions and num_target_partitions > 0:
        # This print is a significant operation, so it's okay.
        print(f"Repartitioning DataFrame from {current_partitions} to {num_target_partitions} partitions.")
        return input_df.repartition(num_target_partitions)
    # No print if not repartitioning, to reduce noise.
    return input_df


def get_top_n_from_rdd(
    scores_rdd: RDD[Tuple[Any, float]],
    top_n_count: int
) -> List[Tuple[Any, float]]:
    """
    Gets the top N results from an RDD of (identifier, score) tuples.
    Results are returned sorted by score descending.
    """
    # This is a utility function, no print needed here.
    return scores_rdd.top(top_n_count, key=lambda x: x[1])
