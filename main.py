# File: main.py

import argparse
import time
import sys
import os
import numpy as np
from functools import partial
import logging
import traceback
from typing import Iterator, Any, List, Tuple, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, DoubleType, DataType
from pyspark.sql.functions import col as spark_col

# --- Imports from src ---
from src.spark_utils import (
    init_spark_session,
    repartition_dataframe,
    get_top_n_from_rdd
)
from src.data_handling import (
    get_or_create_parquet_dataframe,
    create_query_vector
)
from src.processing_logic import (
    transform_to_similarity_space_udf_logic,
    search_in_space_udf_logic
)
from src.reference_selection import select_diverse_reference_vectors
from src.distributed_tree_orchestrator import (
    orchestrate_distributed_tree_build_and_persist,
    orchestrate_distributed_hierarchical_search
)
from src.reporting import print_results_summary
from src.similarity_metrics import BATCH_SIMILARITY_FUNCTIONS_MATRIX_TO_VECTOR

# --- Basic Logger Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
    stream=sys.stdout 
)
logger = logging.getLogger(__name__) # Logger para este módulo (main.py)
# Para ver logs DEBUG de otros módulos (si los configuran para usar logging):
# logging.getLogger('src.distributed_tree_orchestrator').setLevel(logging.DEBUG)


# --- Argument Parsing ---
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ECG Similarity Search Orchestrator.")
    # ... (argumentos igual que antes, incluyendo --python-log-level) ...
    # General Spark & Data Parameters
    parser.add_argument("--app-name", type=str, default="ECGSimilaritySearch", help="Spark application name.")
    parser.add_argument("--spark-cores", type=str, default="4", help="Number of Spark cores to use (e.g., '4' or '*').")
    parser.add_argument("--driver-memory", type=str, default="4g", help="Spark driver memory.")
    parser.add_argument("--spark-log-level", type=str, default="WARN",
                        choices=["ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"],
                        help="Set PySpark's root logger level (for Spark's own Java/Scala logs via Py4J).")
    parser.add_argument("--python-log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set Python's logging level for this script's messages.") 
    parser.add_argument("--num-data-rows", type=int, default=10000, help="Number of rows for synthetic data generation.")
    parser.add_argument("--num-data-cols", type=int, default=100, help="Number of feature columns (K) for synthetic data.")
    parser.add_argument("--id-column-name", type=str, default="ecg_id", help="Name of the ID column.")
    parser.add_argument("--data-path-prefix", type=str, default="./generated_data", help="Prefix for data storage paths.")
    parser.add_argument("--regenerate-data", action="store_true", default=False, help="Force regeneration of synthetic data.")
    parser.add_argument("--num-partitions", type=int, default=0,
                        help="Target number of Spark partitions for processing. 0 for default.")
    parser.add_argument("--query-vector-seed", type=int, default=None, help="Seed for generating a deterministic query vector. None for default (0.5 vector).")

    # Search Mode & Common Search Parameters
    parser.add_argument("--search-mode", type=str, required=True,
                        choices=["absolute", "optimized_transform", "hierarchical_kmeans"],
                        help="The similarity search mode to execute.")
    parser.add_argument("--top-x-results", type=int, default=5, help="Number of top similar results to find.")
    parser.add_argument("--similarity-metric", type=str, default="cosine",
                        choices=list(BATCH_SIMILARITY_FUNCTIONS_MATRIX_TO_VECTOR.keys()),
                        help="Similarity metric to use for comparisons.")

    # Parameters for "optimized_transform"
    parser.add_argument("--num-references-N", type=int, default=20,
                        help="N: Number of reference vectors for dimensionality reduction in optimized modes.")
    parser.add_argument("--ref-selection-sample-size", type=int, default=10000,
                        help="Number of rows to sample for selecting N reference vectors.")
    parser.add_argument("--use-median-for-center", action="store_true", default=False,
                        help="Use median instead of mean for center vector in reference selection.")
    parser.add_argument("--save-transformed-data-path", type=str, default="",
                        help="Optional path to save N-dim transformed DataFrame (for 'optimized_transform' mode).")

    # Parameters for "hierarchical_kmeans"
    parser.add_argument("--tree-filename", type=str, default="hkmeans_trees.pkl",
                        help="Filename for the pickled K-Means tree collection.")
    parser.add_argument("--force-rebuild-tree", action="store_true", default=False,
                        help="Force rebuilding of K-Means trees even if a saved file exists.")
    parser.add_argument("--k-per-level", type=int, default=5, help="K-Means: Number of clusters per tree level.")
    parser.add_argument("--max-depth", type=int, default=3, help="K-Means: Maximum depth of the trees.")
    parser.add_argument("--min-leaf-size", type=int, default=100, help="K-Means: Minimum samples per leaf node.")
    parser.add_argument("--kmeans-seed", type=int, default=42, help="K-Means: Random seed for clustering.")
    
    args = parser.parse_args()


    logging.getLogger().setLevel(getattr(logging, args.python_log_level.upper()))

    if args.num_partitions <= 0:
        try:
            cores = args.spark_cores.strip()
            if cores == '*' or cores.lower() == 'auto':
                num_physical_cores = os.cpu_count()
                args.num_partitions = num_physical_cores if num_physical_cores else 2
            elif 'local[' in cores and cores.endswith(']'):
                num_local_cores_str = cores.split('[')[-1][:-1]
                args.num_partitions = int(num_local_cores_str) if num_local_cores_str != '*' else (os.cpu_count() or 2)
            else: 
                args.num_partitions = int(cores)
        except (ValueError, AttributeError): 
            args.num_partitions = os.cpu_count() or 2 
        args.num_partitions = max(1, args.num_partitions)

    return args


# --- Workflow: Absolute Search ---
def run_absolute_search_workflow(
    spark: SparkSession,
    base_df: DataFrame,
    query_vector_k_dim: np.ndarray,
    args: argparse.Namespace, 
    feature_column_names: List[str]
):
    logger.info(f"--- Starting Absolute Search Workflow (Metric: {args.similarity_metric}, Top {args.top_x_results}) ---")
    search_start_time = time.time()

    logger.debug(f"Base DataFrame for absolute search has {base_df.rdd.getNumPartitions()} partitions.")
    if logger.isEnabledFor(logging.DEBUG):
        base_df.explain(extended=True) # Log physical plan before mapInPandas

    search_udf_callable = partial(
        search_in_space_udf_logic,
        query_vector=query_vector_k_dim,
        feature_column_names=feature_column_names,
        id_column_name=args.id_column_name,
        top_x_per_partition=args.top_x_results,
        similarity_metric_name=args.similarity_metric
    )

    id_col_type = base_df.schema[args.id_column_name].dataType
    search_output_schema = StructType([
        StructField(args.id_column_name, id_col_type, True),
        StructField("similarity_score", DoubleType(), True)
    ])
    
    logger.info("Applying search UDF (mapInPandas) for absolute search...")
    results_df_from_udf = base_df.mapInPandas(search_udf_callable, schema=search_output_schema)
    logger.info("Search UDF application complete. Converting to RDD for top N.")

    similarity_rdd = results_df_from_udf.select(
        spark_col(args.id_column_name),
        spark_col("similarity_score")
    ).rdd.map(lambda r: (r[args.id_column_name], r.similarity_score if r.similarity_score is not None else -float('inf')))

    logger.info("Collecting top N results from RDD...")
    top_n_results_list = get_top_n_from_rdd(similarity_rdd, args.top_x_results)
    logger.info("Top N results collected.")

    search_duration = time.time() - search_start_time
    logger.info(f"Absolute search (metric: {args.similarity_metric}) and Top N selection took: {search_duration:.2f} seconds.")
    return top_n_results_list


# --- Workflow: Optimized Transform & Search ---
def run_optimized_transform_workflow(
    spark: SparkSession,
    base_df_k_dim: DataFrame,
    query_vector_k_dim: np.ndarray,
    args: argparse.Namespace, 
    original_k_dim_feature_names: List[str]
):
    logger.info(f"--- Starting Optimized Transform & Search (Metric: {args.similarity_metric}, N={args.num_references_N}, Top {args.top_x_results}) ---")
    overall_optimized_start_time = time.time()

    logger.info(f"Phase 1: Selecting {args.num_references_N} reference vectors using '{args.similarity_metric}' metric...")
    ref_selection_start_time = time.time()
    
    logger.debug(f"Base K-dim DataFrame for sampling has {base_df_k_dim.rdd.getNumPartitions()} partitions.")
    if logger.isEnabledFor(logging.DEBUG):
         base_df_k_dim.select(*([args.id_column_name] + original_k_dim_feature_names)).limit(1).explain(extended=True) # Plan for sample op

    sample_fraction = min(1.0, (args.ref_selection_sample_size * 1.2) / args.num_data_rows if args.num_data_rows > 0 else 1.0)
    columns_for_sample = [args.id_column_name] + original_k_dim_feature_names
    
    logger.info(f"Taking sample for reference selection (approx. {args.ref_selection_sample_size} rows)...")
    pandas_sample_df = base_df_k_dim.select(*columns_for_sample) \
                                   .sample(withReplacement=False, fraction=sample_fraction, seed=42) \
                                   .limit(args.ref_selection_sample_size) \
                                   .toPandas() # ACTION
    logger.info(f"Sample taken (size: {len(pandas_sample_df)}). Selecting diverse references...")

    N_reference_vectors_K_dim = select_diverse_reference_vectors( # This is a local Python function call
        sample_dataframe=pandas_sample_df,
        feature_column_names=original_k_dim_feature_names,
        num_references_to_select=args.num_references_N,
        similarity_metric_name=args.similarity_metric, 
        use_median_for_center=args.use_median_for_center
    )
    actual_N_selected = N_reference_vectors_K_dim.shape[0]
    if actual_N_selected == 0:
        logger.error("No reference vectors were selected. Cannot proceed with optimized search.")
        return []
    logger.info(f"Selected {actual_N_selected} K-dim reference vectors in {time.time() - ref_selection_start_time:.2f}s. Broadcasting them...")
    broadcast_N_ref_vectors_K_dim = spark.sparkContext.broadcast(N_reference_vectors_K_dim)
    logger.info("Reference vectors broadcasted.")

    logger.info(f"Phase 2: Transforming dataset to N-dim space (N={actual_N_selected}) using '{args.similarity_metric}'...")
    transform_ds_start_time = time.time()
    
    if logger.isEnabledFor(logging.DEBUG):
        base_df_k_dim.explain(extended=True) # Plan for transformation input DF

    transform_udf_callable = partial(
        transform_to_similarity_space_udf_logic, 
        reference_vectors=broadcast_N_ref_vectors_K_dim.value,
        feature_column_names=original_k_dim_feature_names,
        id_column_name=args.id_column_name,
        similarity_metric_name=args.similarity_metric 
    )
    id_col_type = base_df_k_dim.schema[args.id_column_name].dataType
    transformed_output_schema_fields = [StructField(args.id_column_name, id_col_type, True)]
    for i in range(actual_N_selected):
        transformed_output_schema_fields.append(StructField(f"sim_to_ref_{i}", DoubleType(), True))
    transformed_output_schema = StructType(transformed_output_schema_fields)

    logger.info("Applying N-dim transformation UDF (mapInPandas)...")
    df_N_dim_transformed = base_df_k_dim.mapInPandas(transform_udf_callable, schema=transformed_output_schema)
    logger.info("N-dim transformation UDF application defined.")
    
    if args.save_transformed_data_path:
        full_save_path = os.path.join(args.data_path_prefix, args.save_transformed_data_path)
        logger.info(f"Saving N-dim transformed DataFrame to: {full_save_path}...")
        if logger.isEnabledFor(logging.DEBUG): df_N_dim_transformed.explain(extended=True)
        df_N_dim_transformed.write.mode("overwrite").parquet(full_save_path) # ACTION
        logger.info("N-dim DataFrame saved.")
    else:
        logger.info("Caching N-dim transformed DataFrame and triggering count...")
        df_N_dim_transformed.cache()
        if logger.isEnabledFor(logging.DEBUG): df_N_dim_transformed.explain(extended=True)
        df_N_dim_transformed.count() # ACTION
        logger.info("N-dim DataFrame cached and counted.")
        
    logger.info(f"Dataset transformation to N-dim took: {time.time() - transform_ds_start_time:.2f}s.")

    logger.info(f"Phase 3: Transforming K-dim query to N-dim space using '{args.similarity_metric}'...")
    # ... (lógica de transformación de query igual) ...
    from src.similarity_metrics import get_batch_similarity_function 
    query_transform_func = get_batch_similarity_function(args.similarity_metric, mode="batch_to_matrix")

    if query_vector_k_dim.size > 0 and N_reference_vectors_K_dim.size > 0:
         query_N_dim_similarities = query_transform_func(
             query_vector_k_dim.reshape(1, -1), 
             N_reference_vectors_K_dim          
         )
         query_vector_N_dim_transformed = query_N_dim_similarities[0] 
    elif actual_N_selected > 0 :
        query_vector_N_dim_transformed = np.zeros(actual_N_selected)
    else:
        query_vector_N_dim_transformed = np.array([])

    if query_vector_N_dim_transformed.size == 0 and actual_N_selected > 0:
        logger.error("N-dim transformed query vector is empty. Cannot perform search.")
        if not args.save_transformed_data_path: df_N_dim_transformed.unpersist()
        broadcast_N_ref_vectors_K_dim.destroy()
        return []
    logger.info("K-dim query transformed to N-dim.")


    logger.info(f"Phase 4: Performing Top-{args.top_x_results} search in N-dim space using '{args.similarity_metric}' for N-dim vectors...")
    search_N_dim_start_time = time.time()
    n_dim_feature_names_for_search = [f"sim_to_ref_{i}" for i in range(actual_N_selected)]
    
    if logger.isEnabledFor(logging.DEBUG):
        df_N_dim_transformed.explain(extended=True) # Plan for N-dim search input DF

    search_N_dim_udf_callable = partial(
        search_in_space_udf_logic, 
        query_vector=query_vector_N_dim_transformed,
        feature_column_names=n_dim_feature_names_for_search,
        id_column_name=args.id_column_name,
        top_x_per_partition=args.top_x_results,
        similarity_metric_name=args.similarity_metric 
    )
    search_output_schema_N_dim = StructType([
        StructField(args.id_column_name, id_col_type, True),
        StructField("similarity_score", DoubleType(), True)
    ])
    
    logger.info("Applying N-dim search UDF (mapInPandas)...")
    search_results_df_N_dim = df_N_dim_transformed.mapInPandas(search_N_dim_udf_callable, schema=search_output_schema_N_dim)
    logger.info("N-dim search UDF application complete. Converting to RDD for top N.")

    final_search_rdd_N_dim = search_results_df_N_dim.select(
        args.id_column_name, "similarity_score"
    ).rdd.map(lambda r: (r[0], r[1] if r[1] is not None else -float('inf')))
    
    logger.info("Collecting top N results from N-dim search RDD...")
    top_x_final_list_N_dim = get_top_n_from_rdd(final_search_rdd_N_dim, args.top_x_results)
    logger.info("Top N results from N-dim search collected.")
    
    logger.info(f"N-dim search took: {time.time() - search_N_dim_start_time:.2f}s.")
    
    if not args.save_transformed_data_path: df_N_dim_transformed.unpersist()
    broadcast_N_ref_vectors_K_dim.destroy()
    total_optimized_duration = time.time() - overall_optimized_start_time
    logger.info(f"Total Optimized Transform & Search workflow took: {total_optimized_duration:.2f} seconds.")
    return top_x_final_list_N_dim


# --- Workflow: Hierarchical K-Means Search ---
def run_hierarchical_kmeans_workflow(
    spark: SparkSession,
    base_df_k_dim: DataFrame,
    query_vector_k_dim: np.ndarray,
    args: argparse.Namespace, 
    k_dim_feature_names: List[str]
):
    logger.info(f"--- Starting Hierarchical K-Means Search (Search Metric: {args.similarity_metric}, Top {args.top_x_results}) ---")
    overall_hkmeans_start_time = time.time()

    logger.info("Phase 1: Orchestrating K-Means tree building or loading...")
    orchestrator_args = argparse.Namespace(**{
        **vars(args),
        "data_path": args.data_path_prefix, 
    })
    
    # orchestrate_distributed_tree_build_and_persist contains its own logging and the .collect() ACTION
    local_trees_collection = orchestrate_distributed_tree_build_and_persist(
        spark, base_df_k_dim, orchestrator_args, k_dim_feature_names
    )
    if not local_trees_collection:
        logger.error("Failed to build or load K-Means tree collection. Aborting hierarchical_kmeans workflow.")
        return []
    
    logger.info(f"Tree building/loading complete. {len(local_trees_collection)} trees processed. Broadcasting...")
    broadcast_local_trees = spark.sparkContext.broadcast(local_trees_collection)
    broadcast_query_k_dim_vec = spark.sparkContext.broadcast(query_vector_k_dim)
    logger.info("Trees and query vector broadcasted.")

    logger.info(f"Phase 2: Orchestrating distributed search using K-Means trees and '{args.similarity_metric}' metric...")
    search_hkmeans_start_time = time.time()
    id_col_type = base_df_k_dim.schema[args.id_column_name].dataType
    
    # orchestrate_distributed_hierarchical_search contains the .collect() ACTION for search results
    top_x_results = orchestrate_distributed_hierarchical_search(
        spark, base_df_k_dim, broadcast_local_trees, broadcast_query_k_dim_vec,
        orchestrator_args, k_dim_feature_names, id_col_type,
        args.similarity_metric 
    )
    logger.info(f"Distributed K-Means search completed in {time.time() - search_hkmeans_start_time:.2f}s.")

    broadcast_local_trees.destroy()
    broadcast_query_k_dim_vec.destroy()
    total_hkmeans_duration = time.time() - overall_hkmeans_start_time
    logger.info(f"Total Hierarchical K-Means workflow took: {total_hkmeans_duration:.2f} seconds.")
    return top_x_results


# --- Main Orchestrator ---
def main():
    args = parse_arguments() 
    script_start_time = time.time()
    logger.info(f"Starting main orchestrator: App '{args.app_name}', Mode '{args.search_mode}', Metric '{args.similarity_metric}', Python Log Level '{args.python_log_level}', Spark Log Level '{args.spark_log_level}'")

    spark_master_url = f"local[{args.spark_cores.replace('[','').replace(']','').replace('auto','*')}]"
    spark = init_spark_session(
        args.app_name,
        spark_master_url,
        args.driver_memory,
        log_level=args.spark_log_level
    )
    
    target_processing_partitions = args.num_partitions 
    initial_data_partitions = max(target_processing_partitions, target_processing_partitions * 2 if target_processing_partitions < 8 else target_processing_partitions)
    logger.info(f"Targeting {target_processing_partitions} Spark partitions for main processing, {initial_data_partitions} for initial data gen.")

    base_data_parquet_path = os.path.join(args.data_path_prefix, f"k_dim_data_{args.num_data_rows}r_{args.num_data_cols}c")
    
    logger.info(f"Getting or creating K-dim DataFrame from: {base_data_parquet_path}")
    base_df_k_dim_initial = get_or_create_parquet_dataframe( # Contains its own logging
        spark, base_data_parquet_path, args.num_data_rows, args.num_data_cols,
        args.id_column_name, initial_data_partitions, args.regenerate_data
    )
    
    logger.info(f"Repartitioning base DataFrame to {target_processing_partitions} partitions (if necessary)...")
    base_df_k_dim_processed = repartition_dataframe(base_df_k_dim_initial, target_processing_partitions) # Contains its own logging
    
    logger.info(f"Caching and counting base K-dim DataFrame (partitions: {base_df_k_dim_processed.rdd.getNumPartitions()})...")
    if logger.isEnabledFor(logging.DEBUG):
        base_df_k_dim_processed.explain(extended=True)
    base_df_k_dim_processed.cache().count() # ACTION
    logger.info("Base K-dim DataFrame prepared and cached.")

    query_vector_k_dim = create_query_vector(args.num_data_cols, args.query_vector_seed)
    if query_vector_k_dim.size == 0 and args.num_data_cols > 0 :
        logger.warning(f"Query vector is empty for {args.num_data_cols} features. Search results may be affected.")

    k_dim_feature_names = [f"col_{i}" for i in range(args.num_data_cols)]
    final_results = []

    try:
        logger.info(f"Executing search mode: {args.search_mode}")
        if args.search_mode == "absolute":
            final_results = run_absolute_search_workflow(
                spark, base_df_k_dim_processed, query_vector_k_dim, args, k_dim_feature_names
            )
        elif args.search_mode == "optimized_transform":
            final_results = run_optimized_transform_workflow(
                spark, base_df_k_dim_processed, query_vector_k_dim, args, k_dim_feature_names
            )
        elif args.search_mode == "hierarchical_kmeans":
            final_results = run_hierarchical_kmeans_workflow(
                spark, base_df_k_dim_processed, query_vector_k_dim, args, k_dim_feature_names
            )
        else:
            logger.critical(f"Critical Error: Unknown search mode '{args.search_mode}'. This should have been caught by argparse.")
            if spark: spark.stop()
            sys.exit(1)

        logger.info(f"Workflow '{args.search_mode}' completed. Printing results summary...")
        print_results_summary(final_results, args.top_x_results, f"{args.search_mode} (metric: {args.similarity_metric})")

    except Exception as e: # Catch-all for unexpected errors in workflows
        logger.critical(f"A critical error occurred during the '{args.search_mode}' workflow.")
        logger.critical(f"Error type: {type(e).__name__}")
        logger.critical(f"Error message: {str(e)}")
        logger.critical("Traceback (Python):\n" + traceback.format_exc())
        # If this happens, it means an error wasn't caught cleanly within the workflow function
        # or it's an infrastructure issue (like Spark context dying).
    finally:
        if 'spark' in locals() and spark and spark.sparkContext._jsc is not None : # Check if Spark context is still active
            logger.info("Cleaning up and stopping Spark session...")
            if 'base_df_k_dim_processed' in locals() and base_df_k_dim_processed:
                try:
                    # Check if the DataFrame is actually cached before unpersisting
                    if base_df_k_dim_processed.is_cached:
                         logger.debug("Unpersisting base_df_k_dim_processed.")
                         base_df_k_dim_processed.unpersist()
                    else:
                         logger.debug("base_df_k_dim_processed was not cached, no need to unpersist.")
                except Exception: # pylint: disable=broad-except
                    logger.warning("Could not unpersist base_df_k_dim_processed during cleanup.", exc_info=False) # exc_info=False to avoid duplicate traceback if already logged
            spark.stop()
            logger.info("Spark session stopped.")
        else:
            logger.info("Spark session was not active or already stopped at final cleanup.")

        script_duration = time.time() - script_start_time
        logger.info(f"Total script execution time: {script_duration:.2f} seconds.")


if __name__ == "__main__":
    main()
