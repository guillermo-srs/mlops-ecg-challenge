# File: src/distributed_tree_orchestrator.py

import os
import pickle # Still used for saving the collection of NodeMapType to disk on driver
import time
from functools import partial
from typing import List, Dict, Any, Optional, Tuple, Iterator, Callable
import traceback

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, BinaryType, IntegerType, DoubleType, DataType
from pyspark.sql.functions import col as spark_col, spark_partition_id
from pyspark.broadcast import Broadcast

from .hierarchical_kmeans_indexer import (
    build_local_hierarchical_kmeans_index, # Now returns NodeMapType or None
    search_in_local_tree_index,
    NodeMapType # Type alias for Dict[str, TreeNode]
)
# Import new serializer
from .tree_serializer import serialize_tree, deserialize_tree 
from .spark_utils import get_top_n_from_rdd


def _build_tree_on_partition_udf_logic(
    pandas_df_partition: pd.DataFrame,
    id_col_name_param: str,
    k_dim_cols_param: List[str],
    k_per_level_param: int,
    max_depth_param: int,
    min_leaf_size_param: int,
    kmeans_seed_param: int,
    kmeans_use_minibatch_param: bool,
    similarity_metric_name_param: str # For representative point selection
) -> pd.DataFrame:
    serialized_tree_bytes: Optional[bytes] = None
    partition_id_val = None
    pid_for_logging = "N/A_build"

    try:
        if not pandas_df_partition.empty and "pid_group_for_udf" in pandas_df_partition.columns:
             partition_id_val = pandas_df_partition["pid_group_for_udf"].iloc[0]
             pid_for_logging = str(partition_id_val)
        elif pandas_df_partition.empty:
            return pd.DataFrame({"pid_group_out": [None], "pickled_tree_bytes_out": [None]})

        if not pandas_df_partition.empty and k_dim_cols_param:
            node_map, _ = build_local_hierarchical_kmeans_index(
                pandas_df_partition,
                id_column_name=id_col_name_param,
                k_dim_feature_column_names=k_dim_cols_param,
                k_clusters_per_level=k_per_level_param,
                max_tree_depth=max_depth_param,
                min_samples_per_leaf=min_leaf_size_param,
                similarity_metric_name=similarity_metric_name_param, # Pass to find representatives
                kmeans_seed=kmeans_seed_param,
                kmeans_use_minibatch=kmeans_use_minibatch_param
            )
            
            if node_map is not None: # Check if a valid tree (node_map) was built
                serialized_tree_bytes = serialize_tree(node_map) # Use new serializer
            # else: node_map is None, serialized_tree_bytes remains None
                
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"[UDF ERROR - PID {pid_for_logging}] EXCEPTION in _build_tree_on_partition_udf_logic:")
        print(f"[UDF ERROR - PID {pid_for_logging}] Type: {type(e).__name__}, Msg: {str(e)}")
        print(f"[UDF ERROR - PID {pid_for_logging}] Traceback:\n{traceback.format_exc()}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise e

    return pd.DataFrame({
        "pid_group_out": [partition_id_val],
        "pickled_tree_bytes_out": [serialized_tree_bytes] 
    })


def orchestrate_distributed_tree_build_and_persist(
    spark: SparkSession,
    k_dimensional_df: DataFrame,
    args_config: Any, 
    feature_column_names: List[str]
) -> Optional[Dict[int, NodeMapType]]: # Now stores NodeMapType
    tree_collection_storage_path = os.path.join(args_config.data_path, args_config.tree_filename)
    local_node_maps_by_pid: Dict[int, NodeMapType] = {} 

    if not args_config.force_rebuild_tree and os.path.exists(tree_collection_storage_path):
        print(f"INFO: Loading K-Means tree collection (NodeMap objects) from: {tree_collection_storage_path}")
        load_time_start = time.time()
        try:
            with open(tree_collection_storage_path, "rb") as f:
                local_node_maps_by_pid = pickle.load(f) # Loading a dict of NodeMapType
            load_time_end = time.time()
            print(f"INFO: Loaded {len(local_node_maps_by_pid)} local node maps in {load_time_end - load_time_start:.2f}s.")
            if local_node_maps_by_pid:
                 return local_node_maps_by_pid
        except Exception as e:
            print(f"WARNING: Failed to load node map collection from {tree_collection_storage_path}: {e}. Will attempt to rebuild.")
            local_node_maps_by_pid = {}

    num_partitions_for_build = k_dimensional_df.rdd.getNumPartitions()
    print(f"INFO: Starting distributed K-Means tree building for {num_partitions_for_build} partitions.")
    build_time_start = time.time()

    udf_output_schema_tree_build = StructType([
        StructField("pid_group_out", IntegerType(), True),
        StructField("pickled_tree_bytes_out", BinaryType(), True)
    ])

    build_tree_callable_udf = partial(
        _build_tree_on_partition_udf_logic,
        id_col_name_param=args_config.id_column_name,
        k_dim_cols_param=feature_column_names,
        k_per_level_param=args_config.k_per_level,
        max_depth_param=args_config.max_depth,
        min_leaf_size_param=args_config.min_leaf_size,
        kmeans_seed_param=args_config.kmeans_seed,
        kmeans_use_minibatch_param=getattr(args_config, 'kmeans_use_minibatch', True),
        similarity_metric_name_param=args_config.similarity_metric # Pass for representative selection
    )
    
    df_with_pid_for_build = k_dimensional_df.withColumn("pid_group_for_udf", spark_partition_id())
    
    print(f"INFO: Applying tree building UDF to {num_partitions_for_build} partitions...")
    pickled_trees_spark_df = df_with_pid_for_build.groupBy("pid_group_for_udf").applyInPandas(
        build_tree_callable_udf, schema=udf_output_schema_tree_build
    )
    
    print(f"INFO: Attempting to collect serialized tree components from executors...")
    try:
        collected_tree_bytes_rows = pickled_trees_spark_df.collect()
        print(f"INFO: Successfully collected {len(collected_tree_bytes_rows)} rows of serialized tree data.")
    except Exception as e:
        print(f"ERROR: .collect() failed: {e}. Likely due to large results or UDF errors.")
        return None

    processed_trees_count = 0
    for row in collected_tree_bytes_rows:
        pid_out = row["pid_group_out"]
        serialized_bytes = row["pickled_tree_bytes_out"]
        
        if serialized_bytes is not None and pid_out is not None:
            node_map = deserialize_tree(serialized_bytes) # Use new deserializer
            if node_map is not None:
                local_node_maps_by_pid[pid_out] = node_map
                processed_trees_count +=1
            else:
                print(f"ERROR: Failed to deserialize tree components for partition ID {pid_out}.")
        # ... (manejo de otros casos) ...

    build_time_end = time.time()
    print(f"INFO: Distributed tree building and component deserialization completed in {build_time_end - build_time_start:.2f}s. {processed_trees_count} trees successfully processed.")

    if local_node_maps_by_pid:
        print(f"INFO: Saving collection of {len(local_node_maps_by_pid)} node maps to {tree_collection_storage_path}.")
        save_time_start = time.time()
        try:
            os.makedirs(os.path.dirname(tree_collection_storage_path), exist_ok=True)
            with open(tree_collection_storage_path, "wb") as f:
                pickle.dump(local_node_maps_by_pid, f) # Pickle the dict of NodeMapType
            save_time_end = time.time()
            print(f"INFO: Node map collection saved in {save_time_end - save_time_start:.2f}s.")
        except Exception as e:
            print(f"ERROR: Error saving node map collection to {tree_collection_storage_path}: {e}")
    elif not args_config.force_rebuild_tree and os.path.exists(tree_collection_storage_path):
        print(f"ERROR: Failed to load trees from {tree_collection_storage_path} and subsequent build yielded no results.")
        return None
    else:
        print(f"WARNING: No K-Means trees were built or loaded.")
        return None
        
    return local_node_maps_by_pid


def _search_partition_with_local_tree_udf_logic(
    pandas_df_partition_data: pd.DataFrame,
    query_vector_k_dim_bcast_val: np.ndarray,
    local_node_maps_collection_bcast_val: Dict[int, NodeMapType], # Broadcast Dict of NodeMapType
    id_col_name_param: str,
    k_dim_cols_param: List[str], 
    top_x_results_param: int,
    similarity_metric_name_param: str 
) -> pd.DataFrame:
    pid_for_logging = "N/A_search"
    try:
        if pandas_df_partition_data.empty:
            return pd.DataFrame(columns=[id_col_name_param, "similarity_score"])

        if "pid_group_for_udf" in pandas_df_partition_data.columns:
            pid_for_logging = str(pandas_df_partition_data["pid_group_for_udf"].iloc[0])
        
        partition_id = pandas_df_partition_data["pid_group_for_udf"].iloc[0]
        node_map_for_this_partition = local_node_maps_collection_bcast_val.get(partition_id)

        if node_map_for_this_partition is None:
            return pd.DataFrame(columns=[id_col_name_param, "similarity_score"])

        # search_in_local_tree_index now expects node_map directly
        top_x_results_for_partition = search_in_local_tree_index(
            query_k_dim_vector=query_vector_k_dim_bcast_val,
            node_map=node_map_for_this_partition, 
            partition_data_df=pandas_df_partition_data.drop(columns=["pid_group_for_udf"], errors='ignore'),
            id_column_name=id_col_name_param,
            k_dim_feature_column_names=k_dim_cols_param, 
            top_x_results_to_find=top_x_results_param,
            similarity_metric_name=similarity_metric_name_param 
        )

        if top_x_results_for_partition:
            ids, scores = zip(*top_x_results_for_partition)
            return pd.DataFrame({id_col_name_param: list(ids), "similarity_score": list(scores)})
        else:
            return pd.DataFrame(columns=[id_col_name_param, "similarity_score"])
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"[UDF ERROR - Search PID {pid_for_logging}] EXCEPTION in _search_partition_with_local_tree_udf_logic:")
        print(f"[UDF ERROR - Search PID {pid_for_logging}] Type: {type(e).__name__}, Msg: {str(e)}")
        print(f"[UDF ERROR - Search PID {pid_for_logging}] Traceback:\n{traceback.format_exc()}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise e


def orchestrate_distributed_hierarchical_search(
    spark: SparkSession,
    k_dimensional_df: DataFrame,
    broadcast_local_node_maps: Broadcast[Dict[int, NodeMapType]], # Broadcast Dict of NodeMapType
    broadcast_query_k_dim: Broadcast[np.ndarray],
    args_config: Any, 
    feature_column_names: List[str], 
    id_column_spark_type: DataType,
    similarity_metric_name: str 
) -> List[Tuple[Any, float]]:
    search_udf_output_schema = StructType([
        StructField(args_config.id_column_name, id_column_spark_type, True),
        StructField("similarity_score", DoubleType(), True)
    ])

    search_callable_udf = partial(
        _search_partition_with_local_tree_udf_logic,
        query_vector_k_dim_bcast_val=broadcast_query_k_dim.value,
        local_node_maps_collection_bcast_val=broadcast_local_node_maps.value, 
        id_col_name_param=args_config.id_column_name,
        k_dim_cols_param=feature_column_names, 
        top_x_results_param=args_config.top_x_results,
        similarity_metric_name_param=similarity_metric_name 
    )
    
    df_with_pid_for_search = k_dimensional_df.withColumn("pid_group_for_udf", spark_partition_id())
    all_partitions_top_x_df = df_with_pid_for_search.groupBy("pid_group_for_udf").applyInPandas(
        search_callable_udf,
        schema=search_udf_output_schema
    )

    final_search_rdd = all_partitions_top_x_df.select(
        args_config.id_column_name, "similarity_score"
    ).rdd.map(lambda r: (r[0], r[1] if r[1] is not None else -float('inf')))
    
    global_top_x_results = get_top_n_from_rdd(final_search_rdd, args_config.top_x_results)
    return global_top_x_results
