# File: src/hierarchical_kmeans_indexer.py

from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans, MiniBatchKMeans
from dataclasses import dataclass, field
import heapq

from .similarity_metrics import get_batch_similarity_function, BatchSimilarityFunctionMatrixToVector

# --- Tree Data Structures ---
@dataclass
class TreeNode:
    node_id: str
    depth: int
    is_leaf: bool = False
    member_count: int = 0
    # Stores the index (relative to the original partition_data_df) 
    # of the data point chosen to represent this node's cluster center.
    representative_point_original_idx: Optional[int] = None
    # Child node IDs are immutable after creation for a given node
    child_node_ids: Tuple[str, ...] = field(default_factory=tuple) 
    # For leaf nodes, stores indices (relative to original partition_data_df) of their members
    local_data_indices_in_leaf: Optional[np.ndarray] = None


NodeMapType = Dict[str, TreeNode]


def _train_kmeans_on_subset(
    feature_vectors_subset: np.ndarray,
    k_clusters_for_level: int,
    kmeans_seed: int,
    use_minibatch: bool = True,
    minibatch_batch_size: int = 1024,
    minibatch_n_init: int = 3 
) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    num_samples_in_subset = feature_vectors_subset.shape[0]
    actual_k_clusters = min(k_clusters_for_level, num_samples_in_subset)

    if actual_k_clusters < 2:
        return None, None

    if use_minibatch:
        kmeans_model = MiniBatchKMeans(
            n_clusters=actual_k_clusters,
            random_state=kmeans_seed,
            batch_size=minibatch_batch_size,
            n_init=minibatch_n_init,
            max_iter=100 
        )
    else:
        kmeans_model = SklearnKMeans(
            n_clusters=actual_k_clusters,
            random_state=kmeans_seed,
            n_init='auto'
        )
    
    try:
        # Ensure data is C-contiguous, float32 for scikit-learn compatibility and efficiency
        contiguous_subset = np.ascontiguousarray(feature_vectors_subset, dtype=np.float32)
        cluster_labels = kmeans_model.fit_predict(contiguous_subset)
    except Exception:
        # Logging should be handled by the calling UDF or main application logger
        return None, None
        
    return kmeans_model, cluster_labels


def _find_representative_point_for_cluster(
    calculated_centroid_vector: np.ndarray,
    points_in_cluster_vectors: np.ndarray,
    original_indices_of_points_in_cluster: np.ndarray,
    similarity_metric_name: str
) -> Optional[int]:
    """
    Finds the actual data point within a cluster that is most similar
    to the calculated K-Means centroid of that cluster.
    Returns the original index (from partition_data_df) of this representative point.
    """
    if points_in_cluster_vectors.size == 0 or original_indices_of_points_in_cluster.size == 0:
        return None

    calculate_similarity: BatchSimilarityFunctionMatrixToVector = \
        get_batch_similarity_function(similarity_metric_name, mode="matrix_to_vector")

    # Norm of calculated_centroid_vector for cosine, pass None for others
    target_vector_norm = np.linalg.norm(calculated_centroid_vector) if similarity_metric_name == "cosine" else None

    similarities = calculate_similarity(
        points_in_cluster_vectors, # batch_matrix_a
        calculated_centroid_vector,  # vector_b
        target_vector_norm         # vector_b_norm
    )

    if similarities.size == 0: # Should not happen if points_in_cluster_vectors is not empty
        return None
        
    # Find the index (within points_in_cluster_vectors) of the point most similar to the centroid
    idx_of_most_similar_in_subset = np.nanargmax(similarities) # Higher score is better
    
    # Get the original index (from partition_data_df) of this representative point
    representative_original_idx = original_indices_of_points_in_cluster[idx_of_most_similar_in_subset]
    return int(representative_original_idx) # Ensure it's a Python int


def _recursively_build_local_tree_level(
    feature_vectors_for_node: np.ndarray, # Subset of features for current node's K-Means
    local_indices_for_node: np.ndarray,   # Original indices (from partition_data_df) for these feature_vectors
    k_clusters_per_level: int,
    current_depth: int,
    max_tree_depth: int,
    min_samples_per_leaf: int,
    kmeans_seed: int,
    parent_node_id: str,
    tree_node_map: NodeMapType,
    kmeans_use_minibatch: bool,
    similarity_metric_name: str # For selecting representative points
) -> None:
    current_node_sample_count = feature_vectors_for_node.shape[0]
    tree_node_map[parent_node_id].member_count = current_node_sample_count

    if current_node_sample_count < min_samples_per_leaf or current_depth >= max_tree_depth:
        tree_node_map[parent_node_id].is_leaf = True
        tree_node_map[parent_node_id].local_data_indices_in_leaf = local_indices_for_node.astype(np.int32) # Store as int32
        return

    kmeans_model, cluster_labels_for_subset = _train_kmeans_on_subset(
        feature_vectors_for_node, k_clusters_per_level, kmeans_seed, kmeans_use_minibatch
    )

    if kmeans_model is None or cluster_labels_for_subset is None:
        tree_node_map[parent_node_id].is_leaf = True
        tree_node_map[parent_node_id].local_data_indices_in_leaf = local_indices_for_node.astype(np.int32)
        return

    child_node_ids_created_list = []
    for i in range(kmeans_model.n_clusters):
        child_node_id = f"{parent_node_id}_{i}"
        child_node_ids_created_list.append(child_node_id)

        calculated_centroid_vec = kmeans_model.cluster_centers_[i].astype(np.float32)
        
        member_mask_for_child = (cluster_labels_for_subset == i)
        feature_vectors_for_child_cluster = feature_vectors_for_node[member_mask_for_child]
        original_indices_for_child_cluster = local_indices_for_node[member_mask_for_child]

        representative_idx: Optional[int] = None
        if feature_vectors_for_child_cluster.shape[0] > 0:
            representative_idx = _find_representative_point_for_cluster(
                calculated_centroid_vec,
                feature_vectors_for_child_cluster,
                original_indices_for_child_cluster,
                similarity_metric_name
            )

        new_child_node = TreeNode(
            node_id=child_node_id,
            depth=current_depth + 1,
            representative_point_original_idx=representative_idx
        )
        tree_node_map[child_node_id] = new_child_node
        
        if feature_vectors_for_child_cluster.shape[0] > 0:
            _recursively_build_local_tree_level(
                feature_vectors_for_child_cluster, original_indices_for_child_cluster,
                k_clusters_per_level, current_depth + 1, max_tree_depth,
                min_samples_per_leaf, kmeans_seed, child_node_id,
                tree_node_map, kmeans_use_minibatch, similarity_metric_name
            )
        else:
            tree_node_map[child_node_id].is_leaf = True
            tree_node_map[child_node_id].member_count = 0
            
    tree_node_map[parent_node_id].child_node_ids = tuple(child_node_ids_created_list)


def build_local_hierarchical_kmeans_index(
    partition_pandas_df: pd.DataFrame,
    id_column_name: str, 
    k_dim_feature_column_names: List[str],
    k_clusters_per_level: int,
    max_tree_depth: int,
    min_samples_per_leaf: int,
    similarity_metric_name: str, # For selecting representatives
    kmeans_seed: int = 42,
    kmeans_use_minibatch: bool = True
) -> Tuple[Optional[NodeMapType], pd.DataFrame]:
    if partition_pandas_df.empty or not k_dim_feature_column_names:
        return None, partition_pandas_df

    feature_vectors_k_dim_numpy = partition_pandas_df[k_dim_feature_column_names].to_numpy(dtype=np.float32)
    original_local_indices_in_partition = np.arange(feature_vectors_k_dim_numpy.shape[0], dtype=np.int32)

    tree_node_map: NodeMapType = {}
    root_node_id = "local_root"
    root_node = TreeNode(node_id=root_node_id, depth=0)
    tree_node_map[root_node_id] = root_node
    
    _recursively_build_local_tree_level(
        feature_vectors_k_dim_numpy, original_local_indices_in_partition,
        k_clusters_per_level, 0, max_tree_depth, min_samples_per_leaf,
        kmeans_seed, root_node_id, tree_node_map,
        kmeans_use_minibatch, similarity_metric_name
    )
    
    # Check if any actual structure was built beyond the root
    if len(tree_node_map) <= 1 and not tree_node_map[root_node_id].child_node_ids and not tree_node_map[root_node_id].is_leaf:
        # This means only root was created and it's not even a leaf (e.g. initial dataset too small)
        return None, partition_pandas_df
        
    return tree_node_map, partition_pandas_df


def search_in_local_tree_index(
    query_k_dim_vector: np.ndarray,
    node_map: NodeMapType, # Now receives the node_map directly
    partition_data_df: pd.DataFrame, # Still needed to get representative vectors and leaf data
    id_column_name: str,
    k_dim_feature_column_names: List[str],
    top_x_results_to_find: int,
    similarity_metric_name: str
) -> List[Tuple[Any, float]]:
    if not node_map or "local_root" not in node_map or query_k_dim_vector.size == 0:
        return []

    query_k_dim_vector_f32 = query_k_dim_vector.astype(np.float32)
    calculate_similarity_to_target: BatchSimilarityFunctionMatrixToVector = \
        get_batch_similarity_function(similarity_metric_name, mode="matrix_to_vector")
    query_norm_for_cosine = np.linalg.norm(query_k_dim_vector_f32) if similarity_metric_name == "cosine" else None

    current_node_id = "local_root"
    current_node = node_map[current_node_id]

    while not current_node.is_leaf and current_node.child_node_ids:
        best_matching_child_id: Optional[str] = None
        max_similarity_to_child_representative = -float('inf')

        for child_id in current_node.child_node_ids:
            child_node = node_map.get(child_id)
            if child_node and child_node.representative_point_original_idx is not None:
                # Get the representative point's vector from the original partition data
                rep_idx = child_node.representative_point_original_idx
                # Ensure to use .iloc for positional index and select feature columns
                representative_vector = partition_data_df[k_dim_feature_column_names].iloc[rep_idx].to_numpy(dtype=np.float32)
                
                similarity_score_array = calculate_similarity_to_target(
                    query_k_dim_vector_f32.reshape(1, -1),
                    representative_vector,
                    None # Norm of representative_vector calculated by metric function if needed
                )
                similarity = similarity_score_array[0]

                if similarity > max_similarity_to_child_representative:
                    max_similarity_to_child_representative = similarity
                    best_matching_child_id = child_id
        
        if best_matching_child_id:
            current_node_id = best_matching_child_id
            current_node = node_map[current_node_id]
        else: # No valid child representative found or no children improved similarity
            current_node.is_leaf = True 
            break 
    
    if not current_node.is_leaf or \
       current_node.local_data_indices_in_leaf is None or \
       len(current_node.local_data_indices_in_leaf) == 0:
        return []

    leaf_member_local_indices = current_node.local_data_indices_in_leaf # Already np.int32
    leaf_member_features_numpy = partition_data_df.iloc[leaf_member_local_indices][k_dim_feature_column_names].to_numpy(dtype=np.float32)
    leaf_member_ids_list = partition_data_df.iloc[leaf_member_local_indices][id_column_name].tolist()

    if leaf_member_features_numpy.size == 0:
        return []

    scores_in_leaf = calculate_similarity_to_target(
        leaf_member_features_numpy,
        query_k_dim_vector_f32,
        query_norm_for_cosine 
    )
    
    leaf_top_x_heap = []
    tie_breaker = 0
    for i in range(len(leaf_member_ids_list)):
        score = scores_in_leaf[i]
        if not np.isnan(score):
            tie_breaker +=1
            item = (score, tie_breaker, leaf_member_ids_list[i])
            if len(leaf_top_x_heap) < top_x_results_to_find:
                heapq.heappush(leaf_top_x_heap, item)
            elif score > leaf_top_x_heap[0][0]: # Min-heap stores smallest at heap[0]
                heapq.heapreplace(leaf_top_x_heap, item)
    
    return [(item[2], item[0]) for item in leaf_top_x_heap]
