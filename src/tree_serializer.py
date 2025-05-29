# File: src/tree_serializer.py

import io
import pickle # For node_id to int_id mapping and non-NumPy parts of TreeNode
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# Import TreeNode definition to help with type hints and structure
from .hierarchical_kmeans_indexer import TreeNode, NodeMapType


# Helper to ensure consistent data types for arrays
def _get_dtype_or_default(arr: Optional[np.ndarray], default_dtype=np.int32) -> np.dtype:
    if arr is not None and arr.size > 0:
        return arr.dtype
    return default_dtype

def serialize_tree_to_arrays(node_map: NodeMapType) -> Dict[str, Any]:
    """
    Flattens the tree structure (NodeMapType) into a dictionary of NumPy arrays
    and a pickled mapping for node_id strings to integer indices.
    """
    if not node_map:
        return {
            "node_id_map_pickled": pickle.dumps({}),
            "node_attributes": np.array([], dtype=np.int32).reshape(0,4), # depth, is_leaf, member_count, rep_idx
            "child_links_flat": np.array([], dtype=np.int32),
            "child_links_pointers": np.array([0], dtype=np.int32),
            "leaf_indices_flat": np.array([], dtype=np.int32),
            "leaf_indices_pointers": np.array([0], dtype=np.int32),
            "original_node_ids": np.array([], dtype=object) # To store original string node_ids
        }

    # Create a mapping from string node_id to integer index for array representation
    # Sort node_ids to ensure deterministic mapping if order matters (it does for pointers)
    sorted_node_ids = sorted(node_map.keys())
    node_id_to_int_idx = {node_id_str: i for i, node_id_str in enumerate(sorted_node_ids)}
    int_idx_to_node_id_str = {i: node_id_str for node_id_str, i in node_id_to_int_idx.items()}


    num_nodes = len(sorted_node_ids)
    
    # Initialize arrays
    # Columns: depth, is_leaf (0/1), member_count, representative_point_original_idx (-1 if None)
    node_attributes = np.zeros((num_nodes, 4), dtype=np.int32)
    
    all_child_links_flat_list: List[int] = []
    child_links_pointers = np.zeros(num_nodes + 1, dtype=np.int32)
    
    all_leaf_indices_flat_list: List[int] = []
    leaf_indices_pointers = np.zeros(num_nodes + 1, dtype=np.int32)
    original_node_ids_array = np.empty(num_nodes, dtype=object)


    current_child_ptr = 0
    current_leaf_idx_ptr = 0

    for i, node_id_str in enumerate(sorted_node_ids): # Iterate in sorted order
        node = node_map[node_id_str]
        int_node_idx = node_id_to_int_idx[node_id_str] # Should be == i

        original_node_ids_array[int_node_idx] = node_id_str
        node_attributes[int_node_idx, 0] = node.depth
        node_attributes[int_node_idx, 1] = 1 if node.is_leaf else 0
        node_attributes[int_node_idx, 2] = node.member_count
        node_attributes[int_node_idx, 3] = node.representative_point_original_idx if node.representative_point_original_idx is not None else -1
        
        child_links_pointers[int_node_idx] = current_child_ptr
        if node.child_node_ids: # Ensure it's a tuple of strings
            for child_id_str in node.child_node_ids:
                all_child_links_flat_list.append(node_id_to_int_idx[child_id_str])
            current_child_ptr += len(node.child_node_ids)
        
        leaf_indices_pointers[int_node_idx] = current_leaf_idx_ptr
        if node.is_leaf and node.local_data_indices_in_leaf is not None and node.local_data_indices_in_leaf.size > 0:
            all_leaf_indices_flat_list.extend(node.local_data_indices_in_leaf.astype(np.int32))
            current_leaf_idx_ptr += len(node.local_data_indices_in_leaf)

    child_links_pointers[num_nodes] = current_child_ptr
    leaf_indices_pointers[num_nodes] = current_leaf_idx_ptr
    
    # Store the int_idx_to_node_id_str mapping as well, pickled, as it's needed for reconstruction
    # No, store node_id_to_int_idx and original_node_ids_array instead. The latter gives int_idx -> str_id
    
    return {
        "node_id_to_int_idx_pickled": pickle.dumps(node_id_to_int_idx), # Map str_id -> int_idx
        "original_node_ids_array": original_node_ids_array, # Array of str_id, index is int_idx
        "node_attributes": node_attributes,
        "child_links_flat": np.array(all_child_links_flat_list, dtype=np.int32),
        "child_links_pointers": child_links_pointers,
        "leaf_indices_flat": np.array(all_leaf_indices_flat_list, dtype=np.int32),
        "leaf_indices_pointers": leaf_indices_pointers
    }

def deserialize_tree_from_arrays(arrays_dict: Dict[str, np.ndarray]) -> NodeMapType:
    """
    Reconstructs the NodeMapType (Dict[str, TreeNode]) from a dictionary of NumPy arrays.
    """
    node_map: NodeMapType = {}

    try:
        # node_id_to_int_idx_map = pickle.loads(arrays_dict["node_id_to_int_idx_pickled"])
        original_node_ids_arr = arrays_dict["original_node_ids_array"]
        node_attributes = arrays_dict["node_attributes"]
        child_links_flat = arrays_dict["child_links_flat"]
        child_links_pointers = arrays_dict["child_links_pointers"]
        leaf_indices_flat = arrays_dict["leaf_indices_flat"]
        leaf_indices_pointers = arrays_dict["leaf_indices_pointers"]
    except KeyError as e:
        # print(f"Deserialization error: Missing key {e} in arrays_dict") # Log this if in main app
        raise ValueError(f"Serialized tree data is missing expected array: {e}")


    if original_node_ids_arr.size == 0 and node_attributes.shape[0] == 0 : # Empty tree case
        return {}

    num_nodes = node_attributes.shape[0]
    if num_nodes != len(original_node_ids_arr):
        raise ValueError("Mismatch between node_attributes and original_node_ids_array lengths during deserialization.")

    # Create all TreeNode objects first
    for i in range(num_nodes):
        node_id_str = original_node_ids_arr[i]
        
        rep_idx = int(node_attributes[i, 3])
        node_map[node_id_str] = TreeNode(
            node_id=node_id_str,
            depth=int(node_attributes[i, 0]),
            is_leaf=(node_attributes[i, 1] == 1),
            member_count=int(node_attributes[i, 2]),
            representative_point_original_idx=rep_idx if rep_idx != -1 else None,
            child_node_ids=tuple(), # Will be populated next
            local_data_indices_in_leaf=None # Will be populated next
        )

    # Populate child_node_ids and local_data_indices_in_leaf
    for i in range(num_nodes):
        node_id_str = original_node_ids_arr[i]
        node = node_map[node_id_str]
        
        # Populate children
        child_start = child_links_pointers[i]
        child_end = child_links_pointers[i+1]
        if child_start < child_end : # Check if there are children
            child_int_indices = child_links_flat[child_start:child_end]
            node.child_node_ids = tuple(original_node_ids_arr[int_child_idx] for int_child_idx in child_int_indices)
            
        # Populate leaf data
        if node.is_leaf:
            leaf_start = leaf_indices_pointers[i]
            leaf_end = leaf_indices_pointers[i+1]
            if leaf_start < leaf_end : # Check if there is leaf data
                 node.local_data_indices_in_leaf = leaf_indices_flat[leaf_start:leaf_end].astype(np.int32)
            else: # is_leaf but no data, e.g. empty cluster became leaf.
                 node.local_data_indices_in_leaf = np.array([], dtype=np.int32)


    return node_map


def serialize_tree(node_map: NodeMapType) -> bytes:
    """
    Serializes tree components to bytes using numpy.savez_compressed.
    """
    if not node_map: # Handle case of empty or None tree
        # Serialize an empty structure or a specific marker for it
        arrays_to_save = serialize_tree_to_arrays(None) # Get structure for empty
    else:
        arrays_to_save = serialize_tree_to_arrays(node_map)
    
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays_to_save)
    buffer.seek(0)
    return buffer.getvalue()


def deserialize_tree(serialized_bytes: bytes) -> Optional[NodeMapType]:
    """
    Deserializes bytes (created by serialize_tree) back into a NodeMapType.
    Returns None if deserialization fails.
    """
    if not serialized_bytes:
        return None
        
    buffer = io.BytesIO(serialized_bytes)
    buffer.seek(0)
    
    try:
        # allow_pickle=True is needed for 'node_id_to_int_idx_pickled' and 'original_node_ids_array' (dtype=object)
        loaded_npz = np.load(buffer, allow_pickle=True) 
        # Create a mutable dictionary from NpzFile
        arrays_dict = {key: loaded_npz[key] for key in loaded_npz.files}
        loaded_npz.close()
        
        return deserialize_tree_from_arrays(arrays_dict)
    except Exception:
        # print(f"Error during tree deserialization: {e}") # Handled by UDF logger
        return None
