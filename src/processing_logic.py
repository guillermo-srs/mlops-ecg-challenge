# File: src/processing_logic.py

from typing import Iterator, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import heapq

# Import the factory function and type alias from the new similarity_metrics module
from .similarity_metrics import get_batch_similarity_function, BatchSimilarityFunctionMatrixToVector, BatchSimilarityFunctionBatchToMatrix


# The original cosine-specific functions are now removed from here,
# as their logic is in similarity_metrics.py and accessed via get_batch_similarity_function.

def transform_to_similarity_space_udf_logic(
    iterator_of_pandas_dfs: Iterator[pd.DataFrame],
    reference_vectors: np.ndarray, # (N_references, K_original_features)
    feature_column_names: List[str],
    id_column_name: str,
    similarity_metric_name: str # New parameter: name of the similarity metric
) -> Iterator[pd.DataFrame]:
    """
    Pandas UDF logic to transform data into an N-dimensional similarity space.
    Each dimension in the new space is the similarity (using the specified metric)
    to one of N reference vectors.
    """
    num_references = reference_vectors.shape[0]
    
    # Get the appropriate batch_vs_matrix similarity function
    calculate_batch_similarity_to_references: BatchSimilarityFunctionBatchToMatrix = \
        get_batch_similarity_function(similarity_metric_name, mode="batch_to_matrix")


    if num_references == 0: # No transformation to perform
        for pdf_batch in iterator_of_pandas_dfs:
            if not pdf_batch.empty:
                yield pdf_batch[[id_column_name]].copy() # Return only IDs
        return

    for pdf_batch in iterator_of_pandas_dfs:
        if pdf_batch.empty:
            continue

        batch_ids_series = pdf_batch[id_column_name]
        batch_feature_matrix = pdf_batch[feature_column_names].to_numpy(dtype=float)

        if batch_feature_matrix.ndim == 1 and len(feature_column_names) == 1:
            batch_feature_matrix = batch_feature_matrix.reshape(-1, 1)

        if batch_feature_matrix.size == 0: # No features in batch
            # All similarities will be 0 or a default low value if metric implies
            # For simplicity, using zeros.
            similarity_scores_matrix = np.zeros((len(batch_ids_series), num_references))
        else:
            similarity_scores_matrix = calculate_batch_similarity_to_references(
                batch_feature_matrix,
                reference_vectors
            )

        output_data = {id_column_name: batch_ids_series}
        for i in range(num_references):
            # Column names indicate the metric used for clarity, though UDF output schema is fixed
            output_data[f"sim_to_ref_{i}"] = similarity_scores_matrix[:, i]

        yield pd.DataFrame(output_data)


def search_in_space_udf_logic(
    iterator_of_pandas_dfs: Iterator[pd.DataFrame],
    query_vector: np.ndarray, # Query vector (K-dim or N-dim depending on search space)
    feature_column_names: List[str], # Names of columns in the current search space
    id_column_name: str,
    top_x_per_partition: int,
    similarity_metric_name: str # New parameter: name of the similarity metric
) -> Iterator[pd.DataFrame]:
    """
    Pandas UDF logic to search for top_x similar items in a given feature space
    using the specified similarity metric.
    Uses a min-heap to keep track of the top X items per partition.
    """
    if query_vector.size == 0 and len(feature_column_names) > 0 :
        # If query is empty but features are expected, cannot meaningfully search.
        # Depending on metric, could return all items with a default score, or nothing.
        # Returning nothing is safer.
        return

    # Get the appropriate matrix_to_vector similarity function
    calculate_similarity_to_query: BatchSimilarityFunctionMatrixToVector = \
        get_batch_similarity_function(similarity_metric_name, mode="matrix_to_vector")

    query_vector_norm = np.linalg.norm(query_vector) # Precompute for metrics that might use it
    epsilon = 1e-9
    if query_vector_norm < epsilon and query_vector.size > 0: # Handle zero query vector
        # If query vector is zero, specific metrics handle this (e.g. cosine returns 0).
        # For distance based, similarity would be 1 to other zero vectors.
        # The metric functions themselves should handle zero query vectors appropriately.
        pass


    local_top_x_heap = []
    tie_breaker_counter = 0

    for pdf_batch in iterator_of_pandas_dfs:
        if pdf_batch.empty:
            continue

        batch_ids = pdf_batch[id_column_name].tolist()
        
        if not feature_column_names: # If no feature columns, search is not meaningful in this context
            # Create default zero scores if we must yield something
            # For now, assume features exist if this UDF is called for search.
            # If num_cols = 0 for the dataset, then this UDF should not produce results.
            similarity_scores = np.array([0.0] * len(batch_ids)) # Default score
        else:
            batch_feature_matrix = pdf_batch[feature_column_names].to_numpy(dtype=float)
            if batch_feature_matrix.ndim == 1 and len(feature_column_names) == 1:
                batch_feature_matrix = batch_feature_matrix.reshape(-1, 1)

            if batch_feature_matrix.size == 0:
                 similarity_scores = np.array([])
            else:
                # Pass the pre-calculated norm if the metric function uses it (like cosine)
                similarity_scores = calculate_similarity_to_query(
                    batch_feature_matrix,
                    query_vector,
                    query_vector_norm if similarity_metric_name == "cosine" else None
                )

        for i in range(len(similarity_scores)):
            score = similarity_scores[i]
            row_id = batch_ids[i]

            if not np.isnan(score): # Ensure score is valid before adding to heap
                tie_breaker_counter += 1
                item = (score, tie_breaker_counter, row_id)

                if len(local_top_x_heap) < top_x_per_partition:
                    heapq.heappush(local_top_x_heap, item)
                elif score > local_top_x_heap[0][0]:
                    heapq.heapreplace(local_top_x_heap, item)

    if local_top_x_heap:
        result_ids = [item[2] for item in local_top_x_heap]
        result_scores = [item[0] for item in local_top_x_heap]

        output_pdf = pd.DataFrame({
            id_column_name: result_ids,
            "similarity_score": result_scores
        })
        yield output_pdf
