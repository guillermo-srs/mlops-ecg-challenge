# File: src/reference_selection.py

from typing import List, Tuple, Optional, Set
import numpy as np
import pandas as pd

# Use the generalized similarity calculation from similarity_metrics
from .similarity_metrics import get_batch_similarity_function, BatchSimilarityFunctionMatrixToVector


def _calculate_center_vector(
    feature_vectors: np.ndarray,
    use_median_for_center: bool
) -> np.ndarray:
    """
    Calculates the mean or median vector from a matrix of feature vectors.
    """
    if use_median_for_center:
        return np.median(feature_vectors, axis=0)
    else:
        return np.mean(feature_vectors, axis=0)


def _find_initial_reference_vector(
    all_feature_vectors: np.ndarray,
    center_vector: np.ndarray,
    similarity_metric_name: str # Added: to use the specified metric
) -> Tuple[int, np.ndarray]:
    """
    Finds the vector in all_feature_vectors most similar (max score by the chosen metric)
    to the center_vector.
    Returns its original index and the vector itself.
    """
    calculate_similarity: BatchSimilarityFunctionMatrixToVector = \
        get_batch_similarity_function(similarity_metric_name, mode="matrix_to_vector")

    # Precompute norm of center_vector if metric is cosine
    center_vector_norm = np.linalg.norm(center_vector) if similarity_metric_name == "cosine" else None

    similarities_to_center = calculate_similarity(
        all_feature_vectors,
        center_vector,
        center_vector_norm # Pass norm if needed by the metric
    )
    
    initial_ref_original_idx = np.nanargmax(similarities_to_center) # Higher score is better
    return initial_ref_original_idx, all_feature_vectors[initial_ref_original_idx].copy()


def _find_next_dissimilar_reference(
    candidate_vectors: np.ndarray,
    candidate_original_indices: np.ndarray,
    last_selected_reference: np.ndarray,
    similarity_metric_name: str # Added: to use the specified metric
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Finds the vector in candidate_vectors most dissimilar (min score by the chosen metric)
    to the last_selected_reference.
    Returns its original index and the vector, or (None, None) if no valid candidate.
    """
    if candidate_vectors.shape[0] == 0:
        return None, None

    calculate_similarity: BatchSimilarityFunctionMatrixToVector = \
        get_batch_similarity_function(similarity_metric_name, mode="matrix_to_vector")

    # Precompute norm of last_selected_reference if metric is cosine
    last_ref_norm = np.linalg.norm(last_selected_reference) if similarity_metric_name == "cosine" else None

    similarities_to_last_ref = calculate_similarity(
        candidate_vectors,
        last_selected_reference,
        last_ref_norm # Pass norm if needed
    )

    # Since all metrics are defined such that higher is more similar,
    # for dissimilarity, we want the one with the minimum similarity score.
    try:
        idx_in_candidates = np.nanargmin(similarities_to_last_ref)
    except ValueError: # All similarities were NaN
        return None, None

    chosen_vector_original_idx = candidate_original_indices[idx_in_candidates]
    chosen_vector = candidate_vectors[idx_in_candidates].copy()

    return chosen_vector_original_idx, chosen_vector


def select_diverse_reference_vectors(
    sample_dataframe: pd.DataFrame,
    feature_column_names: List[str],
    num_references_to_select: int,
    similarity_metric_name: str, # Added: to specify which metric to use
    use_median_for_center: bool = False
) -> np.ndarray:
    """
    Selects N diverse reference vectors from a sample DataFrame using the specified similarity metric.
    1. Calculates a center vector (mean/median) of the sample.
    2. Selects the first reference as the sample vector most similar (by the metric) to the center.
    3. Iteratively selects subsequent references by choosing the sample vector that is
       most dissimilar (lowest score by the metric) to the *previously selected* reference.
    Returns a NumPy array of shape (N_selected, K_features).
    """
    if sample_dataframe.empty or not feature_column_names or num_references_to_select <= 0:
        num_features = len(feature_column_names) if feature_column_names else 0
        return np.array([], dtype=float).reshape(0, num_features)

    feature_vectors_from_sample = sample_dataframe[feature_column_names].to_numpy(dtype=float)
    num_samples_in_pool, num_features = feature_vectors_from_sample.shape

    if num_samples_in_pool == 0:
        return np.array([], dtype=float).reshape(0, num_features)

    actual_num_to_select = min(num_references_to_select, num_samples_in_pool)
    if actual_num_to_select == 0:
        return np.array([], dtype=float).reshape(0, num_features)

    selected_references_list: List[np.ndarray] = []
    selected_original_indices_set: Set[int] = set()

    center_vector = _calculate_center_vector(feature_vectors_from_sample, use_median_for_center)

    ref_1_original_idx, ref_1_vector = _find_initial_reference_vector(
        feature_vectors_from_sample, center_vector, similarity_metric_name
    )
    selected_references_list.append(ref_1_vector)
    selected_original_indices_set.add(ref_1_original_idx)

    for _ in range(1, actual_num_to_select):
        if len(selected_original_indices_set) >= num_samples_in_pool:
            break

        last_added_reference = selected_references_list[-1]

        candidate_mask = np.ones(num_samples_in_pool, dtype=bool)
        candidate_mask[list(selected_original_indices_set)] = False

        current_candidate_original_indices = np.where(candidate_mask)[0]
        if len(current_candidate_original_indices) == 0:
            break

        current_candidate_vectors = feature_vectors_from_sample[current_candidate_original_indices]

        next_ref_original_idx, next_ref_vector = _find_next_dissimilar_reference(
            current_candidate_vectors,
            current_candidate_original_indices,
            last_added_reference,
            similarity_metric_name
        )

        if next_ref_vector is not None and next_ref_original_idx is not None:
            selected_references_list.append(next_ref_vector)
            selected_original_indices_set.add(next_ref_original_idx)
        else:
            print(f"Warning: Could not find a valid next dissimilar reference vector using {similarity_metric_name}. Stopping selection.")
            break
    
    if not selected_references_list:
        return np.array([], dtype=float).reshape(0, num_features)

    return np.array(selected_references_list)
