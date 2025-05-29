# File: src/similarity_metrics.py

from typing import Callable, Optional, Dict
import numpy as np

# Type alias for a similarity function
# Takes two NumPy arrays (vector_a, vector_b) and an optional precomputed norm for vector_b
# Returns a single float score (higher is more similar)
SimilarityFunction = Callable[[np.ndarray, np.ndarray, Optional[float]], float]

# Type alias for a batch similarity function
# Takes a batch matrix (M, K) and a single vector (K,) or a reference matrix (N, K)
# Returns an array of scores.
# For matrix_to_vector: (M,)
# For batch_to_matrix: (M, N)
BatchSimilarityFunctionMatrixToVector = Callable[[np.ndarray, np.ndarray, Optional[float]], np.ndarray]
BatchSimilarityFunctionBatchToMatrix = Callable[[np.ndarray, np.ndarray], np.ndarray]


# --- Individual Similarity/Distance Functions (vector vs vector) ---

def cosine_similarity_vector(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    norm_b: Optional[float] = None # norm_a is calculated internally
) -> float:
    """
    Calculates cosine similarity between two 1D vectors. Higher is more similar.
    Range: [-1, 1]
    """
    if vector_a.size == 0 or vector_b.size == 0:
        return 0.0 # Or handle as an error/NaN depending on desired behavior

    norm_a = np.linalg.norm(vector_a)
    if norm_b is None:
        norm_b = np.linalg.norm(vector_b)

    epsilon = 1e-9
    if norm_a < epsilon or norm_b < epsilon:
        # If one vector is zero and the other is not, similarity is 0.
        # If both are zero, conventionally, cosine can be 1 (identical) or undefined.
        # Spark's CosineDistance returns 0 if one vector is all zeros.
        # We'll return 0.0 if either is zero. If both are zero, could be 1.0 if sizes match.
        if norm_a < epsilon and norm_b < epsilon and vector_a.size == vector_b.size:
            return 1.0 # Both zero vectors of same dimension are perfectly similar
        return 0.0

    dot_product = np.dot(vector_a, vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return np.nan_to_num(similarity, nan=0.0)


def euclidean_similarity_vector(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    _norm_b: Optional[float] = None # norm_b not used for Euclidean, but kept for consistent signature
) -> float:
    """
    Calculates Euclidean distance and converts it to a similarity score.
    Similarity = 1 / (1 + distance). Higher is more similar.
    Range: (0, 1] (1 if distance is 0)
    """
    if vector_a.size == 0 or vector_b.size == 0:
        return 0.0 # No similarity if vectors are empty
    
    distance = np.linalg.norm(vector_a - vector_b)
    return 1.0 / (1.0 + distance)


def manhattan_similarity_vector(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    _norm_b: Optional[float] = None # norm_b not used for Manhattan
) -> float:
    """
    Calculates Manhattan distance and converts it to a similarity score.
    Similarity = 1 / (1 + distance). Higher is more similar.
    Range: (0, 1] (1 if distance is 0)
    """
    if vector_a.size == 0 or vector_b.size == 0:
        return 0.0

    distance = np.sum(np.abs(vector_a - vector_b))
    return 1.0 / (1.0 + distance)

def dot_product_similarity_vector(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    _norm_b: Optional[float] = None # norm_b not used for simple dot product
) -> float:
    """
    Calculates dot product as a similarity score. Not normalized.
    Higher is more similar (assuming positive feature values or meaningful interpretation).
    Range: Depends on vector magnitudes.
    """
    if vector_a.size == 0 or vector_b.size == 0:
        return 0.0
    return np.dot(vector_a, vector_b)


# --- Batch Similarity/Distance Functions ---
# These are the versions that operate on a batch_matrix_a vs. a single vector_b
# or batch_matrix_a vs. reference_matrix_b

# Replicates the original cosine similarity from processing_logic.py
def cosine_similarity_matrix_to_vector(
    batch_matrix_a: np.ndarray, # (M, K)
    vector_b: np.ndarray,       # (K,)
    vector_b_norm: Optional[float] = None
) -> np.ndarray:                # (M,)
    """
    Calculates cosine similarity for each vector in batch_matrix_a (MxK)
    against a single vector_b (K,). Returns a 1D NumPy array (M,) of scores.
    """
    if batch_matrix_a.size == 0 or vector_b.size == 0:
        return np.array([], dtype=float) if batch_matrix_a.ndim == 2 and batch_matrix_a.shape[0] > 0 else np.array([0.0]*batch_matrix_a.shape[0], dtype=float)


    epsilon = 1e-9
    if vector_b_norm is None:
        vector_b_norm = np.linalg.norm(vector_b)

    if vector_b_norm < epsilon:
        return np.zeros(batch_matrix_a.shape[0], dtype=float)

    dot_products = np.dot(batch_matrix_a, vector_b) # (M,)
    batch_matrix_norms = np.linalg.norm(batch_matrix_a, axis=1) # (M,)
    batch_matrix_norms[batch_matrix_norms < epsilon] = epsilon # Avoid division by zero for rows in A

    similarity_scores = dot_products / (batch_matrix_norms * vector_b_norm)
    return np.nan_to_num(similarity_scores, nan=0.0)


def cosine_similarity_batch_vs_matrix(
    batch_matrix_a: np.ndarray,      # (N_batch, K_features)
    reference_matrix_b: np.ndarray   # (N_references, K_features)
) -> np.ndarray:                     # Returns (N_batch, N_references)
    """
    Calculates cosine similarity for each vector in batch_matrix_a against
    each vector in reference_matrix_b.
    """
    if batch_matrix_a.size == 0 or reference_matrix_b.size == 0:
        return np.array([], dtype=float).reshape(batch_matrix_a.shape[0], reference_matrix_b.shape[0])

    epsilon = 1e-9
    norm_a = np.linalg.norm(batch_matrix_a, axis=1, keepdims=True)
    norm_a[norm_a < epsilon] = epsilon
    normalized_a = batch_matrix_a / norm_a

    norm_b = np.linalg.norm(reference_matrix_b, axis=1, keepdims=True)
    norm_b[norm_b < epsilon] = epsilon
    normalized_b = reference_matrix_b / norm_b

    similarity_matrix = np.dot(normalized_a, normalized_b.T) # (N_batch, N_references)
    return np.nan_to_num(similarity_matrix, nan=0.0)


def euclidean_similarity_matrix_to_vector(
    batch_matrix_a: np.ndarray, # (M, K)
    vector_b: np.ndarray,       # (K,)
    _vector_b_norm: Optional[float] = None # Not used
) -> np.ndarray:                # (M,)
    """
    Calculates Euclidean distance for each vector in batch_matrix_a to vector_b,
    then converts to similarity: 1 / (1 + distance).
    """
    if batch_matrix_a.size == 0 or vector_b.size == 0:
        return np.array([], dtype=float) if batch_matrix_a.ndim == 2 and batch_matrix_a.shape[0] > 0 else np.array([0.0]*batch_matrix_a.shape[0], dtype=float)

    # (A - B) for each row A_i in A
    differences = batch_matrix_a - vector_b # Broadcasting: (M,K) - (K,) -> (M,K)
    distances = np.linalg.norm(differences, axis=1) # (M,)
    return 1.0 / (1.0 + distances)


def euclidean_similarity_batch_vs_matrix(
    batch_matrix_a: np.ndarray,      # (N_batch, K)
    reference_matrix_b: np.ndarray   # (N_references, K)
) -> np.ndarray:                     # (N_batch, N_references)
    """
    For each vector in batch_matrix_a, calculates Euclidean similarity to each vector
    in reference_matrix_b.
    This is more complex with NumPy broadcasting for M-vs-N.
    A common way is to compute squared Euclidean distance:
    d(a,b)^2 = ||a||^2 + ||b||^2 - 2*a.b
    Then convert to similarity.
    """
    if batch_matrix_a.size == 0 or reference_matrix_b.size == 0:
        return np.array([], dtype=float).reshape(batch_matrix_a.shape[0], reference_matrix_b.shape[0])

    # norms_a_sq: (N_batch, 1)
    norms_a_sq = np.sum(np.square(batch_matrix_a), axis=1, keepdims=True)
    # norms_b_sq: (1, N_references)
    norms_b_sq = np.sum(np.square(reference_matrix_b), axis=1, keepdims=True).T
    
    # dot_product_ab: (N_batch, N_references)
    dot_product_ab = np.dot(batch_matrix_a, reference_matrix_b.T)
    
    # distances_sq: (N_batch, N_references)
    # norms_a_sq + norms_b_sq results in (N_batch, N_references) due to broadcasting
    distances_sq = norms_a_sq + norms_b_sq - 2 * dot_product_ab
    
    # Clip negative values that might occur due to precision errors for zero distances
    distances_sq[distances_sq < 0] = 0
    distances = np.sqrt(distances_sq)
    
    return 1.0 / (1.0 + distances)


def manhattan_similarity_matrix_to_vector(
    batch_matrix_a: np.ndarray, # (M, K)
    vector_b: np.ndarray,       # (K,)
    _vector_b_norm: Optional[float] = None # Not used
) -> np.ndarray:                # (M,)
    """
    Calculates Manhattan distance for each vector in batch_matrix_a to vector_b,
    then converts to similarity: 1 / (1 + distance).
    """
    if batch_matrix_a.size == 0 or vector_b.size == 0:
        return np.array([], dtype=float) if batch_matrix_a.ndim == 2 and batch_matrix_a.shape[0] > 0 else np.array([0.0]*batch_matrix_a.shape[0], dtype=float)

    differences = batch_matrix_a - vector_b # Broadcasting
    distances = np.sum(np.abs(differences), axis=1) # (M,)
    return 1.0 / (1.0 + distances)


def manhattan_similarity_batch_vs_matrix(
    batch_matrix_a: np.ndarray,      # (N_batch, K)
    reference_matrix_b: np.ndarray   # (N_references, K)
) -> np.ndarray:                     # (N_batch, N_references)
    """
    For each vector in batch_matrix_a, calculates Manhattan similarity to each vector
    in reference_matrix_b. This requires iterating or a more complex reshape.
    For simplicity, we can iterate here, but for large N_references, this is slow.
    A more optimized version would use broadcasting tricks (e.g., with an intermediate dimension).
    """
    n_batch = batch_matrix_a.shape[0]
    n_references = reference_matrix_b.shape[0]
    
    if n_batch == 0 or n_references == 0:
        return np.array([], dtype=float).reshape(n_batch, n_references)

    # Efficient computation of Manhattan distances between two sets of vectors
    # X (n_batch, K), Y (n_references, K)
    # Result (n_batch, n_references)
    # This can be done by expanding dims and broadcasting:
    # X_expanded = X[:, np.newaxis, :]  (n_batch, 1, K)
    # Y_expanded = Y[np.newaxis, :, :]  (1, n_references, K)
    # diff = X_expanded - Y_expanded     (n_batch, n_references, K)
    # manhattan_distances = np.sum(np.abs(diff), axis=2)
    
    X_expanded = batch_matrix_a[:, np.newaxis, :]
    # Y is already (n_references, K), suitable for broadcasting against X_expanded's last two dims
    # if we treat Y as (1, n_references, K)
    
    distances = np.sum(np.abs(X_expanded - reference_matrix_b[np.newaxis, :, :]), axis=2)
    return 1.0 / (1.0 + distances)


def dot_product_matrix_to_vector(
    batch_matrix_a: np.ndarray, # (M, K)
    vector_b: np.ndarray,       # (K,)
    _vector_b_norm: Optional[float] = None # Not used
) -> np.ndarray:                # (M,)
    if batch_matrix_a.size == 0 or vector_b.size == 0:
        return np.array([], dtype=float) if batch_matrix_a.ndim == 2 and batch_matrix_a.shape[0] > 0 else np.array([0.0]*batch_matrix_a.shape[0], dtype=float)
    return np.dot(batch_matrix_a, vector_b)

def dot_product_batch_vs_matrix(
    batch_matrix_a: np.ndarray,      # (N_batch, K)
    reference_matrix_b: np.ndarray   # (N_references, K)
) -> np.ndarray:                     # (N_batch, N_references)
    if batch_matrix_a.size == 0 or reference_matrix_b.size == 0:
        return np.array([], dtype=float).reshape(batch_matrix_a.shape[0], reference_matrix_b.shape[0])
    return np.dot(batch_matrix_a, reference_matrix_b.T)


# --- Factory for Batch Similarity Functions ---
BATCH_SIMILARITY_FUNCTIONS_MATRIX_TO_VECTOR: Dict[str, BatchSimilarityFunctionMatrixToVector] = {
    "cosine": cosine_similarity_matrix_to_vector,
    "euclidean_sim": euclidean_similarity_matrix_to_vector,
    "manhattan_sim": manhattan_similarity_matrix_to_vector,
    "dot_product": dot_product_matrix_to_vector,
}

BATCH_SIMILARITY_FUNCTIONS_BATCH_TO_MATRIX: Dict[str, BatchSimilarityFunctionBatchToMatrix] = {
    "cosine": cosine_similarity_batch_vs_matrix,
    "euclidean_sim": euclidean_similarity_batch_vs_matrix,
    "manhattan_sim": manhattan_similarity_batch_vs_matrix,
    "dot_product": dot_product_batch_vs_matrix,
}

def get_batch_similarity_function(
    metric_name: str,
    mode: str = "matrix_to_vector" # "matrix_to_vector" or "batch_to_matrix"
) -> Callable:
    """
    Returns the requested batch similarity calculation function.
    All returned functions are designed so that higher scores mean more similar.
    """
    if mode == "matrix_to_vector":
        func = BATCH_SIMILARITY_FUNCTIONS_MATRIX_TO_VECTOR.get(metric_name.lower())
    elif mode == "batch_to_matrix":
        func = BATCH_SIMILARITY_FUNCTIONS_BATCH_TO_MATRIX.get(metric_name.lower())
    else:
        raise ValueError(f"Unknown similarity function mode: {mode}")

    if func is None:
        raise ValueError(f"Unknown similarity metric: {metric_name} for mode {mode}. "
                         f"Available for matrix_to_vector: {list(BATCH_SIMILARITY_FUNCTIONS_MATRIX_TO_VECTOR.keys())}. "
                         f"Available for batch_to_matrix: {list(BATCH_SIMILARITY_FUNCTIONS_BATCH_TO_MATRIX.keys())}.")
    return func
