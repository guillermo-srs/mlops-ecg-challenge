# MLOps Engineer Challenge: ECG Similarity Search Engine

This project implements a system for performing similarity-based analysis on (synthetic) ECG-like data samples using PySpark. It explores different strategies to balance search accuracy, scalability, and query speed, addressing challenges common in high-dimensional data and large datasets.

## Project Goal

The primary goal is to develop and evaluate different methods for efficiently finding similar ECG-like samples within a large dataset. The system is designed to be scalable and allows for configurable similarity definitions. While the original problem specified ECG data, this implementation uses synthetically generated high-dimensional vector data.

---

## Original Specifications & Challenges

### Initial Specifications

*(As provided in the original challenge description)*

* The use case database is composed of 10-second intervals of ECGs (no sampling rate given).
* The goal is to efficiently produce a similarity-based analysis over ECGs samples.
* Non-functional requirements: scalability and high speed (~1 second answer to queries regardless of size, though this is a target for a production system and this project explores trade-offs). No “random access” specification.
* The solution should be able to handle $\sim 10^9$ elements (this project tests with configurable data sizes).
* Multiple methods will be used to define “similarity” measures between observations.

### Base Issues Addressed

*(Adapted from the original challenge description)*

1.  **Similarity matrix initialization cost:** Given that the problem is similarity analysis with no “given optimizations”, the magnitude of operations (at least in the initialization) will be of order $\mathcal{O}(N^2)$. Given that later on these measures are defined as *metrics*, the magnitude will be reduced to $\frac{N(N+1)}{2}$, which is still exponential.
2.  **Search costs:** This element will be utterly dependent on the solution implemented to solve the search. However, the query's answer time must be limited to ~1 second, regardless of the number of elements to compare to.
3.  **Sample and similarities storage:** Each observation (ECG sample) will have 10 seconds at a 500Hz sampling rate. Which implies a $5 \cdot 10^3$ feature vector per sample. The system should handle at least $10^9$ samples. Compute a whole similarity matrix will imply a matrix size of at least $\sim 10^{17}$. Which, assuming a 32-bit float (4 bytes), would imply ~2 Zetabytes[^1]. Not a reasonable amount of numbers to store :) 
4.  **Data access:** When a new pair of “*sample, metric*” arrives, the system should provide a number of similar ECG samples to the new sample, following a given metric.

[^1]: This calculus is done assuming a proper similarity metric, where we can discard half of the values due to symmetry.

## System Design & Technical Approach

To define the technical solution, a few caveats must first be considered, as this is an interview challenge and not an actual product:

- The chosen libraries and frameworks do not necessarily reflect the best fit for the current problem, as the selection process is outside the scope of this test.
- The available hardware resource is a remote machine equipped with an Intel i5-13500 processor, 64 GB of RAM, and a 500GB NVMe SSD for storing the generated data. Consequently, the time constraint of the test dictates the final solution design, as it would otherwise be impossible to accurately "guess" the solution's performance post-deployment.
- The solution's design is not always strictly followed in the implementation, as the number of methods, optimizations, and experiments required for optimal results are beyond the scope of this test. Instead, while the desired "approach" will be defined, it will also be simplified for the final implementation.

Given the described hardware limitations, certain adjustments have been made to the problem definition:

- The sampling rate has been reduced from 500Hz to 250Hz, because 5000 columns with $10^9$ samples would make additional optimization critical for system viability, and half the sampling rate is an acceptable trade off.
- To achieve reasonable query answer times, two non-exhaustive search have been developed. These searches will implement a heuristic approach to find similar values without either the need to scan the entire dataset or analyze it with fewer dimensions.
- To measure query answer times, an exhaustive search has also been implemented.
- Furthermore, ECG datasets with specific, pre-defined characteristics or induced biases were not synthetically generated for targeted performance evaluation. This decision was made to prioritize overall system viability and processing efficiency.

For the optimized search, two different approaches have been implemented:

- The first selects a subset of samples to use as 'references,' against which similarity measurements are performed, thereby minimizing the number of columns to improve search times.
- The second implements a recursive Hierarchical KMeans tree based on the subset of references from the first approach. The idea is to leverage the improved performance gained from the change in features to develop a structure that "maps" the space of differences among the various observations.
- The accuracy of both approaches cannot be guaranteed to a certain degree, as the scope of this test is limited.

Thus, while the absolute approach ensures exact results through exhaustive and precise comparisons (albeit being the most computationally expensive), the `hierarchical_kmeans` approach offers a balance, increasing speed by using approximate similarity heuristics within its tree-guided search and potentially not evaluating every candidate. The `optimized_transform` method, in turn, pushes further for speed by relying on comparisons made within a significantly transformed, lower-dimensional representation of the data.

### Core Technology

* **PySpark**: Used for distributed data processing and managing large datasets. It allows for **scalable** computations across a cluster (though this project is primarily run in local mode for development).
* **Pandas UDFs (User Defined Functions)**: Applied for per-partition processing in Spark, allowing the use of Python libraries like Pandas, NumPy, and Scikit-learn on subsets of data within Spark tasks.
* **Scikit-learn**: Utilized for K-Means clustering (specifically `MiniBatchKMeans` by default, or `KMeans`) within the hierarchical K-Means indexing strategy.
* **NumPy & Pandas**: For efficient numerical computations and data manipulation within UDFs.
* **Parquet**: Chosen as the storage format for the synthetic vector data due to its efficiency and columnar storage benefits.

### Implemented Search Modes

The application supports three distinct search modes, selectable via a command-line argument:

1.  **`absolute` (Exhaustive Search)**:
    * Performs a brute-force search.
    * Calculates the similarity between the query vector and every vector in the original dataset.
    * Provides exact results but is the most computationally expensive, serving as a baseline.
2.  **`optimized_transform` (N-Dimensional Similarity Space Search)**:
    * Transforms the dataset into an reduced dimensional space where each dimension represents the similarity of an original data point to one of N reference vectors.
    * The *furthest* observations, using the similarity measure as base, are chosen as references
    * Aims to improve speed by searching in a potentially much lower-dimensional space, at the cost of accuracy.
    * This optimization alters similarity scores; consequently, they may not be directly comparable to those from the other two modes
3.  **`hierarchical_kmeans` (Hierarchical K-Means Indexing Search)**:
    * Details for this mode are expanded in the [Hierarchical K-Means Indexing Details](#hierarchical-k-means-indexing-details) section.
    * Aims for faster queries by narrowing down the search space via tree traversal, but the search is approximate.

### Similarity Metrics

The system supports multiple similarity metrics, configurable via the `--similarity-metric` command-line argument. All metrics are defined such that a higher score indicates greater similarity. Distance metrics are converted to similarity scores (e.g., `1 / (1 + distance)`).
Implemented metrics include:

* `cosine`: Cosine similarity.
* `euclidean_sim`: Euclidean distance converted to similarity.
* `manhattan_sim`: Manhattan distance converted to similarity.
* `dot_product`: Dot product (unnormalized).

These are implemented in `src/similarity_metrics.py` and used by the core processing logic.

### Hierarchical K-Means Indexing Details

Given the nature and time constraints of this challenge, the current implementation has not undergone the full cycle of experimentation and refinement that would be required for a production solution. 

* **Distributed Tree Building**: The strategy is based on constructing a hierarchical K-Means tree index for each partition of the K-dimensional Spark DataFrame.

  **Local Tree Construction Logic**:

  - For clustering at each level of the tree, Scikit-learn's `MiniBatchKMeans` was used 
  - **Representative Points for Memory Optimization:** A fundamental design decision is that, instead of storing the calculated K-Means centroid vectors (which can be memory-intensive), each internal tree node stores an index to an actual data point from the original dataset partition. This representative point is chosen as the data point within the cluster that is most similar (according to the selected `--similarity-metric`) to the calculated centroid for that cluster. This choice aims to significantly reduce the memory footprint of each tree node. At each depth level, multiple branches are defined, which implies many centroids would typically be needed to define search paths (if not for this optimization). Leaf nodes store indices to their constituent data points from the original partition..

  **Compact Tree Serialization**: The tree constructed for each partition (represented as a `NodeMapType`: a dictionary of `TreeNode` objects) is serialized into a compact byte format using a custom serializer in `src/tree_serializer.py`. This serializer flattens the tree structure into NumPy arrays and uses `numpy.savez_compressed`, aiming for efficient storage and transport and seeking to be more memory-efficient than direct `pickle` on complex objects with many NumPy arrays. These bytes are then collected by the Spark driver.

  **Tree Persistence & Broadcast**: The driver deserializes these bytes to reconstruct the `NodeMapType` objects. The collection of these objects (one per original partition) is then pickled for optional persistence to disk and subsequently broadcast to executors for the search phase.

  **Approximate Search Mechanism**: To find similar items, the K-dimensional query vector traverses these trees. Both the tree traversal (query vs. representative point comparison) and the final search within the identified leaf node use the user-selected similarity metric.

### Key Design Considerations & Trade-offs

* **Scalability vs. Exactness**: The `absolute` search is exact but scales poorly. While `hierarchical_kmeans` could trade some exactness for better scalability and speed, `optimized_transform` produces a huge increase in speed, but the accuracy should be take with a grain of salt
* **Memory Management**:
  * High-dimensional data (e.g., 2500 columns) poses significant memory challenges.
  * Strategies to manage this include:
    * Using `MiniBatchKMeans` (default) for reduced memory footprint during tree building in the `hierarchical_kmeans` mode.
    * Storing representative point *indices* instead of full centroid vectors in tree nodes for the `hierarchical_kmeans` mode.
    * Implementing a custom tree serializer (`src/tree_serializer.py`) using `numpy.savez_compressed` to reduce the size of serialized trees sent to the driver, addressing `spark.driver.maxResultSize` issues.
    * Careful tuning of K-Means tree parameters (`MAX_DEPTH`, `K_PER_LEVEL`, `MIN_LEAF_SIZE`) to control tree size and complexity. Drastically reducing `MAX_DEPTH` from initial high values is crucial.
    * Adjusting Spark partitions (`--num-partitions`) to control the data size processed by each individual K-Means task. This is extremely relevant for RAM management.
    * Configuring Spark's temporary directory (`spark.local.dir`) via `--spark-local-dir` for shuffle spills and intermediate disk usage.
* **Parameter Tuning**: The performance and accuracy of `optimized_transform` (choice of N) and `hierarchical_kmeans` (tree parameters, choice of representative point metric) are highly dependent on parameter tuning.
* **Representative Point Selection**: In `hierarchical_kmeans`, using a representative point (actual data point closest to a K-Means centroid) instead of the K-Means centroid itself is a trade-off between memory efficiency and potentially a slight deviation from the "true" cluster center.

---

## Project Structure

```
.
├── Makefile                # For managing common tasks like setup, running, cleaning
├── main.py                 # Main entry point for the application, orchestrates workflows
├── requirements.txt        # Python dependencies
├── conf/                   # Optional: For Spark configuration files
│   └── log4j.properties    # Example: To control Spark's Java-level logging
├── spark-events/           # Default output directory for Spark event logs (for History Server)
├── generated_data_make_* # Default output directories for Parquet data & tree files from Makefile runs
├── src/                    # Source code modules
│   ├── init.py
│   ├── data_handling.py    # Data generation, loading
│   ├── distributed_tree_orchestrator.py # Orchestrates distributed K-Means tree building/searching
│   ├── hierarchical_kmeans_indexer.py   # Logic for local K-Means tree building & searching
│   ├── processing_logic.py # Core UDF logic for data transformation and search
│   ├── reference_selection.py # Logic for selecting diverse reference vectors
│   ├── reporting.py        # Utility for printing results
│   ├── similarity_metrics.py # Implementation of various similarity/distance functions
│   ├── tree_serializer.py  # Custom serialization for tree structures
│   └── spark_utils.py      # SparkSession initialization, RDD utilities
└── README.md               # This file

```


---

## Setup and Installation

### Prerequisites

* Python (3.8+ recommended)
* Apache Spark (e.g., 3.x.x, accessible via `spark-submit` or a PySpark installation from Anaconda/pip)
* Java (8/11/17, depending on Spark version)
* Anaconda/Miniconda (recommended for managing Python environments)

### Environment Setup

Th `Makefile` provides a setup.

```bash
make setup
```

This will also guide you on creating a sample `conf/log4j.properties` if you wish to customize Spark's Java-level logging.

In case `make` is not available:

1. #### **Create and Activate Python Environment**:

   Using Conda:

   ```bash
   conda create -n ecg_sim_env python=3.10
   conda activate ecg_sim_env
   ```

   Or using `venv`:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   A `requirements.txt` file should be present in the project root.

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

### Using the Makefile

The `Makefile` is the recommended way to run the application with different configurations.

```bash
# View available targets and common override variables
make help

# Example: Run hierarchical_kmeans search with specific parameters
make run-hkmeans NUM_DATA_ROWS=100000 NUM_DATA_COLS=100 SPARK_CORES=4 \
                 SIMILARITY_METRIC=euclidean_sim PYTHON_LOG_LEVEL=DEBUG MAX_DEPTH=3

# Example: Run all modes, forcing data regeneration
make run-all-regenerate
````

The Makefile handles setting parameters and constructs the spark-submit command. Data and tree files generated by Makefile runs are stored in directories dynamically named based on the mode and similarity metric (e.g., generated_data_make_hkm_cosine).

### Key Command-Line Arguments

(Run `python main.py --help` for a full list)

- `--search-mode`: (Required) `absolute`, `optimized_transform`, `hierarchical_kmeans`.
- `--similarity-metric`: `cosine`, `euclidean_sim`, `manhattan_sim`, `dot_product`. Default: `cosine`.
- `--num-data-rows`, `--num-data-cols`: Define synthetic dataset size.
- `--spark-cores`, `--driver-memory`: Spark resource allocation.
- `--num-partitions`: Target number of Spark partitions for processing. Defaults to number of Spark cores if not set or set to 0.
- `--python-log-level`: Logging verbosity for Python messages from `main.py` (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `--spark-log-level`: Logging verbosity for Spark's internal engine messages (`WARN`, `ERROR`, `INFO`), passed to `SparkContext.setLogLevel()`.
- `--spark-local-dir`: Directory for Spark's temporary shuffle/spill files (default: `./spark-tmp-shuffle`).
- For `optimized_transform`: `--num-references-N`, `--ref-selection-sample-size`.
- For `hierarchical_kmeans`: `--tree-filename`, `--force-rebuild-tree`, `--k-per-level`, `--max-depth`, `--min-leaf-size`, `--kmeans-use-minibatch`.

------

## Logging and Monitoring

### Application Logs (Python)

- The `main.py` script uses Python's `logging` module.
- Log level is controlled by `--python-log-level` (default `INFO`). Set to `DEBUG` for more verbose output from the script, including Spark execution plans.
- Logs are printed to standard output with a format: `TIMESTAMP - LEVEL - MODULE.FUNCTION:LINENO - MESSAGE`.

### Spark Engine Logs (Log4j)

- Primarily controlled by `conf/log4j.properties` (if provided and passed via Makefile). This affects the Java/Scala side of Spark.
- The `--spark-log-level` argument to `main.py` calls `SparkContext.setLogLevel()`, which also influences PySpark-interfaced logging.

---

## License Notice

This code is shared exclusively for evaluation purposes related to the recruitment process at IDOVEN. Copying, distribution, modification, or reuse of this code is strictly prohibited without explicit written authorization from the author.

© Guillermo Sarasa, 2025. All rights reserved.
