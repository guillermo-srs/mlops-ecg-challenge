# Makefile for ECG Similarity Search Project

# --- Variables ---
PYTHON_INTERPRETER ?= python3
VENV_DIR         := .venv
SPARK_SUBMIT     ?= spark-submit
LOG4J_PROPERTIES_PATH := ./conf/log4j.properties

# --- Default values for main.py arguments ---
# These can be overridden by setting them on the 'make' command line
# e.g., make run-absolute SPARK_CORES_VAL=2
DEFAULT_APP_NAME_VAL         := ECGSimilaritySearchApp
DEFAULT_SPARK_CORES_VAL      := 4
DEFAULT_DRIVER_MEMORY_VAL    := 4g
DEFAULT_SPARK_LOG_LEVEL_VAL  := ERROR
DEFAULT_PYTHON_LOG_LEVEL_VAL := INFO
DEFAULT_NUM_DATA_ROWS_VAL    := 10000
DEFAULT_NUM_DATA_COLS_VAL    := 100
DEFAULT_ID_COLUMN_NAME_VAL   := ecg_id
DEFAULT_DATA_PATH_PREFIX_VAL := ./generated_data_make
DEFAULT_NUM_PARTITIONS_VAL   := # Will be derived from SPARK_CORES_VAL if not set
DEFAULT_QUERY_VECTOR_SEED_VAL:= 42
DEFAULT_TOP_X_RESULTS_VAL    := 5
DEFAULT_SIMILARITY_METRIC_VAL:= cosine
DEFAULT_NUM_REFERENCES_N_VAL      := 20
DEFAULT_REF_SELECTION_SAMPLE_SIZE_VAL := 5000
DEFAULT_TREE_FILENAME_VAL    := hkmeans_trees_make.pkl
DEFAULT_K_PER_LEVEL_VAL      := 5
DEFAULT_MAX_DEPTH_VAL        := 3
DEFAULT_MIN_LEAF_SIZE_VAL    := 50
DEFAULT_KMEANS_SEED_VAL      := 42

# --- Helper to get numeric core count ---
get_spark_cores_num = $(shell echo $(1) | sed 's/local\[//' | sed 's/\]//' | sed 's/\*//' | sed 's/auto//' )

# --- Targets ---

.PHONY: setup
setup: $(VENV_DIR)/touchfile

$(VENV_DIR)/touchfile: requirements.txt
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Activating virtual environment and installing dependencies..."
	@. $(VENV_DIR)/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt
	@echo "Creating conf directory for log4j.properties if it doesn't exist..."
	@mkdir -p ./conf
	@echo "Please ensure you have a 'conf/log4j.properties' file for custom logging."
	@echo "A sample one might be:"
	@echo "log4j.rootCategory=WARN, CONSOLE"
	@echo "log4j.appender.CONSOLE=org.apache.log4j.ConsoleAppender"
	@echo "log4j.appender.CONSOLE.layout=org.apache.log4j.PatternLayout"
	@echo "log4j.appender.CONSOLE.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n"
	@echo "log4j.appender.CONSOLE.target=System.err"
	@echo "log4j.logger.org.apache.spark=WARN"
	@echo "Setup complete. Activate with: source $(VENV_DIR)/bin/activate"
	@touch $(VENV_DIR)/touchfile

.PHONY: clean
clean:
	@echo "Cleaning generated data, Spark event logs, Python cache, and conf dir..."
	rm -rf $(DEFAULT_DATA_PATH_PREFIX_VAL)* ./generated_data_make_* # Clean specific and pattern
	rm -rf ./spark-events
	rm -rf $(VENV_DIR)
	rm -rf ./conf
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Clean complete."

.PHONY: lint
lint:
	@echo "Linting Python files..."
	@. $(VENV_DIR)/bin/activate && \
		flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
	@. $(VENV_DIR)/bin/activate && \
		flake8 . --count --exit-zero --max-complexity=12 --max-line-length=127 --statistics || true

# --- Base script execution ---
# This macro takes the SEARCH_MODE_ARG as its first argument
define _run_script
	$(eval _APP_NAME := $(or $(APP_NAME),$(DEFAULT_APP_NAME_VAL)$(1)))
	$(eval _SPARK_CORES_STR := $(or $(SPARK_CORES),$(DEFAULT_SPARK_CORES_VAL)))
	$(eval _SPARK_CORES_NUM := $(call get_spark_cores_num,$(_SPARK_CORES_STR)))
	$(eval _DRIVER_MEMORY := $(or $(DRIVER_MEMORY),$(DEFAULT_DRIVER_MEMORY_VAL)))
	$(eval _SPARK_LOG_LEVEL := $(or $(SPARK_LOG_LEVEL),$(DEFAULT_SPARK_LOG_LEVEL_VAL)))
	$(eval _PYTHON_LOG_LEVEL := $(or $(PYTHON_LOG_LEVEL),$(DEFAULT_PYTHON_LOG_LEVEL_VAL)))
	$(eval _NUM_DATA_ROWS := $(or $(NUM_DATA_ROWS),$(DEFAULT_NUM_DATA_ROWS_VAL)))
	$(eval _NUM_DATA_COLS := $(or $(NUM_DATA_COLS),$(DEFAULT_NUM_DATA_COLS_VAL)))
	$(eval _ID_COLUMN_NAME := $(or $(ID_COLUMN_NAME),$(DEFAULT_ID_COLUMN_NAME_VAL)))
	$(eval _SIMILARITY_METRIC_ARG := $(or $(SIMILARITY_METRIC),$(DEFAULT_SIMILARITY_METRIC_VAL)))
	$(eval _DATA_PATH_PREFIX_ARG := $(or $(DATA_PATH_PREFIX),$(DEFAULT_DATA_PATH_PREFIX_VAL))_$(subst optimized_transform,opt,$(subst hierarchical_kmeans,hkm,$(subst absolute,abs,$(1))))_$(_SIMILARITY_METRIC_ARG))
	$(eval _REGENERATE_DATA_FLAG_ARG := $(if $(REGENERATE_DATA),--regenerate-data,))
    $(eval _NUM_PARTITIONS_TEMP := $(or $(NUM_PARTITIONS),$(_SPARK_CORES_NUM)))
    $(eval _NUM_PARTITIONS_ARG := $(if $(filter 0,$(_NUM_PARTITIONS_TEMP)),$(_SPARK_CORES_NUM),$(_NUM_PARTITIONS_TEMP)))
	$(eval _QUERY_VECTOR_SEED_ARG := $(or $(QUERY_VECTOR_SEED),$(DEFAULT_QUERY_VECTOR_SEED_VAL)))
	$(eval _TOP_X_RESULTS_ARG := $(or $(TOP_X_RESULTS),$(DEFAULT_TOP_X_RESULTS_VAL)))
	$(eval _LOG4J_CONFIG_STR_ARG = $(if $(wildcard $(LOG4J_PROPERTIES_PATH)),--driver-java-options "-Dlog4j.configuration=file:$(LOG4J_PROPERTIES_PATH)"))

	@echo "--- Running $(1) Search ---"
	@echo "  App Name: $(_APP_NAME)"
	@echo "  Spark Cores: $(_SPARK_CORES_STR) (Numeric: $(_SPARK_CORES_NUM))"
	@echo "  Driver Memory: $(_DRIVER_MEMORY)"
	@echo "  Spark Log Level (Python): $(_SPARK_LOG_LEVEL)"
	@echo "  Main Log Level (Python): $(_PYTHON_LOG_LEVEL)"
	@echo "  Log4j Config: $(if $(_LOG4J_CONFIG_STR_ARG),Using $(LOG4J_PROPERTIES_PATH),Not specified)"
	@echo "  Num Data Rows: $(_NUM_DATA_ROWS)"
	@echo "  Num Data Cols: $(_NUM_DATA_COLS)"
	@echo "  ID Column: $(_ID_COLUMN_NAME)"
	@echo "  Similarity Metric: $(_SIMILARITY_METRIC_ARG)"
	@echo "  Data Path Prefix: $(_DATA_PATH_PREFIX_ARG)"
	@echo "  Regenerate Data: $(if $(_REGENERATE_DATA_FLAG_ARG),true,false)"
	@echo "  Num Partitions: $(_NUM_PARTITIONS_ARG)"
	@echo "  Query Seed: $(_QUERY_VECTOR_SEED_ARG)"
	@echo "  Top X Results: $(_TOP_X_RESULTS_ARG)"
	
	$(SPARK_SUBMIT) \
	--master local[$(_SPARK_CORES_NUM)] \
	--driver-memory $(_DRIVER_MEMORY) \
	$(_LOG4J_CONFIG_STR_ARG) \
	main.py \
	--app-name "$(_APP_NAME)" \
	--spark-cores "$(_SPARK_CORES_STR)" \
	--driver-memory "$(_DRIVER_MEMORY)" \
	--spark-log-level "$(_SPARK_LOG_LEVEL)" \
	--python-log-level "$(_PYTHON_LOG_LEVEL)" \
	--num-data-rows $(_NUM_DATA_ROWS) \
	--num-data-cols $(_NUM_DATA_COLS) \
	--id-column-name "$(_ID_COLUMN_NAME)" \
	--data-path-prefix "$(_DATA_PATH_PREFIX_ARG)" \
	$(_REGENERATE_DATA_FLAG_ARG) \
	--num-partitions $(_NUM_PARTITIONS_ARG) \
	--query-vector-seed $(_QUERY_VECTOR_SEED_ARG) \
	--search-mode $(1) \
	--top-x-results $(_TOP_X_RESULTS_ARG) \
	--similarity-metric "$(_SIMILARITY_METRIC_ARG)" \
	$(2) # Extra arguments for specific modes
endef

.PHONY: run-absolute
run-absolute:
	$(call _run_script,absolute)

.PHONY: run-optimized
run-optimized:
	$(eval _NUM_REFERENCES_N_ARG := $(or $(NUM_REFERENCES_N),$(DEFAULT_NUM_REFERENCES_N_VAL)))
	$(eval _REF_SELECTION_SAMPLE_SIZE_ARG := $(or $(REF_SELECTION_SAMPLE_SIZE),$(DEFAULT_REF_SELECTION_SAMPLE_SIZE_VAL)))
	$(eval _USE_MEDIAN_FLAG_ARG := $(if $(USE_MEDIAN_FOR_CENTER),--use-median-for-center,))
	@echo "  Num References (N): $(_NUM_REFERENCES_N_ARG)"
	@echo "  Ref Sample Size: $(_REF_SELECTION_SAMPLE_SIZE_ARG)"
	@echo "  Use Median for Center: $(if $(_USE_MEDIAN_FLAG_ARG),true,false)"
	$(call _run_script,optimized_transform,--num-references-N $(_NUM_REFERENCES_N_ARG) --ref-selection-sample-size $(_REF_SELECTION_SAMPLE_SIZE_ARG) $(_USE_MEDIAN_FLAG_ARG))

.PHONY: run-hkmeans
run-hkmeans:
	$(eval _TREE_FILENAME_ARG := $(or $(TREE_FILENAME),$(DEFAULT_TREE_FILENAME_VAL)))
	$(eval _FORCE_REBUILD_TREE_FLAG_ARG := $(if $(FORCE_REBUILD_TREE),--force-rebuild-tree,))
	$(eval _K_PER_LEVEL_ARG := $(or $(K_PER_LEVEL),$(DEFAULT_K_PER_LEVEL_VAL)))
	$(eval _MAX_DEPTH_ARG := $(or $(MAX_DEPTH),$(DEFAULT_MAX_DEPTH_VAL)))
	$(eval _MIN_LEAF_SIZE_ARG := $(or $(MIN_LEAF_SIZE),$(DEFAULT_MIN_LEAF_SIZE_VAL)))
	$(eval _KMEANS_SEED_ARG := $(or $(KMEANS_SEED),$(DEFAULT_KMEANS_SEED_VAL)))
	@echo "  Tree Filename: $(_TREE_FILENAME_ARG)"
	@echo "  Force Rebuild Tree: $(if $(_FORCE_REBUILD_TREE_FLAG_ARG),true,false)"
	@echo "  K Per Level: $(_K_PER_LEVEL_ARG)"
	@echo "  Max Depth: $(_MAX_DEPTH_ARG)"
	@echo "  Min Leaf Size: $(_MIN_LEAF_SIZE_ARG)"
	@echo "  KMeans Seed: $(_KMEANS_SEED_ARG)"
	$(call _run_script,hierarchical_kmeans,--tree-filename "$(_TREE_FILENAME_ARG)" $(_FORCE_REBUILD_TREE_FLAG_ARG) --k-per-level $(_K_PER_LEVEL_ARG) --max-depth $(_MAX_DEPTH_ARG) --min-leaf-size $(_MIN_LEAF_SIZE_ARG) --kmeans-seed $(_KMEANS_SEED_ARG))

.PHONY: run-all
run-all:
	$(MAKE) run-absolute $(filter-out $@,$(MAKECMDGOALS))
	$(MAKE) run-optimized $(filter-out $@,$(MAKECMDGOALS))
	$(MAKE) run-hkmeans $(filter-out $@,$(MAKECMDGOALS))

.PHONY: run-all-regenerate
run-all-regenerate: REGENERATE_DATA=true FORCE_REBUILD_TREE=true
run-all-regenerate: run-all

.PHONY: help
help:
	@echo "Makefile for ECG Similarity Search Project"
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "Core Targets:"
	@echo "  setup                 - Set up Python virtual environment, install dependencies, and create sample conf dir."
	@echo "                          Ensure 'conf/log4j.properties' is configured to control Spark's Java logging."
	@echo "  lint                  - Lint Python source files."
	@echo "  clean                 - Remove generated files, virtual environment, conf dir, and caches."
	@echo ""
	@echo "Run Targets (variables can be overridden on the command line):"
	@echo "  run-absolute          - Run 'absolute' search mode."
	@echo "  run-optimized         - Run 'optimized_transform' search mode."
	@echo "  run-hkmeans           - Run 'hierarchical_kmeans' search mode."
	@echo "  run-all               - Run all three search modes sequentially (passes overrides to sub-makes)."
	@echo "  run-all-regenerate    - Run all modes, forcing data regeneration and tree rebuild."
	@echo ""
	@echo "Common Override Variables (examples):"
	@echo "  SPARK_CORES=8               (Default: $(DEFAULT_SPARK_CORES_VAL))"
	@echo "  DRIVER_MEMORY=8g            (Default: $(DEFAULT_DRIVER_MEMORY_VAL))"
	@echo "  SPARK_LOG_LEVEL=INFO        (Default: $(DEFAULT_SPARK_LOG_LEVEL_VAL)) (For PySpark logs)"
	@echo "  NUM_DATA_ROWS=100000        (Default: $(DEFAULT_NUM_DATA_ROWS_VAL))"
	@echo "  NUM_DATA_COLS=50            (Default: $(DEFAULT_NUM_DATA_COLS_VAL))"
	@echo "  ID_COLUMN_NAME=my_id        (Default: $(DEFAULT_ID_COLUMN_NAME_VAL))"
	@echo "  DATA_PATH_PREFIX=./my_data  (Default: $(DEFAULT_DATA_PATH_PREFIX_VAL))"
	@echo "  REGENERATE_DATA=true        (Default: false)"
	@echo "  NUM_PARTITIONS=16           (Default: derived from SPARK_CORES)"
	@echo "  QUERY_VECTOR_SEED=123       (Default: $(DEFAULT_QUERY_VECTOR_SEED_VAL))"
	@echo "  TOP_X_RESULTS=10            (Default: $(DEFAULT_TOP_X_RESULTS_VAL))"
	@echo "  SIMILARITY_METRIC=euclidean_sim (Default: $(DEFAULT_SIMILARITY_METRIC_VAL))"
	@echo ""
	@echo "  For 'run-optimized':"
	@echo "    NUM_REFERENCES_N=30"
	@echo "    REF_SELECTION_SAMPLE_SIZE=20000"
	@echo "    USE_MEDIAN_FOR_CENTER=true"
	@echo "  For 'run-hkmeans':"
	@echo "    TREE_FILENAME=custom_tree.pkl"
	@echo "    FORCE_REBUILD_TREE=true"
	@echo "    K_PER_LEVEL=4, MAX_DEPTH=5, MIN_LEAF_SIZE=20, KMEANS_SEED=7"
	@echo ""
	@echo "Example: make run-hkmeans NUM_DATA_ROWS=50000 SPARK_CORES=4 SIMILARITY_METRIC=manhattan_sim"
	@echo "         (To control Spark's Java logs, edit 'conf/log4j.properties' e.g. log4j.rootCategory=ERROR, CONSOLE)"
	@echo ""

default: help
