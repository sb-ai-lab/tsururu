# Examples
- This directory collects and aggregates benchmark results across multiple datasets, and provides the full set of scripts and notebooks needed to reproduce all figures and tables for any dataset.

- Also there are scripts to check reproducibility of the models in Tsururu and compare them with original implementations.

- Finally, it contains examples of how to use Tsururu for various tasks, including tutorials and scripts for benchmarking.

## Directory Structure
```
all_configurations_benchmark
├── scripts
│   ├── constants.py      # Parameters for models for each dataset
│   └── validation.py     # Some useful functions to do proper validation
│   ├── run_exp.py        # Run a full experiments over all interesting configurations
│   ├── run_exp_ratio.py  # The same as run_exp.py, but for ratio regime in transformers
│   ├── get_results.py    # Load and parse raw experiment results to scaled metrics and aggregate them in one csv file
|
├── notebooks
│   ├── clean_results.ipynb       # Clean and normalize results from get_results.py
│   └── aggregated_results.ipynb  # Generate summary tables & plots
│
├── results
│   └── agg_results__normalized_True_cleaned.csv  # Output of clean_results.ipynb for some datasets
|
reproducibility_check
└── config/     # Configuration files
└── run_all.sh  # Shell script to execute the full reproducibility_check pipeline end-to-end
└── run_exp.py  # Script to launch experiments for the config file
|
Example_1_All_configurations.py             # Script for benchmarking available strategies, models and preprocessing methods on a dataset.
Tutorial_1_Quick_start.ipynb                # Simple usage examples
Tutorial_2_Strategies.ipynb                 # Covers forecasting strategies.
Tutorial_3_Transformers_and_Pipeline.ipynb  # Provides a description of available data preprocessing techniques.
Tutorial_4_Neural_Networks.ipynb            # Demonstrates working with neural networks.
```

## How to Use to get and visualize results
1. **Run experiments**  
   - Add settings to `scripts/constants.py`, related to the dataset and models you want to benchmark.
   - Use `scripts/run_exp.py` to launch your experiments on any dataset.
   - Use `scripts/get_results.py` to parse the raw logs and generate a CSV file with scaled metrics.

2. **Clean raw outputs**  
   - Use `notebooks/clean_results.ipynb`.

3. **Aggregate and visualize**  
   - Use `notebooks/aggregated_results.ipynb`.  
