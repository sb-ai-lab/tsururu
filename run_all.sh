#!/bin/bash

OUTPUT_DIR="results"

Run the experiments with different configurations
python run_exp.py --config=config/ili/dlinear.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/ili/cyclenet.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/ili/patchtst.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/ili/gpt4ts.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/ili/timemixer.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/ili/timesnet.yaml --output_dir=$OUTPUT_DIR

python run_exp.py --config=config/etth1/dlinear.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/etth1/cyclenet.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/etth1/patchtst.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/etth1/gpt4ts.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/etth1/timemixer.yaml --output_dir=$OUTPUT_DIR
python run_exp.py --config=config/etth1/timesnet.yaml --output_dir=$OUTPUT_DIR
