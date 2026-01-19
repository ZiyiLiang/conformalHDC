#!/bin/bash

# Initialize Conda
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate hdc

# Run the python script with passed arguments: <seed> <alpha>
python3 exp_isolet.py $1 $2