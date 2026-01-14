#!/bin/bash

# Initialize Conda
eval "$(conda shell.bash hook)"

# Activate the correct environment
conda activate hdc

# Run the python script with passed arguments: <seed> <rat_id> <alpha>
python3 exp_rat.py $1 $2 $3