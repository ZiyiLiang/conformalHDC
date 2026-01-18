#!/bin/bash

# Initialize Conda
eval "$(conda shell.bash hook)"

# module load slurm

# Activate the correct environment
conda activate hdc

# Run the python script with the passed seed_group_id and alpha
python3 exp_mnist.py $1 $2