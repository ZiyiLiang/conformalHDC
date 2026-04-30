#!/bin/bash

# Initialize Conda
eval "$(conda shell.bash hook)"

# module load slurm

# Activate the correct environment
conda activate hdc

# Run the python script with the passed seed_group_id
python3 ood_isolet.py $1