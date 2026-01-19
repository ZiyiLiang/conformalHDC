#!/bin/bash

# Initialize Conda
eval "$(conda shell.bash hook)"

# Activate the correct environment
conda activate hdc

# Run python script
# Arguments: $1 = Seed Group ID, $2 = Alpha
python3 exp_languages.py $1 $2