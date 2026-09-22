#!/bin/bash
set -e

# Run from this experiment directory.
source ../data/config.sh
python3 exp_rat.py "$@"
