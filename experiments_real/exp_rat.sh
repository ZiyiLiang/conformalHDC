#!/bin/bash
set -e

# Run from this experiment directory.
source ../data/config.sh
# python3 exp_rat.py "$@"
/usr/bin/time -v python3 exp_rat.py "$@"