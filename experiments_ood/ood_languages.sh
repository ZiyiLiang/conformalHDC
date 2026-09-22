#!/bin/bash
set -e

# Run from this experiment directory.
source ../data/config.sh
python3 ood_languages.py "$@"
