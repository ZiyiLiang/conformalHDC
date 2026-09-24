#!/bin/bash
set -e

# Run from this experiment directory.
source ../data/config.sh
/usr/bin/time -v python3 exp_har.py "$@"
