#!/bin/bash

# Parameters
# This runs 10 jobs. Since each job has 10 reps, total = 100 independent runs.
SEED_LIST=$(seq 1 20)
EXPNAME="mnist"

# Slurm parameters
MEMO=8G                             # Memory (8GB is plenty for MNIST HDC)
TIME=00-01:00:00                    # Time (20 mins to be safe)
CORE=1                              # Cores
GPU=1                               # Request 1 GPU

# Assemble order
# Adjust partition name/GPU flags to your specific cluster config
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CORE" --gres=gpu:1 --time="$TIME" --partition=biodatascience.p"

# Directories
LOGS="logs/"$EXPNAME
OUT_DIR="results/"$EXPNAME

mkdir -p $LOGS
mkdir -p $OUT_DIR

comp=0
incomp=0

for SEED in $SEED_LIST; do
    JOBN="seed"$SEED
    OUT_FILE=$OUT_DIR"/seed"$SEED".csv"
    COMPLETE=0
    
    if [[ -f $OUT_FILE ]]; then
        COMPLETE=1
        ((comp++))
    fi

    if [[ $COMPLETE -eq 0 ]]; then
        ((incomp++))
        
        # Script to be run
        SCRIPT="exp_mnist.sh $SEED"
        
        # Log files
        OUTF=$LOGS"/"$JOBN".out"
        ERRF=$LOGS"/"$JOBN".err"
        
        # Assemble slurm order
        ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
        
        # Print and Submit
        echo $ORD
        $ORD
    fi
done

echo "Jobs already completed: $comp, submitted unfinished jobs: $incomp"