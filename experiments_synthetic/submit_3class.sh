#!/bin/bash

# --- QUICK TEST LIST ---
SEED_LIST=(1)
ALPHA_LIST=(0.1)

# --- FULL EXPERIMENT LIST ---
# SEED_LIST=($(seq 1 10))
# ALPHA_LIST=(0.1)

# Experiment Name (Used for folder naming)
EXPNAME="3class"

# Slurm parameters
MEMO=4G                             
TIME=00-01:00:00                    
CORE=1                              

# Assemble order
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CORE" --time="$TIME" --partition=biodatascience.p"

# Directories
LOGS="logs/"$EXPNAME
OUT_DIR="results/"$EXPNAME

mkdir -p $LOGS
mkdir -p $OUT_DIR

comp=0
incomp=0

for SEED in ${SEED_LIST[@]}; do
    for ALPHA in ${ALPHA_LIST[@]}; do
    
        # Define Job Name and Output File (Matches format in exp_3class.py)
        JOBN="seed"$SEED"_alpha"$ALPHA
        OUT_FILE=$OUT_DIR"/seed"$SEED"_alpha"$ALPHA".csv"
        
        COMPLETE=0
        # Check if results already exist to avoid re-running
        if [[ -f $OUT_FILE ]]; then
            COMPLETE=1
            ((comp++))
        fi

        if [[ $COMPLETE -eq 0 ]]; then
            ((incomp++))
            
            # Script to be run: <seed> <alpha>
            SCRIPT="exp_3class.sh $SEED $ALPHA"
            
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
done

echo "Jobs already completed: $comp, submitted unfinished jobs: $incomp"