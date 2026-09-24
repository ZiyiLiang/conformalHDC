#!/bin/bash

# one seed, 4 rep takes 10 mins , 2 G
# # --- QUICK TEST LIST ---
# SEED_LIST=(1)
# ALPHA_LIST=(0.05)

# --- EXPERIMENT LIST ---
SEED_LIST=($(seq 1 25))
ALPHA_LIST=(0.05 0.02)

# Slurm parameters
EXPNAME="isolet"
MEMO=8G                            
TIME=00-06:00:00                 
CORE=4                     

# Assemble order
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CORE" --time="$TIME" --partition=biodatascience.p"

# Thread limit: match the requested cores
export OPENBLAS_NUM_THREADS=$CORE
export OMP_NUM_THREADS=$CORE
export MKL_NUM_THREADS=$CORE

# Directories
LOGS="logs/"$EXPNAME
OUT_DIR="results/"$EXPNAME

mkdir -p $LOGS
mkdir -p $OUT_DIR

comp=0
incomp=0

for SEED in ${SEED_LIST[@]}; do
    for ALPHA in ${ALPHA_LIST[@]}; do
    
        # Define Job Name and Output File
        JOBN="seed"$SEED"_alpha"$ALPHA
        OUT_FILE=$OUT_DIR"/seed"$SEED"_alpha"$ALPHA".csv"
        
        COMPLETE=0
        if [[ -f $OUT_FILE ]]; then
            COMPLETE=1
            ((comp++))
        fi

        if [[ $COMPLETE -eq 0 ]]; then
            ((incomp++))
            
            # Script arguments: <seed> <alpha>
            SCRIPT="exp_isolet.sh $SEED $ALPHA"
            
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

echo "Jobs already completed: $comp"
echo "Jobs submitted: $incomp"
