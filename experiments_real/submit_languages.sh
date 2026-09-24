#!/bin/bash

# one seed, 4 rep, 20 mins,  ~ 8 G
# # --- QUICK TEST LIST ---
# SEED_LIST=(1)
# ALPHA_LIST=(0.01)

# # --- FULL EXPERIMENT LIST ---
SEED_LIST=($(seq 1 25))
ALPHA_LIST=(0.01)

# Slurm parameters
EXPNAME="languages"
MEMO=16G                             
TIME=00-04:00:00                     
CORE=8                           

# Assemble submission command
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CORE" --time="$TIME" --gres=gpu:1 --partition=biodatascience.p"

# Thread limit: match the requested cores
export OPENBLAS_NUM_THREADS=$CORE
export OMP_NUM_THREADS=$CORE
export MKL_NUM_THREADS=$CORE

LOGS="logs/"$EXPNAME
OUT_DIR="results/"$EXPNAME
mkdir -p $LOGS
mkdir -p $OUT_DIR

comp=0
incomp=0

for SEED in ${SEED_LIST[@]}; do
    for ALPHA in ${ALPHA_LIST[@]}; do
        # unique job name
        JOBN="seed"$SEED"_alpha"$ALPHA
        
        # expected output file
        OUT_FILE=$OUT_DIR"/seed"$SEED"_alpha"$ALPHA".csv"
        
        COMPLETE=0
        if [[ -f $OUT_FILE ]]; then
            COMPLETE=1
            ((comp++))
        fi

        if [[ $COMPLETE -eq 0 ]]; then
            ((incomp++))
            
            # Arguments passed to the wrapper script
            SCRIPT="exp_languages.sh $SEED $ALPHA"
            
            OUTF=$LOGS"/"$JOBN".out"
            ERRF=$LOGS"/"$JOBN".err"
            
            ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
            echo $ORD
            $ORD
        fi
    done
done

echo "Jobs already completed: $comp"
echo "Jobs submitted: $incomp"
