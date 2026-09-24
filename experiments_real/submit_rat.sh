#!/bin/bash

# One rat for 25 seeds takes 8 mins, peaks memory < 1GB

# --- QUICK TEST LIST ---
# SEED_LIST=(1)
# RAT_LIST=(0)
# ALPHA_LIST=(0.2)

# # --- FULL EXPERIMENT LIST ---
SEED_LIST=($(seq 1 25))
RAT_LIST=$(seq 0 4)      
#ALPHA_LIST=(0.1 0.2) 
ALPHA_LIST=(0.2) 
BETA_LIST=(0.3)

# Slurm parameters (CPU Only)
EXPNAME="odor_decoding"
MEMO=2G                           
TIME=00-04:00:00                   
CORE=1                              

# Assemble order
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CORE" --time="$TIME" --partition=biodatascience.p"

# Thred limit
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Directories
LOGS="logs/"$EXPNAME
OUT_DIR="results/"$EXPNAME

mkdir -p $LOGS
mkdir -p $OUT_DIR

comp=0
incomp=0

for RAT in ${RAT_LIST[@]}; do
    for SEED in ${SEED_LIST[@]}; do
        for ALPHA in ${ALPHA_LIST[@]}; do
            for BETA in ${BETA_LIST[@]}; do
            
                # Define Job Name and Output File (Added BETA)
                JOBN="rat"$RAT"_seed"$SEED"_alpha"$ALPHA"_beta"$BETA
                OUT_FILE=$OUT_DIR"/rat"$RAT"_seed"$SEED"_alpha"$ALPHA"_beta"$BETA".csv"
                
                COMPLETE=0
                if [[ -f $OUT_FILE ]]; then
                    COMPLETE=1
                    ((comp++))
                fi

                if [[ $COMPLETE -eq 0 ]]; then
                    ((incomp++))
                    
                    # Script arguments: <seed> <rat_id> <alpha> <beta>
                    SCRIPT="exp_rat.sh $SEED $RAT $ALPHA $BETA"
                    
                    # Log files
                    OUTF=$LOGS"/"$JOBN".out"
                    ERRF=$LOGS"/"$JOBN".err"
                    
                    # Assemble order
                    ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
                    
                    # Print and Submit
                    echo $ORD
                    $ORD
                fi
            done
        done
    done
done

echo "Jobs already completed: $comp"
echo "Jobs submitted: $incomp"
