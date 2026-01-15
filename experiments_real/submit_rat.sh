#!/bin/bash

# --- QUICK TEST LIST ---
# SEED_LIST=(1)
# RAT_LIST=(0)
# ALPHA_LIST=(0.2)

# --- FULL EXPERIMENT LIST ---
SEED_LIST=(1)
RAT_LIST=$(seq 0 4)      
ALPHA_LIST=(0.1 0.2 0.3) 

# Slurm parameters (CPU Only)
EXPNAME="odor_decoding"
MEMO=4G                           
TIME=00-02:00:00                   
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

for RAT in ${RAT_LIST[@]}; do
    mkdir -p $OUT_DIR

    for SEED in ${SEED_LIST[@]}; do
        for ALPHA in ${ALPHA_LIST[@]}; do
            
            # Define Job Name and Output File
            JOBN="rat"$RAT"_seed"$SEED"_alpha"$ALPHA
            OUT_FILE=$OUT_DIR"/rat"$RAT"_seed"$SEED"_alpha"$ALPHA".csv"
            
            COMPLETE=0
            if [[ -f $OUT_FILE ]]; then
                COMPLETE=1
                ((comp++))
            fi

            if [[ $COMPLETE -eq 0 ]]; then
                ((incomp++))
                
                # Script arguments: <seed> <rat_id> <alpha>
                SCRIPT="exp_rat.sh $SEED $RAT $ALPHA"
                
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

echo "Jobs already completed: $comp"
echo "Jobs submitted: $incomp"
