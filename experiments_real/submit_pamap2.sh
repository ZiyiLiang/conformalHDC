#!/bin/bash


# --- EXPERIMENT LIST ---
SEED_LIST=($(seq 1 25))
ALPHA_LIST=(0.1)

# Slurm parameters
EXPNAME="pamap2"
MEMO=4G                            
TIME=00-03:00:00                 
CORE=1                        

# Build the Slurm command.
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CORE" --time="$TIME" --partition=biodatascience.p"

# Directories
LOGS="logs/"$EXPNAME
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/$EXPNAME"  # RESULTS_ROOT in data/config.py

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
            SCRIPT="exp_pamap2.sh $SEED $ALPHA"
            
            # Log files
            OUTF=$LOGS"/"$JOBN".out"
            ERRF=$LOGS"/"$JOBN".err"
            
            # Build the job submission command.
            ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
            
            # Print and Submit
            echo $ORD
            $ORD
        fi
    done
done

echo "Jobs already completed: $comp"
echo "Jobs submitted: $incomp"
