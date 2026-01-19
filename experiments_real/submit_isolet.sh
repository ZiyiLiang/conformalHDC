#!/bin/bash

# --- EXPERIMENT LIST ---
SEED_LIST=($(seq 1 5))
ALPHA_LIST=(0.05)

# Slurm parameters
EXPNAME="isolet"
MEMO=8G                            
TIME=00-04:00:00                 
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