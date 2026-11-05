#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/paths_ood.sh"
if [[ $# -lt 1 ]]; then
    echo "Usage: submit_ood.sh {mnist|fashion_mnist|pamap2|har} [options]" >&2
    exit 2
fi
dataset="$1"
shift
case "$dataset" in
    mnist|fashion_mnist|pamap2|har) ;;
    *) echo "Unknown OOD dataset: $dataset" >&2; exit 2 ;;
esac

usage() {
    cat <<EOF
Usage: submit_${dataset}.sh [--local] [--dry-run] [--seed N | --seeds N ...]
                          [--results-root PATH] [Python options]

Without --local, submit seed groups 1..25 to Slurm (four repetitions per group).
--local runs one group in the current process; its default group is 1.
--seed N / --seeds N ... restrict the groups submitted.
--dry-run prints commands without submitting, running Python, or creating files.
--force reruns completed groups. Other options are passed to the Python entry point,
including --data-root, --repetitions, --temperature, --device, and --threads.

Environment: PYTHON, CHDC_ENV_PATH, CHDC_RESULTS_ROOT, CHDC_DATA_ROOT,
             CHDC_PARTITION, CHDC_JOB_MEMORY, CHDC_JOB_TIME, CHDC_JOB_CPUS.
EOF
}

local_mode=0
dry_run=0
force=0
results_root="${CHDC_RESULTS_ROOT:-$REPO_ROOT/results}"
seeds=()
job_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) local_mode=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --seed)
            if [[ $# -lt 2 ]]; then echo "--seed requires an integer" >&2; exit 2; fi
            seeds+=("$2")
            shift 2
            ;;
        --seeds)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "--seeds requires one or more integers" >&2; exit 2
            fi
            while [[ $# -gt 0 && "$1" != --* ]]; do seeds+=("$1"); shift; done
            ;;
        --results-root)
            if [[ $# -lt 2 ]]; then echo "--results-root requires a path" >&2; exit 2; fi
            results_root="$2"
            shift 2
            ;;
        --force) force=1; job_args+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; job_args+=("$@"); break ;;
        *) job_args+=("$1"); shift ;;
    esac
done

if [[ ${#seeds[@]} -eq 0 ]]; then
    if (( local_mode )); then
        seeds=(1)
    else
        for ((seed=1; seed<=25; seed++)); do seeds+=("$seed"); done
    fi
fi
for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[1-9][0-9]*$ ]]; then
        echo "Seed groups must be positive integers: $seed" >&2
        exit 2
    fi
done
if (( local_mode )) && [[ ${#seeds[@]} -ne 1 ]]; then
    echo "--local runs exactly one seed group; use --seed N." >&2
    exit 2
fi

results_root="$(ood_path native "$results_root")"
exp_name="ood_${dataset}"
output_dir="$results_root/$exp_name"
log_dir="$results_root/logs/$exp_name"
job_script="$SCRIPT_DIR/ood_${dataset}.sh"
# Always pass the same resolved results root to the completion check and job.
python_args=(--results-root "$results_root" "${job_args[@]}")
completed=0
launched=0

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

if (( ! dry_run )); then mkdir -p -- "$output_dir" "$log_dir"; fi
for seed in "${seeds[@]}"; do
    marker="$output_dir/seed${seed}_complete.json"
    auroc="$output_dir/seed${seed}_ood.csv"
    operating="$output_dir/seed${seed}_operating.csv"
    if (( ! dry_run && ! force )) && [[ -s "$marker" && -s "$auroc" && -s "$operating" ]]; then
        # Python verifies configuration, provenance, CSV hashes and row counts.
        if bash "$job_script" "$seed" "${python_args[@]}" --check-complete; then
            echo "Already complete: $exp_name seed group $seed"
            completed=$((completed + 1))
            continue
        fi
    fi

    job_name="${exp_name}_seed${seed}"
    if (( local_mode )); then
        command=(bash "$job_script" "$seed" "${python_args[@]}")
    else
        command=(sbatch
            "--mem=${CHDC_JOB_MEMORY:-8G}"
            --nodes=1 --ntasks=1
            "--cpus-per-task=${CHDC_JOB_CPUS:-1}"
            "--time=${CHDC_JOB_TIME:-00-03:00:00}"
            "--chdir=$SCRIPT_DIR"
            "--job-name=$job_name"
            "--output=$log_dir/${job_name}.out"
            "--error=$log_dir/${job_name}.err")
        if [[ -n "${CHDC_PARTITION-biodatascience.p}" ]]; then
            command+=("--partition=${CHDC_PARTITION-biodatascience.p}")
        fi
        command+=("$job_script" "$seed" "${python_args[@]}")
    fi
    print_command "${command[@]}"
    if (( ! dry_run )); then
        if (( local_mode )); then
            "${command[@]}" 2>&1 | tee "$log_dir/${job_name}.out"
        else
            "${command[@]}"
        fi
    fi
    launched=$((launched + 1))
done
if (( dry_run )); then
    echo "Dry run: $launched job(s) listed; no jobs run or submitted."
else
    echo "Jobs already complete: $completed; jobs launched: $launched"
fi
