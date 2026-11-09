#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/paths_ood.sh"

usage() {
    cat <<'EOF'
Usage: submit_ood_diagnostics.sh [options] [Python options]
  --dataset IDS          all (default), or comma-separated dataset IDs
                         mnist,fashion_mnist,pamap2,har
  --seed N | --seeds N ...
                         Groups to run (Slurm default 1..20; local default 1)
  --repetitions N        Repetitions per group (default 5)
  --results-root PATH    Base results root; diagnostic outputs use its
                         ood_diagnostics/ood_DATASET subdirectories
  --max-concurrent N     Global array concurrency, 1..40 (default 40)
  --local               Run exactly one dataset/group in the foreground
  --dry-run             Print the plan without creating files or running Python
  --monitor             Monitor this Slurm array to completion

Other options, including --data-root, --dimension, --temperature, --device,
--threads and --check-complete, are passed to the shared Python diagnostic runner.
There is no force-overwrite option. Incomplete existing outputs cause an error.

Environment: PYTHON, CHDC_ENV_PATH, CHDC_RESULTS_ROOT, CHDC_DATA_ROOT,
CHDC_PARTITION, CHDC_JOB_MEMORY, CHDC_JOB_TIME, CHDC_JOB_CPUS.
EOF
}

local_mode=0
dry_run=0
monitor=0
dataset_spec=all
repetitions=5
max_concurrent=40
results_root="${CHDC_RESULTS_ROOT:-$REPO_ROOT/results}"
seeds=()
python_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) local_mode=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --monitor) monitor=1; shift ;;
        --dataset|--datasets)
            if [[ $# -lt 2 ]]; then echo "$1 requires dataset IDs." >&2; exit 2; fi
            dataset_spec="$2"; shift 2 ;;
        --seed)
            if [[ $# -lt 2 ]]; then echo "--seed requires an integer." >&2; exit 2; fi
            seeds+=("$2"); shift 2 ;;
        --seeds)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then echo "--seeds requires integers." >&2; exit 2; fi
            while [[ $# -gt 0 && "$1" != --* ]]; do seeds+=("$1"); shift; done
            ;;
        --repetitions|--max-concurrent|--results-root)
            if [[ $# -lt 2 ]]; then echo "$1 requires a value." >&2; exit 2; fi
            case "$1" in
                --repetitions) repetitions="$2" ;;
                --max-concurrent) max_concurrent="$2" ;;
                --results-root) results_root="$2" ;;
            esac
            shift 2 ;;
        --force|--force=*) echo "Diagnostic outputs cannot be force-overwritten." >&2; exit 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; python_args+=("$@"); break ;;
        *) python_args+=("$1"); shift ;;
    esac
done
if [[ ! "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
    echo "--repetitions must be a positive integer." >&2; exit 2
fi
if [[ ! "$max_concurrent" =~ ^[1-9][0-9]*$ ]] || (( max_concurrent > 40 )); then
    echo "--max-concurrent must be between 1 and 40." >&2; exit 2
fi
if [[ "$dataset_spec" == all ]]; then
    datasets=(mnist fashion_mnist pamap2 har)
else
    IFS=',' read -r -a datasets <<< "$dataset_spec"
fi
if [[ ${#datasets[@]} -eq 0 ]]; then echo "At least one dataset is required." >&2; exit 2; fi
declare -A seen_datasets=()
for dataset in "${datasets[@]}"; do
    case "$dataset" in mnist|fashion_mnist|pamap2|har) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
    if [[ -n "${seen_datasets[$dataset]:-}" ]]; then echo "Duplicate dataset: $dataset" >&2; exit 2; fi
    seen_datasets[$dataset]=1
done
if [[ ${#seeds[@]} -eq 0 ]]; then
    if (( local_mode )); then seeds=(1); else for ((seed=1; seed<=20; seed++)); do seeds+=("$seed"); done; fi
fi
declare -A seen_seeds=()
for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[1-9][0-9]*$ ]]; then echo "Seed groups must be positive integers: $seed" >&2; exit 2; fi
    if [[ -n "${seen_seeds[$seed]:-}" ]]; then echo "Duplicate seed group: $seed" >&2; exit 2; fi
    seen_seeds[$seed]=1
done
if (( local_mode )) && [[ ${#datasets[@]} -ne 1 || ${#seeds[@]} -ne 1 ]]; then
    echo "--local requires exactly one dataset and one seed group." >&2; exit 2
fi
for option in "${python_args[@]}"; do
    case "$option" in
        --dataset|--dataset=*|--datasets|--datasets=*|--results-root|--results-root=*|--repetitions|--repetitions=*|--force|--force=*)
            echo "Use submission options before -- for: $option" >&2; exit 2 ;;
    esac
done

results_root="$(ood_path native "$results_root")"
if [[ "$results_root" != /* && ! "$results_root" =~ ^[A-Za-z]:[\\/] ]]; then results_root="$PWD/$results_root"; fi
run_script="$SCRIPT_DIR/run_ood_diagnostics.sh"
pending_tasks=()
completed=0
for dataset in "${datasets[@]}"; do
    dataset_pending=0
    dataset_completed=0
    for seed in "${seeds[@]}"; do
        prefix="$results_root/ood_diagnostics/ood_${dataset}/seed${seed}"
        if [[ -f "${prefix}_scores.csv.gz" && -f "${prefix}_roc.csv.gz" &&
              -f "${prefix}_ood.csv" && -f "${prefix}_operating.csv" ]]; then
            completed=$((completed + 1))
            dataset_completed=$((dataset_completed + 1))
        else
            pending_tasks+=("$dataset"$'\t'"$seed")
            dataset_pending=$((dataset_pending + 1))
        fi
    done
    printf '%s: %s complete; %s jobs x %s repetitions = %s repetitions pending\n' "$dataset" "$dataset_completed" "$dataset_pending" "$repetitions" "$((dataset_pending * repetitions))"
done
task_count=${#pending_tasks[@]}
printf 'Total: %s complete; %s pending jobs; maximum concurrent jobs: %s\n' "$completed" "$task_count" "$max_concurrent"
if (( task_count == 0 )); then
    echo "All requested diagnostic result files already exist; no jobs run or submitted."
    exit 0
fi
print_command() { printf '%q ' "$@"; printf '\n'; }

if (( local_mode )); then
    command=(bash "$run_script" "${datasets[0]}" "${seeds[0]}" --results-root "$results_root" --repetitions "$repetitions" "${python_args[@]}")
    print_command "${command[@]}"
    if (( dry_run )); then echo "Dry run: no files created and no Python run."; exit 0; fi
    local_log_dir="$results_root/logs/ood_diagnostics/local_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    mkdir -p -- "$local_log_dir"
    "${command[@]}" 2>&1 | tee "$local_log_dir/job.out"
    exit 0
fi

if [[ -z "${PYTHON:-}" ]]; then
    if command -v python3 >/dev/null 2>&1; then export PYTHON="$(command -v python3)"; else echo "Set PYTHON to the existing HDC interpreter." >&2; exit 2; fi
fi
submission_dir="$results_root/logs/ood_diagnostics/submission_$(date -u +%Y%m%dT%H%M%SZ)_$$"
task_map="$submission_dir/tasks.tsv"
# --wrap invokes the repository file itself, so its BASH_SOURCE is never Slurm's copy.
printf -v wrap '%q ' bash "$run_script" --task-map "$task_map" --results-root "$results_root" --repetitions "$repetitions" "${python_args[@]}"
command=(sbatch --parsable
    "--mem=${CHDC_JOB_MEMORY:-8G}" --nodes=1 --ntasks=1
    "--cpus-per-task=${CHDC_JOB_CPUS:-1}" "--time=${CHDC_JOB_TIME:-00-03:00:00}"
    "--chdir=$SCRIPT_DIR" --job-name=ood_diagnostics
    "--array=0-$((task_count - 1))%${max_concurrent}"
    "--output=$submission_dir/%A_%a.out" "--error=$submission_dir/%A_%a.err")
if [[ -n "${CHDC_PARTITION-biodatascience.p}" ]]; then command+=("--partition=${CHDC_PARTITION-biodatascience.p}"); fi
command+=(--wrap "$wrap")
print_command "${command[@]}"
if (( dry_run )); then
    echo "Dry run: one global Slurm array listed; no files created or jobs submitted."
    exit 0
fi
for required in sbatch squeue; do
    if ! command -v "$required" >/dev/null 2>&1; then echo "Required Slurm command unavailable: $required" >&2; exit 2; fi
done
if (( monitor )) && ! command -v sacct >/dev/null 2>&1; then echo "--monitor requires sacct." >&2; exit 2; fi
user_name="${USER:-$(id -un)}"
existing_jobs="$(squeue -h -u "$user_name" -o '%i')"
if [[ -n "$existing_jobs" ]]; then
    echo "Existing user jobs found; review concurrency before submitting another array:" >&2
    printf '%s\n' "$existing_jobs" >&2
    exit 2
fi
mkdir -p -- "$submission_dir"
printf '%s\n' "${pending_tasks[@]}" > "$task_map"
submission="$("${command[@]}")"
job_id="${submission%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then echo "Unexpected sbatch response: $submission" >&2; exit 2; fi
{
    printf 'array_job_id=%s\ntasks=%s\nrepetitions_per_group=%s\nmax_concurrent=%s\n' "$job_id" "$task_count" "$repetitions" "$max_concurrent"
    printf 'task_map=%s\nresults_root=%s\npython=%s\n' "$task_map" "$results_root" "$PYTHON"
    print_command "${command[@]}"
} > "$submission_dir/submission.txt"
printf 'Submitted diagnostic array job %s (%s tasks, cap %s).\n' "$job_id" "$task_count" "$max_concurrent"
printf 'Submission record: %s\n' "$submission_dir/submission.txt"
if (( ! monitor )); then exit 0; fi

read_accounting() {
    sacct -X -j "$job_id" -n -P -o 'JobID%40,State%32,ExitCode'
}
failed_tasks() {
    awk -F'|' -v root="$job_id" '$1 ~ ("^" root "_[0-9]+$") && ($2 ~ /^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)/ || ($2 == "COMPLETED" && $3 != "0:0")) { print }'
}
while true; do
    accounting="$(read_accounting)"
    failures="$(printf '%s\n' "$accounting" | failed_tasks)"
    if [[ -n "$failures" ]]; then
        echo "Diagnostic array contains failed tasks; inspect logs before retrying:" >&2
        printf '%s\n' "$failures" >&2
        exit 1
    fi
    active="$(squeue -r -h -u "$user_name" -o '%i|%T' | awk -F'|' -v root="$job_id" '$1 ~ ("^" root "_[0-9]+$") { print $2 }')"
    if [[ -z "$active" ]]; then break; fi
    printf 'Array %s at %s: ' "$job_id" "$(date -u +%H:%M:%SZ)"
    printf '%s\n' "$active" | awk '{ states[$0]++ } END { for (state in states) printf "%s=%d ", state, states[state]; printf "\n" }'
    sleep 20
done
for ((attempt=0; attempt<6; attempt++)); do
    accounting="$(read_accounting)"
    failures="$(printf '%s\n' "$accounting" | failed_tasks)"
    if [[ -n "$failures" ]]; then printf '%s\n' "$failures" >&2; exit 1; fi
    completed="$(printf '%s\n' "$accounting" | awk -F'|' -v root="$job_id" '$1 ~ ("^" root "_[0-9]+$") && $2 == "COMPLETED" && $3 == "0:0" { seen[$1]=1 } END { for (id in seen) n++; print n+0 }')"
    if [[ "$completed" -eq "$task_count" ]]; then
        printf 'All %s diagnostic tasks completed with exit 0.\n' "$task_count"
        exit 0
    fi
    sleep 2
done
echo "Could not confirm every array task completed successfully; review accounting:" >&2
printf '%s\n' "$accounting" >&2
exit 1
