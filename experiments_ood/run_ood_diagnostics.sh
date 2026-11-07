#!/usr/bin/env bash
set -euo pipefail

# Called by its real absolute path, including from Slurm's --wrap script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths_ood.sh"

usage() {
    echo "Usage: run_ood_diagnostics.sh DATASET SEED_GROUP [Python options]" >&2
    echo "Internal Slurm mode: run_ood_diagnostics.sh --task-map PATH [Python options]" >&2
}

if [[ "${1:-}" == --task-map ]]; then
    if [[ $# -lt 2 || ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
        echo "--task-map requires a file and SLURM_ARRAY_TASK_ID." >&2
        exit 2
    fi
    task_map="$2"
    shift 2
    if [[ ! -r "$task_map" ]]; then
        echo "Task map is not readable: $task_map" >&2
        exit 2
    fi
    line_number=$((SLURM_ARRAY_TASK_ID + 1))
    task_line="$(sed -n "${line_number}p" "$task_map")"
    if [[ -z "$task_line" ]]; then
        echo "No task-map row for array index $SLURM_ARRAY_TASK_ID." >&2
        exit 2
    fi
    IFS=$'\t' read -r dataset seed_group extra <<< "$task_line"
    if [[ -n "${extra:-}" ]]; then
        echo "Invalid task-map row for array index $SLURM_ARRAY_TASK_ID." >&2
        exit 2
    fi
else
    if [[ $# -lt 2 ]]; then usage; exit 2; fi
    dataset="$1"
    seed_group="$2"
    shift 2
fi
case "$dataset" in
    mnist|fashion_mnist|pamap2|har) ;;
    *) echo "Unknown diagnostic dataset: $dataset" >&2; exit 2 ;;
esac
if [[ ! "$seed_group" =~ ^[1-9][0-9]*$ ]]; then
    echo "Seed group must be a positive integer: $seed_group" >&2
    exit 2
fi

if [[ -n "${CHDC_ENV_PATH:-}" ]]; then
    environment_dir="$(ood_path native "$CHDC_ENV_PATH")"
    activate="${environment_dir%/}/bin/activate"
    if [[ ! -f "$activate" ]]; then
        echo "CHDC_ENV_PATH has no bin/activate: $CHDC_ENV_PATH" >&2
        exit 2
    fi
    source "$activate"
fi
python_bin="${PYTHON:-python3}"
entrypoint="$SCRIPT_DIR/diagnose_ood.py"
args=("$@")
for option in "${args[@]}"; do
    case "$option" in
        --dataset|--dataset=*)
            echo "The dataset is supplied as the launcher's first argument." >&2
            exit 2
            ;;
    esac
done

if [[ "${python_bin,,}" == *.exe ]]; then
    python_bin="$(ood_path native "$python_bin")"
    entrypoint="$(ood_path windows "$entrypoint")"
    converted=()
    while [[ ${#args[@]} -gt 0 ]]; do
        case "${args[0]}" in
            --results-root|--data-root)
                if [[ ${#args[@]} -lt 2 ]]; then
                    echo "${args[0]} requires a path" >&2
                    exit 2
                fi
                converted+=("${args[0]}" "$(ood_path windows "${args[1]}")")
                args=("${args[@]:2}")
                ;;
            --results-root=*|--data-root=*)
                option="${args[0]%%=*}"
                value="${args[0]#*=}"
                converted+=("$option=$(ood_path windows "$value")")
                args=("${args[@]:1}")
                ;;
            *) converted+=("${args[0]}"); args=("${args[@]:1}") ;;
        esac
    done
    args=("${converted[@]}")
    if [[ -n "${CHDC_RESULTS_ROOT:-}" ]]; then
        export CHDC_RESULTS_ROOT="$(ood_path windows "$CHDC_RESULTS_ROOT")"
    fi
    if [[ -n "${CHDC_DATA_ROOT:-}" ]]; then
        export CHDC_DATA_ROOT="$(ood_path windows "$CHDC_DATA_ROOT")"
    fi
fi
# The Python runner also applies its explicit --threads setting.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$python_bin" -u "$entrypoint" "$seed_group" --dataset "$dataset" "${args[@]}"
