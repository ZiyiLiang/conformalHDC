#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for the four current benchmark entry points.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths_ood.sh"
if [[ $# -lt 1 ]]; then
    echo "Usage: run_ood.sh {mnist|fashion_mnist|pamap2|har} <seed_group> [Python options]" >&2
    exit 2
fi
dataset="$1"
shift
case "$dataset" in
    mnist|fashion_mnist|pamap2|har) ;;
    *) echo "Unknown OOD dataset: $dataset" >&2; exit 2 ;;
esac

# Use an already activated environment, an explicit interpreter, or an optional
# Linux virtual environment. The legacy ../chdc environment is never assumed.
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
entrypoint="$SCRIPT_DIR/ood_${dataset}.py"
args=("$@")
if [[ "${python_bin,,}" == *.exe ]]; then
    # WSL cannot exec C:/... directly, and Windows Python cannot read /mnt/c/...
    # as a Windows path. Keep every conversion explicit and preserve quoting.
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
exec "$python_bin" -u "$entrypoint" "${args[@]}"
