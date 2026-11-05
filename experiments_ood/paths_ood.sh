#!/usr/bin/env bash
# Convert paths only at the Bash/Windows-Python boundary. Linux jobs keep their
# ordinary paths; WSL and Git Bash use their installed native conversion utility.
ood_path() {
    local target="$1" value="$2"
    if [[ "$target" == native ]]; then
        if [[ "$value" =~ ^[A-Za-z]:[\\/] || "$value" == \\\\* ]]; then
            if command -v wslpath >/dev/null 2>&1; then
                wslpath -a -u "$value"
            elif command -v cygpath >/dev/null 2>&1; then
                cygpath -a -u "$value"
            else
                printf '%s\n' "$value"
            fi
        else
            printf '%s\n' "$value"
        fi
    elif [[ "$target" == windows ]]; then
        if [[ "$value" =~ ^[A-Za-z]:[\\/] || "$value" == \\\\* ]]; then
            printf '%s\n' "$value"
        elif command -v wslpath >/dev/null 2>&1; then
            wslpath -a -w "$value"
        elif command -v cygpath >/dev/null 2>&1; then
            cygpath -a -w "$value"
        else
            printf '%s\n' "$value"
        fi
    else
        echo "Unknown path target: $target" >&2
        return 2
    fi
}
