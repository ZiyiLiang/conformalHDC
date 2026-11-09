"""Comment-insensitive source compatibility for completed OOD runs."""
import ast
from functools import lru_cache
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Normalize only the completion-check statements introduced by this migration.
_CHECKER_TESTS = {
    "experiments_ood/common.py": (
        "is_complete",
        'not run_configs_match(marker["run_config"], config, marker.get("git_revision"))',
        'marker["run_config"] != config',
    ),
    "experiments_ood/diagnose_ood.py": (
        "validate_export",
        'not common.run_configs_match(marker.get("run_config"), config, marker.get("git_revision")) '
        'or any(marker.get(key) != value for key, value in marker["run_config"].items())',
        'marker.get("run_config") != config '
        'or any(marker.get(key) != value for key, value in config.items())',
    ),
}


def source_fingerprint(source, relative_path):
    """Hash Python syntax, preserving executable statements and strings."""
    tree = ast.parse(source)
    path = str(relative_path).replace("\\", "/")
    if path == "experiments_ood/common.py":
        checker_import = ast.dump(ast.parse(
            "from experiments_ood.completion import run_configs_match").body[0])
        tree.body = [node for node in tree.body if ast.dump(node) != checker_import]
    if path in _CHECKER_TESTS:
        name, replacement, original = _CHECKER_TESTS[path]
        replacement = ast.dump(ast.parse(replacement, mode="eval").body)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                for statement in ast.walk(node):
                    if isinstance(statement, ast.If) and ast.dump(statement.test) == replacement:
                        statement.test = ast.parse(original, mode="eval").body
    syntax = ast.dump(tree, include_attributes=False).encode("utf-8")
    return "python-ast-v1:" + hashlib.sha256(syntax).hexdigest()


def _source_paths(hashes):
    if not isinstance(hashes, dict) or not hashes:
        return None
    normalized = {}
    for path, digest in hashes.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            return None
        path = path.replace("\\", "/")
        parts = PurePosixPath(path)
        if (parts.is_absolute() or ".." in parts.parts or ":" in path or
                parts.suffix != ".py" or not re.fullmatch(r"[0-9a-f]{64}", digest) or
                parts.as_posix() in normalized):
            return None
        normalized[parts.as_posix()] = digest
    return normalized


@lru_cache(maxsize=256)
def _historical_source(repository, revision, path, digest):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        return None
    try:
        result = subprocess.run(
            ["git", "show", f"{revision}:{path}"], cwd=repository,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    source = result.stdout
    lf = source.replace(b"\r\n", b"\n")
    for candidate in (source, lf, lf.replace(b"\n", b"\r\n")):
        if hashlib.sha256(candidate).hexdigest() == digest:
            return candidate
    return None


def run_configs_match(stored, requested, git_revision=None):
    """Require matching settings and verified source, allowing ordinary comments."""
    if not isinstance(stored, dict) or not isinstance(requested, dict):
        return False
    if ({key: value for key, value in stored.items() if key != "source_hashes"} !=
            {key: value for key, value in requested.items() if key != "source_hashes"}):
        return False
    old = _source_paths(stored.get("source_hashes"))
    new = _source_paths(requested.get("source_hashes"))
    if old is None or new is None or old.keys() != new.keys():
        return False
    if stored == requested:
        return True
    for path, digest in old.items():
        if digest == new[path]:
            continue
        previous = _historical_source(str(PROJECT_ROOT), git_revision, path, digest)
        if previous is None:
            return False
        try:
            current = (PROJECT_ROOT / path).read_bytes()
            if hashlib.sha256(current).hexdigest() != new[path]:
                return False
            if source_fingerprint(previous, path) != source_fingerprint(current, path):
                return False
        except (OSError, SyntaxError, ValueError, TypeError):
            return False
    return True
