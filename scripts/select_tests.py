#!/usr/bin/env python3
"""Select which pytest files to run based on files changed since a git ref.

Mapping:
    flux1km/foo.py                -> tests/test_foo.py
    flux1km/sub/foo.py            -> tests/test_foo.py (or tests/sub/test_foo.py if it exists)
    tests/test_foo.py             -> tests/test_foo.py
    shared files (see SHARED_PATTERNS) -> ALL (full suite)

Prints one of:
    ALL             -> run the full suite
    NONE            -> nothing relevant changed
    <paths...>      -> space-separated list of test files

Example:
    python scripts/select_tests.py --base origin/main
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
from pathlib import Path

SRC_ROOT = "flux1km"
TESTS_ROOT = "tests"
# Changes to these mean "many modules may be affected" -> run everything.
SHARED_PATTERNS = [
    "pyproject.toml",
    "setup.cfg",
    "requirements*.txt",
    "**/conftest.py",
    "conftest.py",
    "**/__init__.py",
    ".github/workflows/*",
    "scripts/select_tests.py",
    "*/common/*",
    "*/utils/*",
    "**/common.py",
    "**/utils.py",
    # Core modules imported by most others: a change there runs the full suite.
    "flux1km/schema.py",
    "flux1km/io.py",
    "flux1km/table.py",
    "flux1km/aggregation.py",
    "flux1km/features.py",
    "flux1km/models.py",
]


def changed_files(base: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_shared(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in SHARED_PATTERNS)


def tests_for(path: str) -> list[str]:
    p = Path(path)
    if p.suffix != ".py":
        return []
    if p.parts and p.parts[0] == TESTS_ROOT:
        return [path] if p.name.startswith("test_") and p.exists() else []
    if p.parts and p.parts[0] == SRC_ROOT:
        # Drop the package prefix, keep any subfolders.
        rel = Path(*p.parts[1:])
        candidates = [
            Path(TESTS_ROOT) / rel.parent / f"test_{p.stem}.py",
            Path(TESTS_ROOT) / f"test_{p.stem}.py",
        ]
        return [str(c) for c in candidates if c.exists()][:1]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="git ref to diff against")
    args = parser.parse_args()

    try:
        files = changed_files(args.base)
    except subprocess.CalledProcessError:
        print("ALL")  # unknown base (e.g. first push) -> be safe
        return

    if any(is_shared(f) for f in files):
        print("ALL")
        return

    selected = sorted({t for f in files for t in tests_for(f)})
    print(" ".join(selected) if selected else "NONE")


if __name__ == "__main__":
    main()
