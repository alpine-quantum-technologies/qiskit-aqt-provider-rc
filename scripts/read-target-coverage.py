#!/usr/bin/env python3
"""Extract the coverage target from the pyproject.toml file.

This is used by the Github workflows to write the coverage report comment on PRs."""

import shlex
import subprocess
from pathlib import Path

import tomlkit

if __name__ == "__main__":
    repo_root = Path(
        subprocess.run(shlex.split("git rev-parse --show-toplevel"), capture_output=True)
        .stdout.strip()
        .decode("utf-8")
    )

    pyproject_path = repo_root / "pyproject.toml"
    with open(pyproject_path, encoding="utf-8") as fp:
        data = tomlkit.load(fp)
        print(float(data["tool"]["coverage"]["report"]["fail_under"]) / 100.0)  # type: ignore[index, arg-type]
