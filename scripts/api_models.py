#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright Alpine Quantum Technologies GmbH 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import hashlib
import io
import shlex
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Union

import typer

app = typer.Typer()


@lru_cache
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(
        subprocess.run(
            shlex.split("git rev-parse --show-toplevel"),  # noqa: S603
            capture_output=True,
            check=True,
        )
        .stdout.strip()
        .decode("utf-8")
    ).absolute()


def default_schema_path() -> Path:
    """Default location of the API schema definition."""
    return repo_root() / "api" / "aqt_public.yml"


def default_models_path() -> Path:
    """Default destination of generated Pydantic models."""
    return repo_root() / "qiskit_aqt_provider" / "api_models.py"


def hash_stream(stream: Union[io.RawIOBase, io.BufferedIOBase]) -> str:
    """Compute the SHA256 hash of a binary IO stream."""
    hasher = hashlib.new("sha256")

    buffer = stream.read()
    hasher.update(buffer or b"")

    return hasher.hexdigest()


def generate_models(schema_path: Path) -> bytes:
    """Generate Pydantic models from a given schema.

    Args:
        schema_path: path to the file that contains the schema.

    Returns:
        Source code of the Pydantic models.
    """
    proc = subprocess.run(
        shlex.split(f"datamodel-codegen --input {schema_path}"),  # noqa: S603
        capture_output=True,
        check=True,
    )

    return proc.stdout


@app.command()
def generate(
    schema_path: Path = typer.Argument(default_schema_path),
    models_path: Path = typer.Argument(default_models_path),
) -> None:
    """Generate Pydantic models from a schema.

    Any existing content in `models_path` is will be lost!

    Args:
        schema_path: path to the file that contains the schema
        models_path: path of the file to write the generated models to.
    """
    models_path.write_bytes(generate_models(schema_path))


@app.command()
def check(
    schema_path: Path = typer.Argument(default_schema_path),
    models_path: Path = typer.Argument(default_models_path),
) -> None:
    """Check if the Python models in `models_path` match the schema in `schema_path`.

    For the check to succeed, `models_path` must contain exactly what the
    generator produces with `schema_path` as input.
    """
    reference_models = io.BytesIO()
    reference_models.write(generate_models(schema_path))
    reference_models.seek(0)
    reference_hash = hash_stream(reference_models)

    test_hash = ""
    with models_path.open(mode="rb", buffering=0) as fp:
        test_hash = hash_stream(fp)

    if test_hash != reference_hash:
        print("FAILED")
        sys.exit(1)

    print("OK")


if __name__ == "__main__":
    app()
