"""Convert the percent-format source file into the Kaggle .ipynb artifact."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "streamlined_kd_cifar100.py"
TARGET = ROOT / "streamlined_kd_cifar100.ipynb"


def markdown_source(lines: list[str]) -> list[str]:
    converted = []
    for line in lines:
        if line.startswith("# "):
            converted.append(line[2:])
        elif line.startswith("#"):
            converted.append(line[1:])
        else:
            converted.append(line)
    return [line + "\n" for line in converted]


def code_source(lines: list[str]) -> list[str]:
    return [line + "\n" for line in lines]


def build() -> None:
    cells: list[dict[str, object]] = []
    kind: str | None = None
    buffered: list[str] = []

    def flush() -> None:
        nonlocal buffered
        if kind is None:
            return
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": markdown_source(buffered)})
        else:
            cells.append({
                "cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": code_source(buffered),
            })
        buffered = []

    for line in SOURCE.read_text().splitlines():
        if line == "# %% [markdown]":
            flush()
            kind = "markdown"
        elif line == "# %%":
            flush()
            kind = "code"
        else:
            buffered.append(line)
    flush()

    notebook = {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "gpu", "dataSources": [], "dockerImageVersionId": None},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {TARGET} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
