"""Build a Kaggle notebook from streamlined_kd_cifar100.py with chosen env-var DEFAULTS.

    python3 kaggle/make_variant_notebook.py OUT.ipynb KEY=VALUE [KEY=VALUE ...]

Each KEY must appear exactly once as os.getenv("KEY", "<default>") in the script; only that default is replaced, so the
resulting notebook differs from the shared source by exactly those lines (verify with a diff of the .ipynb text).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import build_notebook as nb_builder

ROOT = Path(__file__).resolve().parent


def main() -> None:
    out = Path(sys.argv[1])
    overrides = dict(arg.split("=", 1) for arg in sys.argv[2:])
    tmp = out.with_suffix(".tmp.ipynb")
    nb_builder.TARGET = tmp
    nb_builder.build()
    notebook = json.loads(tmp.read_text())
    tmp.unlink()
    remaining = dict(overrides)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        for key, value in overrides.items():
            pattern = re.compile(r'os\.getenv\("%s", "[^"]*"\)' % re.escape(key))
            text, count = pattern.subn('os.getenv("%s", "%s")' % (key, value), text)
            if count:
                assert count == 1 and key in remaining, f"{key} appears {count} times"
                remaining.pop(key)
        lines = text.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    assert not remaining, f"env defaults not found in script: {sorted(remaining)}"
    out.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {out} with defaults {overrides}")


if __name__ == "__main__":
    main()
