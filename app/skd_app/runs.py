"""Local run history (app/runs/runs.json). Stores config, status and metrics — never credentials."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RUNS_DIR

_LOCK = threading.Lock()
FORBIDDEN_KEYS = {"key", "api_key", "kaggle_key", "password", "token"}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_clean(obj: Any) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"refusing to store secret-like field {k!r} in run history")
            _check_clean(v)
    elif isinstance(obj, list):
        for v in obj:
            _check_clean(v)


class RunStore:
    def __init__(self, root: Path = RUNS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "runs.json"

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return []

    def _write(self, runs: list[dict]) -> None:
        _check_clean(runs)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(runs, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def list(self) -> list[dict]:
        return sorted(self._read(), key=lambda r: r.get("submitted_at", ""), reverse=True)

    def get(self, run_id: str) -> dict | None:
        return next((r for r in self._read() if r["run_id"] == run_id), None)

    def add(self, record: dict) -> dict:
        with _LOCK:
            runs = [r for r in self._read() if r["run_id"] != record["run_id"]]
            record.setdefault("submitted_at", now_iso())
            runs.append(record)
            self._write(runs)
        return record

    def update(self, run_id: str, **fields: Any) -> dict:
        with _LOCK:
            runs = self._read()
            for run in runs:
                if run["run_id"] == run_id:
                    run.update(fields)
                    run["updated_at"] = now_iso()
                    self._write(runs)
                    return run
        raise KeyError(run_id)

    def delete(self, run_id: str) -> None:
        with _LOCK:
            self._write([r for r in self._read() if r["run_id"] != run_id])

    def run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path
