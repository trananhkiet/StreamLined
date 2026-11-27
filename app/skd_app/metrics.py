"""Parse training logs / history.csv into per-epoch metrics, summarise them, and export TensorBoard events."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

EPOCH_RE = re.compile(
    r"epoch (\d+)/(\d+) \| lr=([0-9.eE+-]+) \| train top1=([0-9.]+) \| test top1=([0-9.]+) \| best=([0-9.]+)"
)
TEACHER_RE = re.compile(r"Teacher (\w+) \((\w+)\): top1=([0-9.]+), top5=([0-9.]+)")


def _log_text(raw: str) -> str:
    """Kaggle logs come as a JSON list of {stream_name, time, data}; plain text is accepted too."""
    stripped = raw.strip()
    if stripped.startswith("[") or stripped.startswith(",{"):
        try:
            entries = json.loads(stripped if stripped.startswith("[") else "[" + stripped.lstrip(",") + "]")
            return "".join(e.get("data", "") for e in entries if isinstance(e, dict))
        except json.JSONDecodeError:
            pass
    return raw


def parse_log(raw: str) -> tuple[pd.DataFrame, dict]:
    text = _log_text(raw)
    rows = {}
    for m in EPOCH_RE.finditer(text):
        epoch = int(m.group(1))
        rows[epoch] = {
            "epoch": epoch,
            "epochs_total": int(m.group(2)),
            "lr": float(m.group(3)),
            "train_top1": float(m.group(4)),
            "test_top1": float(m.group(5)),
            "best_so_far": float(m.group(6)),
        }
    info: dict = {}
    teacher = TEACHER_RE.search(text)
    if teacher:
        info.update(teacher_arch=teacher.group(1), teacher_source=teacher.group(2),
                    teacher_top1=float(teacher.group(3)), teacher_top5=float(teacher.group(4)))
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["epoch"]))
    return df, info


def load_history_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("epoch").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty or "test_top1" not in df:
        return {}
    top1 = df["test_top1"]
    best_idx = int(top1.idxmax())
    return {
        "epochs_done": int(df["epoch"].max()),
        "epochs_total": int(df["epochs_total"].max()) if "epochs_total" in df else int(df["epoch"].max()),
        "best": round(float(top1.max()), 2),
        "best_epoch": int(df.loc[best_idx, "epoch"]),
        "final": round(float(top1.iloc[-1]), 2),
        "tb10": round(float(top1.tail(10).mean()), 2),
    }


def find_artifacts(output_dir: Path) -> dict[str, Path]:
    """Locate history.csv / summary.json / best checkpoint / kernel log inside a downloaded Kaggle output folder."""
    found: dict[str, Path] = {}
    if not output_dir.exists():
        return found
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name == "history.csv":
            found["history"] = path
        elif name == "summary.json":
            found["summary"] = path
        elif name.endswith("_skd_best.pth"):
            found["best_checkpoint"] = path
        elif name == "last_checkpoint.pt":
            found["last_checkpoint"] = path
        elif name.endswith(".log") or (name.endswith(".json") and "log" in name):
            found.setdefault("log", path)
    return found


def load_run_metrics(output_dir: Path, log_text: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Best available per-epoch table for a run: history.csv > downloaded log > fetched log text."""
    artifacts = find_artifacts(output_dir)
    info: dict = {}
    df = pd.DataFrame()
    if "history" in artifacts:
        df = load_history_csv(artifacts["history"])
    if "log" in artifacts:
        log_df, info = parse_log(artifacts["log"].read_text(errors="ignore"))
        if df.empty:
            df = log_df
    if df.empty and log_text:
        df, info = parse_log(log_text)
    if "summary" in artifacts:
        summary = json.loads(artifacts["summary"].read_text())
        teacher = summary.get("teacher_metrics", {})
        if "top1" in teacher:
            info.setdefault("teacher_top1", round(float(teacher["top1"]), 2))
        info["epochs_total"] = summary.get("config", {}).get("epochs")
    return df, info


def export_tensorboard(df: pd.DataFrame, out_dir: Path) -> int:
    """Write every numeric column as a scalar per epoch. Returns the number of scalars written."""
    from torch.utils.tensorboard import SummaryWriter

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("events.out.tfevents.*"):
        old.unlink()
    written = 0
    writer = SummaryWriter(log_dir=str(out_dir))
    try:
        numeric = [c for c in df.columns if c not in ("epoch", "epochs_total") and pd.api.types.is_numeric_dtype(df[c])]
        for _, row in df.iterrows():
            step = int(row["epoch"])
            for col in numeric:
                if pd.notna(row[col]):
                    group = "accuracy" if "top" in col else ("loss" if "loss" in col else "train")
                    writer.add_scalar(f"{group}/{col}", float(row[col]), step)
                    written += 1
    finally:
        writer.close()
    return written
