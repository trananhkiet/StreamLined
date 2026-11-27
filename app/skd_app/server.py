"""FastAPI backend for the SKD demo dashboard (static frontend in skd_app/static/).

Kaggle credentials are kept in process memory only, keyed by an HttpOnly session cookie. They are never written
to disk, never returned by the API (only the username is), and never stored in run records.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import secrets
import socket
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Cookie, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import kaggle_client as kc
from . import metrics, models
from .config import DATASETS, RECIPES, REFERENCE, RUNS_DIR, STUDENTS, TEACHERS, TrainConfig, estimated_hours
from .runs import RunStore, now_iso

STATIC_DIR = Path(__file__).resolve().parent / "static"
COOKIE = "skd_session"


class CredIn(BaseModel):
    username: str
    key: str

class ConfigIn(BaseModel):
    teacher: str = "resnet56"
    student: str = "resnet20"
    dataset: str = "cifar100"
    recipe: str = "F1b"
    epochs: int = 300
    learning_rate: float = 0.1
    lr_milestones: str = "100,188,250"
    batch_size: int = 64
    temperature: float = 4.0
    seed: int = 2022
    ce_weight: float = 0.5
    kd_weight: float = 0.5
    smoke_test: bool = False


def uid() -> str:
    return secrets.token_hex(2)


class State:
    """Process-wide mutable state (one local user)."""

    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.store = RunStore(runs_dir)
        self.runs_dir = Path(runs_dir)
        self.sessions: dict[str, kc.Credentials] = {}
        self.jobs: dict[str, dict] = {}
        self.models: dict[str, tuple] = {}
        self.cifar = None
        self.tensorboard: subprocess.Popen | None = None
        self.lock = threading.Lock()


def create_app(runs_dir: Path = RUNS_DIR) -> FastAPI:
    app = FastAPI(title="SKD Demo", docs_url="/api/docs", openapi_url="/api/openapi.json")
    st = State(runs_dir)
    app.state.skd = st

    # ------------------------------------------------------------ helpers

    def creds_of(sid: str | None) -> kc.Credentials | None:
        return st.sessions.get(sid or "")

    def need_creds(sid: str | None) -> kc.Credentials:
        c = creds_of(sid)
        if not c:
            raise HTTPException(401, "Chưa kết nối Kaggle — nhập API key ở góc trên bên phải.")
        return c

    def get_run(run_id: str) -> dict:
        run = st.store.get(run_id)
        if not run:
            raise HTTPException(404, "Không tìm thấy run")
        return run

    def run_metrics(run: dict) -> tuple[Any, dict]:
        rdir = st.store.run_dir(run["run_id"])
        log_file = rdir / "kernel_log.txt"
        log_text = log_file.read_text(errors="ignore") if log_file.exists() else None
        df, info = metrics.load_run_metrics(rdir / "output", log_text)
        summary = metrics.summarize(df)
        merged = {**summary, **{k: v for k, v in info.items() if v is not None}}
        if merged and merged != run.get("metrics"):
            st.store.update(run["run_id"], metrics=merged)
        return df, merged

    def start_job(kind: str, target, *args) -> str:
        job_id = uuid.uuid4().hex[:10]
        st.jobs[job_id] = {"id": job_id, "kind": kind, "state": "running", "message": "", "started_at": now_iso()}

        def work():
            try:
                result = target(*args)
                st.jobs[job_id].update(state="done", result=result)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                st.jobs[job_id].update(state="error", message=str(exc)[-500:])

        threading.Thread(target=work, daemon=True).start()
        return job_id

    def load_model_spec(spec: str):
        """spec = teacher:<arch> | run:<run_id> | upload:<sha256>"""
        if spec in st.models:
            return st.models[spec]
        kind, _, ref = spec.partition(":")
        if kind == "teacher":
            path, arch = models.teacher_path(ref, st.runs_dir / "teachers"), ref
            if not path.exists():
                raise HTTPException(400, f"Teacher {ref} chưa được tải")
        elif kind == "run":
            run = get_run(ref)
            found = metrics.find_artifacts(st.store.run_dir(ref) / "output")
            if "best_checkpoint" not in found:
                raise HTTPException(400, "Run này chưa có checkpoint")
            path, arch = found["best_checkpoint"], run.get("config", {}).get("student")
        elif kind == "upload":
            path, arch = st.runs_dir / "uploads" / f"{ref}.pth", None
            if not path.exists():
                raise HTTPException(400, "Checkpoint upload không còn")
        else:
            raise HTTPException(400, "Model không hợp lệ")
        model, arch = models.load_model(path, arch)
        st.models[spec] = (model, arch)
        return st.models[spec]

    def image_b64(img: Image.Image, scale: int = 1) -> str:
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------------------ session / Kaggle account

    @app.get("/api/session")
    def session(skd_session: str | None = Cookie(default=None)):
        c = creds_of(skd_session)
        return {"connected": bool(c), "username": c.username if c else None}

    @app.post("/api/session")
    def login(body: CredIn, response: Response, skd_session: str | None = Cookie(default=None)):
        c = kc.Credentials(body.username.strip(), body.key.strip())
        if not c.is_complete():
            raise HTTPException(400, "Thiếu username hoặc key")
        sid = skd_session if skd_session in st.sessions else secrets.token_urlsafe(24)
        st.sessions[sid] = c
        response.set_cookie(COOKIE, sid, httponly=True, samesite="strict")
        return {"connected": True, "username": c.username}

    @app.delete("/api/session")
    def logout(response: Response, skd_session: str | None = Cookie(default=None)):
        st.sessions.pop(skd_session or "", None)
        response.delete_cookie(COOKIE)
        return {"connected": False}

    @app.post("/api/session/verify")
    def verify(skd_session: str | None = Cookie(default=None)):
        res = kc.verify(need_creds(skd_session))
        return {"ok": res.ok, "message": "Kết nối Kaggle OK" if res.ok else res.output[-400:]}

    # ------------------------------------------------------------ options / preview / submit

    @app.get("/api/options")
    def options():
        return {
            "teachers": TEACHERS, "students": STUDENTS, "recipes": RECIPES,
            "datasets": {k: {"label": v[0], "enabled": v[1]} for k, v in DATASETS.items()},
            "reference": {f"{t}|{s}": v for (t, s), v in REFERENCE.items()},
            "defaults": {k: v for k, v in asdict(TrainConfig()).items() if k != "extra_env"},
        }

    def to_cfg(body: ConfigIn) -> TrainConfig:
        return TrainConfig(**body.model_dump())

    @app.post("/api/preview")
    def preview(body: ConfigIn, skd_session: str | None = Cookie(default=None)):
        cfg = to_cfg(body)
        c = creds_of(skd_session)
        stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M")
        return {"errors": cfg.validate(), "env": cfg.to_env(), "slug": cfg.slug(stamp),
                "owner": c.username if c else None, "hours": round(estimated_hours(cfg), 1)}

    @app.post("/api/runs")
    def submit(body: ConfigIn, dry_run: bool = False, skd_session: str | None = Cookie(default=None)):
        cfg = to_cfg(body)
        errors = cfg.validate()
        if errors:
            raise HTTPException(400, "; ".join(errors))
        c = None if dry_run else need_creds(skd_session)
        owner = c.username if c else (creds_of(skd_session).username if creds_of(skd_session) else "your-kaggle-username")
        stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
        slug = cfg.slug(stamp[:9])
        run_id = f"{stamp}-{cfg.recipe.lower()}-s{cfg.seed}-{uid()}" + ("-draft" if dry_run else "")
        kdir = kc.build_kernel_dir(st.store.run_dir(run_id) / "kernel", owner, slug, cfg)
        record = {"run_id": run_id, "source": "draft" if dry_run else "kaggle", "kaggle_user": owner,
                  "kernel_ref": f"{owner}/{slug}", "url": f"https://www.kaggle.com/code/{owner}/{slug}",
                  "config": {**body.model_dump(), "env": cfg.to_env()}, "status": "draft" if dry_run else "queued"}
        if not dry_run:
            res = kc.push(kdir, c)
            info = kc.parse_push(res.output)
            record.update(status="queued" if res.ok else "push_failed", version=info.get("version"),
                          url=info.get("url", record["url"]), push_output=res.output[-2000:])
        st.store.add(record)
        return record

    # ------------------------------------------------------------ runs

    @app.get("/api/runs")
    def list_runs():
        return [{k: v for k, v in r.items() if k != "push_output"} for r in st.store.list()]

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        run = get_run(run_id)
        df, summary = run_metrics(run)
        rdir = st.store.run_dir(run_id)
        found = metrics.find_artifacts(rdir / "output")
        log_file = rdir / "kernel_log.txt" if (rdir / "kernel_log.txt").exists() else found.get("log")
        log_tail = metrics._log_text(log_file.read_text(errors="ignore"))[-6000:] if log_file else ""
        series = {c: [None if v != v else v for v in df[c].tolist()] for c in df.columns} if not df.empty else {}
        artifacts = {k: str(v.relative_to(rdir)) for k, v in found.items()}
        return {"run": get_run(run_id), "summary": summary, "series": series, "log_tail": log_tail,
                "artifacts": artifacts, "has_notebook": (rdir / "kernel" / kc.NOTEBOOK_NAME).exists()}

    @app.post("/api/runs/{run_id}/status")
    def refresh_status(run_id: str, skd_session: str | None = Cookie(default=None)):
        run = get_run(run_id)
        state, res = kc.status(run["kernel_ref"], need_creds(skd_session))
        if not res.ok:
            raise HTTPException(502, res.output[-400:])
        return st.store.update(run_id, status=state, status_checked_at=now_iso())

    @app.post("/api/runs/{run_id}/logs")
    def fetch_logs(run_id: str, skd_session: str | None = Cookie(default=None)):
        run = get_run(run_id)
        res = kc.logs(run["kernel_ref"], need_creds(skd_session))
        if not res.ok:
            raise HTTPException(502, res.output[-400:])
        if not res.output.strip():
            raise HTTPException(409, "Kaggle chưa trả log — log chỉ có sau khi run kết thúc. Trong lúc chạy hãy dùng Cập nhật trạng thái.")
        (st.store.run_dir(run_id) / "kernel_log.txt").write_text(res.output)
        return run_detail(run_id)

    @app.post("/api/runs/{run_id}/output")
    def fetch_output(run_id: str, skd_session: str | None = Cookie(default=None)):
        run = get_run(run_id)
        c = need_creds(skd_session)

        def work():
            res = kc.download_output(run["kernel_ref"], st.store.run_dir(run_id) / "output", c)
            if not res.ok:
                raise RuntimeError(res.output[-400:])
            run_metrics(get_run(run_id))
            return "ok"

        return {"job": start_job("output", work)}

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        get_run(run_id)
        st.store.delete(run_id)
        return {"deleted": run_id}

    @app.post("/api/runs/import")
    async def import_run(file: UploadFile = File(...), teacher: str = Form("resnet56"), student: str = Form("resnet20"),
                         recipe: str = Form("E4"), label: str = Form("")):
        run_id = "import-" + datetime.now(timezone.utc).strftime("%m%d-%H%M%S") + f"-{uid()}"
        out = st.store.run_dir(run_id) / "output"
        out.mkdir(parents=True, exist_ok=True)
        name = "history.csv" if (file.filename or "").endswith(".csv") else "imported.log"
        (out / name).write_bytes(await file.read())
        st.store.add({"run_id": run_id, "source": "import", "status": "complete", "file": file.filename,
                      "label": label, "config": {"teacher": teacher, "student": student, "recipe": recipe}})
        df, summary = run_metrics(get_run(run_id))
        if df.empty:
            st.store.delete(run_id)
            raise HTTPException(400, "Không đọc được epoch nào trong file này")
        return get_run(run_id)

    @app.post("/api/runs/import-sample")
    def import_sample():
        sample = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_kaggle_log.json"
        run_id = "sample-" + datetime.now(timezone.utc).strftime("%m%d-%H%M%S") + f"-{uid()}"
        out = st.store.run_dir(run_id) / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "imported.log").write_bytes(sample.read_bytes())
        st.store.add({"run_id": run_id, "source": "import", "status": "complete", "file": sample.name,
                      "label": "Log mẫu E4 (12 epoch đầu)",
                      "config": {"teacher": "resnet56", "student": "resnet20", "recipe": "E4", "seed": 2021}})
        run_metrics(get_run(run_id))
        return get_run(run_id)

    @app.get("/api/runs/{run_id}/notebook")
    def notebook(run_id: str):
        path = st.store.run_dir(run_id) / "kernel" / kc.NOTEBOOK_NAME
        if not path.exists():
            raise HTTPException(404, "Không có notebook")
        return FileResponse(path, filename=f"{run_id}.ipynb")

    @app.get("/api/runs/{run_id}/metrics.csv")
    def metrics_csv(run_id: str):
        df, _ = run_metrics(get_run(run_id))
        return PlainTextResponse(df.to_csv(index=False), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{run_id}_metrics.csv"'})

    # ------------------------------------------------------------ jobs

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in st.jobs:
            raise HTTPException(404, "Job không tồn tại")
        return {k: v for k, v in st.jobs[job_id].items() if k != "result" or isinstance(v, (str, int, float))}

    # ------------------------------------------------------------ TensorBoard

    def tb_running() -> bool:
        return st.tensorboard is not None and st.tensorboard.poll() is None

    def port_free(port: int) -> bool:
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", port)) != 0

    @app.get("/api/tensorboard")
    def tb_status():
        return {"running": tb_running(), "url": "http://localhost:6006" if (tb_running() or not port_free(6006)) else None,
                "logdir": str(st.runs_dir)}

    @app.post("/api/tensorboard/export")
    def tb_export():
        total, exported = 0, 0
        for run in st.store.list():
            df, _ = run_metrics(run)
            if not df.empty:
                total += metrics.export_tensorboard(df, st.store.run_dir(run["run_id"]) / "tb")
                exported += 1
        return {"runs": exported, "scalars": total}

    @app.post("/api/tensorboard/start")
    def tb_start():
        if not tb_running():
            if not port_free(6006):
                return {"running": False, "url": "http://localhost:6006", "note": "Port 6006 đang được dùng"}
            st.tensorboard = subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main", "--logdir", str(st.runs_dir), "--port", "6006",
                 "--host", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"running": True, "url": "http://localhost:6006"}

    @app.post("/api/tensorboard/stop")
    def tb_stop():
        if tb_running():
            st.tensorboard.terminate()
        st.tensorboard = None
        return {"running": False}

    # ------------------------------------------------------------ models / predict

    @app.get("/api/models")
    def available_models():
        teachers = [{"spec": f"teacher:{a}", "label": TEACHERS[a], "arch": a,
                     "ready": models.teacher_path(a, st.runs_dir / "teachers").exists()} for a in TEACHERS]
        students = []
        for run in st.store.list():
            if "best_checkpoint" in metrics.find_artifacts(st.store.run_dir(run["run_id"]) / "output"):
                cfg = run.get("config", {})
                students.append({"spec": f"run:{run['run_id']}", "arch": cfg.get("student"),
                                 "label": f"{cfg.get('student')} · {cfg.get('recipe')} · {run['run_id']}",
                                 "best": run.get("metrics", {}).get("best")})
        uploads = []
        for path in sorted((st.runs_dir / "uploads").glob("*.json")):
            uploads.append(json.loads(path.read_text()))
        return {"teachers": teachers, "students": students, "uploads": uploads}

    @app.post("/api/teachers/{arch}/download")
    def download_teacher(arch: str):
        if arch not in TEACHERS:
            raise HTTPException(404, "Teacher không hợp lệ")
        return {"job": start_job("teacher", lambda: str(models.download_teacher(arch, st.runs_dir / "teachers")))}

    @app.post("/api/checkpoints")
    async def upload_checkpoint(file: UploadFile = File(...)):
        data = await file.read()
        digest = hashlib.sha256(data).hexdigest()[:16]
        updir = st.runs_dir / "uploads"
        updir.mkdir(parents=True, exist_ok=True)
        path = updir / f"{digest}.pth"
        path.write_bytes(data)
        try:
            model, arch = models.load_model(path)
        except Exception as exc:  # noqa: BLE001
            path.unlink()
            raise HTTPException(400, f"Không đọc được checkpoint: {exc}")
        st.models[f"upload:{digest}"] = (model, arch)
        meta = {"spec": f"upload:{digest}", "arch": arch, "label": f"{arch} · {file.filename}"}
        (updir / f"{digest}.json").write_text(json.dumps(meta))
        return meta

    @app.post("/api/cifar/load")
    def cifar_load():
        if st.cifar is not None:
            return {"ready": True}

        def work():
            st.cifar = models.load_cifar100_test(st.runs_dir / "data")
            return "ok"

        return {"ready": False, "job": start_job("cifar", work)}

    @app.get("/api/cifar/sample")
    def cifar_sample(index: int | None = None):
        if st.cifar is None:
            raise HTTPException(409, "CIFAR-100 test chưa được tải")
        idx = random.randrange(len(st.cifar)) if index is None else index
        img, label = st.cifar[idx]
        return {"index": idx, "label": models.CIFAR100_CLASSES[label], "image": image_b64(img, 4)}

    @app.post("/api/predict")
    async def predict(specs: str = Form(...), file: UploadFile | None = File(None), cifar_index: int | None = Form(None)):
        truth = None
        if file is not None:
            try:
                image = Image.open(io.BytesIO(await file.read())).convert("RGB")
            except Exception:  # noqa: BLE001
                raise HTTPException(400, "File ảnh không hợp lệ")
        elif cifar_index is not None and st.cifar is not None:
            image, label = st.cifar[cifar_index]
            truth = models.CIFAR100_CLASSES[label]
        else:
            raise HTTPException(400, "Cần ảnh upload hoặc ảnh CIFAR-100")
        results = []
        for spec in [s for s in json.loads(specs) if s]:
            model, arch = load_model_spec(spec)
            preds = models.predict(model, image, k=5)
            results.append({"spec": spec, "arch": arch,
                            "predictions": [{"label": n, "prob": round(p, 5)} for n, p in preds],
                            "correct": None if truth is None else preds[0][0] == truth})
        return {"truth": truth, "input32": image_b64(image.convert("RGB").resize((32, 32), Image.BICUBIC), 6),
                "results": results}

    # ------------------------------------------------------------ frontend

    @app.middleware("http")
    async def no_cache_api(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
