import io
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from skd_app import kaggle_client as kc
from skd_app import models
from skd_app.server import create_app

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_KEY = "FAKEKEY0123456789abcdef"


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "runs")
    with TestClient(app) as c:
        c.runs_dir = tmp_path / "runs"
        yield c


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "SKD Studio" in r.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/vendor/chart.umd.js").status_code == 200


def test_session_never_echoes_key(client, monkeypatch):
    assert client.get("/api/session").json() == {"connected": False, "username": None}
    r = client.post("/api/session", json={"username": "demo-user", "key": FAKE_KEY})
    assert r.json() == {"connected": True, "username": "demo-user"} and FAKE_KEY not in r.text
    assert "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/api/session").json()["username"] == "demo-user"
    client.delete("/api/session")
    assert client.get("/api/session").json()["connected"] is False


def test_preview_and_submit_requires_kaggle(client):
    p = client.post("/api/preview", json={"teacher": "resnet110", "student": "resnet32", "recipe": "E4"}).json()
    assert p["errors"] == [] and p["env"]["TEACHER_ARCH"] == "resnet110" and p["hours"] > 1
    assert client.post("/api/preview", json={"dataset": "cifar10"}).json()["errors"]
    assert client.post("/api/runs", json={}).status_code == 401  # real push needs Kaggle credentials


def test_submit_pushes_with_session_creds_and_never_stores_key(client, monkeypatch):
    seen = {}

    def fake_push(kdir, creds):
        seen["creds"] = creds
        return kc.CliResult(True, "Kernel version 1 successfully pushed. Please check progress at "
                                  "https://www.kaggle.com/code/demo-user/skd-app-x", 0)

    monkeypatch.setattr(kc, "push", fake_push)
    client.post("/api/session", json={"username": "demo-user", "key": FAKE_KEY})
    rec = client.post("/api/runs", json={"recipe": "F1b"}).json()
    assert rec["status"] == "queued" and rec["version"] == "1" and seen["creds"].key == FAKE_KEY
    for path in client.runs_dir.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text(errors="ignore"), path
    assert FAKE_KEY not in client.get("/api/runs").text


def test_import_detects_pair_and_recipe(client, monkeypatch):
    log = (FIXTURES / "sample_kaggle_log.json").read_bytes()
    run = client.post("/api/runs/import", files={"file": ("e4.json", log)}).json()
    assert run["config"] == {"teacher": "resnet56", "student": "resnet20", "recipe": "E4"}
    d = client.get(f"/api/runs/{run['run_id']}").json()
    assert d["summary"]["epochs_done"] == 12 and len(d["series"]["epoch"]) == 12 and "epoch 012/300" in d["log_tail"]
    text = ("Teacher ResNet110 (mdistiller): top1=74.31, top5=93.0 | student arch resnet32\n"
            "epoch 001/300 | lr=0.1 | train top1=9.00 | test top1=15.00 | best=15.00 | teacher_train_top1=60.00\n")
    r110 = client.post("/api/runs/import", files={"file": ("f1b.log", text.encode())}).json()
    assert r110["config"] == {"teacher": "resnet110", "student": "resnet32", "recipe": "F1 (teacher-aware)"}
    assert client.post("/api/runs/import", files={"file": ("x.txt", b"nothing here")}).status_code == 400
    assert len(client.get("/api/runs").json()) == 2
    monkeypatch.setattr("skd_app.server.subprocess.Popen", lambda *a, **k: None)
    tb = client.post("/api/tensorboard").json()
    assert tb["runs"] == 2 and list((client.runs_dir / "tb").glob("*/events.out.tfevents.*"))


def test_refresh_status_then_download_on_complete(client, monkeypatch):
    calls = []
    monkeypatch.setattr(kc, "push", lambda kdir, c: kc.CliResult(True, "Kernel version 1 successfully pushed.", 0))
    client.post("/api/session", json={"username": "demo-user", "key": FAKE_KEY})
    rec = client.post("/api/runs", json={"smoke_test": True, "epochs": 2}).json()
    monkeypatch.setattr(kc, "status", lambda ref, c: ("running", kc.CliResult(True, "running", 0)))
    assert client.post(f"/api/runs/{rec['run_id']}/refresh").json() == {"status": "running", "job": None}

    def fake_output(ref, dest, c):
        calls.append(ref)
        (dest / "streamlined_kd").mkdir(parents=True, exist_ok=True)
        (dest / "streamlined_kd" / "history.csv").write_text("epoch,lr,train_top1,test_top1\n1,0.1,5,6\n2,0.1,7,8\n")
        return kc.CliResult(True, "ok", 0)

    monkeypatch.setattr(kc, "status", lambda ref, c: ("complete", kc.CliResult(True, "complete", 0)))
    monkeypatch.setattr(kc, "download_output", fake_output)
    res = client.post(f"/api/runs/{rec['run_id']}/refresh").json()
    assert res["status"] == "complete" and res["job"]
    for _ in range(50):
        if client.get(f"/api/jobs/{res['job']}").json()["state"] != "running":
            break
        import time; time.sleep(0.05)
    assert client.get(f"/api/jobs/{res['job']}").json()["state"] == "done" and calls == [rec["kernel_ref"]]
    assert client.get(f"/api/runs/{rec['run_id']}").json()["summary"]["best"] == 8


def test_upload_checkpoint_and_predict(client):
    buf = io.BytesIO()
    torch.save({"model": models.build_model("resnet32").state_dict()}, buf)
    meta = client.post("/api/checkpoints", files={"file": ("s.pth", buf.getvalue())}).json()
    assert meta["arch"] == "resnet32"
    img = io.BytesIO()
    Image.new("RGB", (50, 40), (10, 200, 10)).save(img, format="PNG")
    r = client.post("/api/predict", data={"specs": json.dumps([meta["spec"]])}, files={"file": ("a.png", img.getvalue())})
    body = r.json()
    assert r.status_code == 200 and len(body["results"][0]["predictions"]) == 5 and body["truth"] is None
    assert client.post("/api/predict", data={"specs": json.dumps(["teacher:resnet56"])},
                       files={"file": ("a.png", img.getvalue())}).status_code == 400  # teacher not downloaded
    assert client.post("/api/checkpoints", files={"file": ("bad.pth", b"junk")}).status_code == 400
