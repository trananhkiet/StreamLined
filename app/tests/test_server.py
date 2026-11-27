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


def test_preview_and_draft_run(client):
    p = client.post("/api/preview", json={"teacher": "resnet110", "student": "resnet32", "recipe": "E4"}).json()
    assert p["errors"] == [] and p["env"]["TEACHER_ARCH"] == "resnet110" and p["hours"] > 1
    bad = client.post("/api/preview", json={"dataset": "cifar10"}).json()
    assert bad["errors"]
    rec = client.post("/api/runs?dry_run=true", json={"smoke_test": True}).json()
    assert rec["status"] == "draft"
    assert client.get(f"/api/runs/{rec['run_id']}/notebook").status_code == 200
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


def test_import_sample_and_detail(client):
    run = client.post("/api/runs/import-sample").json()
    d = client.get(f"/api/runs/{run['run_id']}").json()
    assert d["summary"]["epochs_done"] == 12 and len(d["series"]["epoch"]) == 12
    assert client.get(f"/api/runs/{run['run_id']}/metrics.csv").text.startswith("epoch")
    up = client.post("/api/runs/import", files={"file": ("x.log", (FIXTURES / "sample_kaggle_log.json").read_bytes())},
                     data={"teacher": "resnet110", "student": "resnet32", "recipe": "F1b"})
    assert up.status_code == 200
    bad = client.post("/api/runs/import", files={"file": ("x.txt", b"nothing here")})
    assert bad.status_code == 400
    assert len(client.get("/api/runs").json()) == 2
    assert client.post("/api/tensorboard/export").json()["runs"] == 2


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
