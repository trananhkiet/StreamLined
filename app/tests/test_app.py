import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from skd_app import kaggle_client as kc
from skd_app import metrics, models
from skd_app.config import DOCKER_IMAGE, REPO_DIR, TRAIN_SCRIPT, TrainConfig
from skd_app.runs import RunStore

FIXTURES = Path(__file__).parent / "fixtures"
FAKE = kc.Credentials("demo-user", "FAKEKEY0123456789abcdef")


# ---------------------------------------------------------------- config


def test_defaults_valid_and_f1b_env_matches_repo_runs():
    cfg = TrainConfig()
    assert cfg.validate() == []
    env = cfg.to_env()
    # Same overrides as kaggle/experiments/r110r32_f1b_seed2023.ipynb (apart from arch/seed)
    assert env["TEACHER_SOURCE"] == "mdistiller"
    assert env["EPOCHS"] == "300" and env["LR_MILESTONES"] == "100,188,250"
    assert env["SAMPLE_WEIGHT_MODE"] == "teacher_correct" and env["SW_DISTRUST"] == "0.5"
    assert "KD_WEIGHT" not in env


def test_every_env_key_exists_in_training_script():
    text = TRAIN_SCRIPT.read_text()
    cfg = TrainConfig(recipe="E4", smoke_test=True)
    for key in cfg.to_env():
        assert f'os.getenv("{key}"' in text, key


def test_validation_errors():
    bad = TrainConfig(dataset="cifar10", epochs=0, lr_milestones="200,100", batch_size=48)
    errs = bad.validate()
    assert len(errs) == 4


def test_slug_short_and_unique_parts():
    slug = TrainConfig(teacher="resnet110", student="resnet32", smoke_test=True).slug("1127-0715")
    assert slug == "skd-app-r110-r32-f1b-s2022-smoke-1127-0715" and len(slug) <= 50


# ---------------------------------------------------------------- kaggle client


def test_docker_image_matches_repo_pin():
    meta = json.loads((REPO_DIR / "kaggle/experiments/r110r32_f1b_seed2023_kernel-metadata.json").read_text())
    assert meta["docker_image"] == DOCKER_IMAGE


def test_build_kernel_dir(tmp_path):
    cfg = TrainConfig(teacher="resnet110", student="resnet32", seed=7)
    kdir = kc.build_kernel_dir(tmp_path / "k", "demo-user", "skd-app-x", cfg)
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    assert meta["id"] == "demo-user/skd-app-x" and meta["enable_gpu"] and meta["docker_image"] == DOCKER_IMAGE
    assert meta["dataset_sources"] == ["fedesoriano/cifar100"]
    nb = json.loads((kdir / kc.NOTEBOOK_NAME).read_text())
    env_cell = "".join(nb["cells"][1]["source"])
    assert "os.environ['TEACHER_ARCH'] = 'resnet110'" in env_cell and "os.environ['SEED'] = '7'" in env_cell
    assert "".join(nb["cells"][2]["source"]) == TRAIN_SCRIPT.read_text()
    assert FAKE.key not in (kdir / kc.NOTEBOOK_NAME).read_text()


def test_cli_passes_creds_only_via_env_and_redacts(monkeypatch):
    seen = {}

    def fake_run(cmd, env, **kw):
        seen["cmd"], seen["env"] = cmd, env
        return subprocess.CompletedProcess(cmd, 0, stdout=f"echo {FAKE.key} ok", stderr="")

    monkeypatch.setattr(kc.subprocess, "run", fake_run)
    res = kc.run_cli(["kernels", "list"], FAKE)
    assert res.ok and FAKE.key not in res.output and "***" in res.output
    assert FAKE.key not in " ".join(seen["cmd"])
    assert seen["env"]["KAGGLE_KEY"] == FAKE.key and seen["env"]["KAGGLE_USERNAME"] == "demo-user"
    assert not Path(seen["env"]["KAGGLE_CONFIG_DIR"]).exists()  # temp config dir removed afterwards
    assert FAKE.key not in repr(FAKE) and FAKE.key not in str(FAKE)
    assert "KAGGLE_API_TOKEN" not in seen["env"]


def test_new_style_token_goes_to_api_token_env(monkeypatch):
    seen = {}
    token = kc.Credentials("demo-user", "KGAT_fake0123456789abcdef0123456789")
    monkeypatch.setattr(kc.subprocess, "run", lambda cmd, env, **kw: seen.update(env=env) or
                        subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    kc.run_cli(["kernels", "list"], token)
    assert seen["env"]["KAGGLE_API_TOKEN"] == token.key and "KAGGLE_KEY" not in seen["env"]


def test_parse_status_and_push():
    assert kc.parse_status('demo-user/skd-app-x has status "KernelWorkerStatus.RUNNING"') == "running"
    assert kc.parse_status('demo-user/x has status "complete"') == "complete"
    info = kc.parse_push("Kernel version 3 successfully pushed.  Please check progress at "
                         "https://www.kaggle.com/code/demo-user/skd-app-x")
    assert info == {"version": "3", "url": "https://www.kaggle.com/code/demo-user/skd-app-x"}


def test_kaggle_json():
    c = kc.Credentials.from_kaggle_json('{"username": "u", "key": "k"}')
    assert c.username == "u" and c.is_complete()


# ---------------------------------------------------------------- runs + metrics


def test_run_store_roundtrip_and_refuses_secrets(tmp_path):
    store = RunStore(tmp_path)
    store.add({"run_id": "a", "config": {"teacher": "resnet56"}})
    store.update("a", status="running")
    assert store.get("a")["status"] == "running"
    with pytest.raises(ValueError):
        store.add({"run_id": "b", "config": {"key": "x"}})
    assert [r["run_id"] for r in store.list()] == ["a"]


def test_parse_kaggle_log_fixture():
    df, info = metrics.parse_log((FIXTURES / "sample_kaggle_log.json").read_text())
    assert list(df["epoch"]) == list(range(1, 13))
    assert df.loc[0, "test_top1"] == pytest.approx(17.34) and info["teacher_top1"] == pytest.approx(72.41)
    s = metrics.summarize(df)
    assert s["epochs_done"] == 12 and s["epochs_total"] == 300
    assert s["best"] == round(df["test_top1"].max(), 2)
    assert s["tb10"] == round(df["test_top1"].tail(10).mean(), 2)


def test_parse_plain_text_log():
    text = "epoch 001/2 | lr=0.1 | train top1=1.00 | test top1=2.00 | best=2.00\n" \
           "epoch 002/2 | lr=0.01 | train top1=3.00 | test top1=1.50 | best=2.00 | gradnorm ce/kd/bc=1/2/3\n"
    s = metrics.summarize(metrics.parse_log(text)[0])
    assert s == {"epochs_done": 2, "epochs_total": 2, "best": 2.0, "best_epoch": 1, "final": 1.5, "tb10": 1.75}


def test_load_run_metrics_prefers_history_and_exports_tensorboard(tmp_path):
    out = tmp_path / "output" / "streamlined_kd"
    out.mkdir(parents=True)
    pd.DataFrame({"epoch": [1, 2, 3], "lr": [0.1] * 3, "train_loss": [3, 2, 1], "train_top1": [10, 20, 30],
                  "test_top1": [11, 22, 21]}).to_csv(out / "history.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"teacher_metrics": {"top1": 74.31}, "config": {"epochs": 3}}))
    df, info = metrics.load_run_metrics(tmp_path / "output")
    assert len(df) == 3 and info["teacher_top1"] == 74.31
    assert metrics.summarize(df)["best"] == 22
    n = metrics.export_tensorboard(df, tmp_path / "tb")
    assert n == 12 and list((tmp_path / "tb").glob("events.out.tfevents.*"))


# ---------------------------------------------------------------- models


@pytest.mark.parametrize("arch", ["resnet20", "resnet32", "resnet56", "resnet110"])
def test_checkpoint_formats_and_arch_inference(tmp_path, arch):
    model = models.build_model(arch)
    state = model.state_dict()
    for i, wrapped in enumerate([state, {"model": state, "optimizer": {}}, {"student": state, "epoch": 3},
                                 {f"module.{k}": v for k, v in state.items()}]):
        path = tmp_path / f"{i}.pth"
        torch.save(wrapped, path)
        loaded, got = models.load_model(path)
        assert got == arch
    loaded, _ = models.load_model(path.read_bytes())
    assert not loaded.training


def test_predict_top5_probabilities():
    model = models.build_model("resnet20").eval()
    preds = models.predict(model, Image.new("RGB", (64, 48), (200, 30, 30)))
    assert len(preds) == 5 and all(name in models.CIFAR100_CLASSES for name, _ in preds)
    probs = [p for _, p in preds]
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1.0001


def test_model_keys_match_training_script():
    """Our copied ResNet must produce exactly the training script's state_dict keys and shapes."""
    import ast

    src = TRAIN_SCRIPT.read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in {"conv3x3", "BasicBlock", "CifarResNet"}]
    ns: dict = {}
    exec("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
         + "\n".join(ast.unparse(n) for n in keep), ns)
    for depth, arch in [(20, "resnet20"), (110, "resnet110")]:
        ref = {k: v.shape for k, v in ns["CifarResNet"](depth).state_dict().items()}
        ours = {k: v.shape for k, v in models.build_model(arch).state_dict().items()}
        assert ref == ours
