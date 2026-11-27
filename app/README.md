# SKD Studio

SKD Studio is a local web dashboard for the Streamlined KD (knowledge distillation) experiments on CIFAR-100. It lets you:

- **train on a Kaggle GPU**;
- see the results next to the paper;
- run predictions with a teacher or student.

The backend is FastAPI and the frontend is hand-written HTML/CSS/JS. No Node or npm is needed: Chart.js is bundled in `skd_app/static/vendor/`.

## Run on localhost

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.10 or newer.

```bash
cd app
uv sync
uv run python -m skd_app
```

The browser opens **http://localhost:8000** automatically. Options:

- `--port 9000` to use a different port;
- `--no-browser` to skip opening a browser tab.

## Pages

| Page | What it does |
|---|---|
| Huấn luyện | Pick the model pair (ResNet56→20 or ResNet110→32), the recipe (E4 / F1b) and epochs / learning rate / seed, then press **Train trên Kaggle**. |
| Kết quả | One row per run with best / TB10 vs the paper. Click a run for the curve and log. **Cập nhật** checks Kaggle and downloads the results once the run is finished. **Import log có sẵn** adds a run from an existing log (pair and recipe are detected automatically). **Mở TensorBoard** opens the same curves in TensorBoard. |
| Dự đoán | Choose the teacher and/or a student, drop in an image (or a random CIFAR-100 image), and compare their top-5 predictions. |

## Notes

- **Kaggle API key:** use the **Connect Kaggle** button in the top right. You can type the key or drop in `kaggle.json`. The key stays only in the local server's memory; it is never written to disk or stored in the history.
- **Local data:** runs, checkpoints and datasets are kept in `app/runs/` (gitignored).
- **API docs:** http://localhost:8000/api/docs
- **Tests:** `uv run pytest -q`
