# SKD Studio

SKD Studio is a local web dashboard for the Streamlined KD (knowledge distillation) experiments on CIFAR-100. It lets you:

- **train on a Kaggle GPU**;
- track runs and results;
- open TensorBoard;
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
| Overview | Paper numbers vs reproduction (E4, F1b), the demo workflow, and recent runs. |
| Training | Choose the teacher (ResNet56/110), student (ResNet20/32), recipe E4/F1b and hyperparameters. The right-hand panel shows the summary, estimated GPU hours and the env vars. Use **Train on Kaggle**, or **Create notebook** (draft only). |
| Experiments | Table of runs and a comparison chart against the paper. Click a run to update its status, fetch the log, download the output, and see best/final/TB10, curves and the log. **Load sample log** gives you a demo without Kaggle. |
| TensorBoard | Export event files and start TensorBoard on port 6006, embedded in the page. |
| Prediction | Load the teacher (mdistiller, downloaded automatically) and/or a student (from a run or an uploaded `.pth`), drop in an image or pick a random CIFAR-100 image, then compare the top-5 predictions. |

## Notes

- **Kaggle API key:** use the **Connect Kaggle** button in the top right. You can type the key or drop in `kaggle.json`. The key stays only in the local server's memory; it is never written to disk or stored in the history.
- **Local data:** runs, checkpoints and datasets are kept in `app/runs/` (gitignored).
- **API docs:** http://localhost:8000/api/docs
- **Tests:** `uv run pytest -q`
