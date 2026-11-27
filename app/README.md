# SKD Demo App

This is a Streamlit app for the Streamlined KD (knowledge distillation) experiments on CIFAR-100. It lets you:

- set up a training run and **train it on a Kaggle GPU**;
- track **run history**, best / final / TB10 scores and training curves;
- export **TensorBoard** files;
- run **predictions** with a teacher or student checkpoint.

## Run on localhost

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.10 or newer.

```bash
cd app
uv sync
uv run streamlit run skd_app/main.py
```

Then open http://localhost:8501.

## Using the app

| Tab | What it does |
|---|---|
| 🚀 Train | Enter your Kaggle username and API key in the sidebar, or upload `kaggle.json`. Pick the teacher (ResNet56 / ResNet110), the student (ResNet20 / ResNet32), the recipe (E4 / F1b) and hyperparameters (learning rate, epochs, batch size, T, seed). **Train trên Kaggle** pushes the kernel to Kaggle. **Smoke test** runs only a few batches so you can check the pipeline quickly. |
| 📜 History | Lists your runs. Buttons update the status, fetch the log and download the output (history.csv, checkpoint). The tab shows best / final / TB10 scores, curves, and a comparison against the paper. **Import log** shows results from an existing log without using Kaggle. |
| 📈 TensorBoard | Exports TensorBoard event files and starts TensorBoard inside the app. You can also run `uv run tensorboard --logdir runs` yourself. |
| 🔮 Predict | Load a teacher (mdistiller pretrained, downloaded automatically) and/or a student (from a finished run, or an uploaded `.pth`). Then upload an image, or pick a random CIFAR-100 test image, to see the top-5 predictions. |

## Notes

- **API key:** the key is kept only in the browser session's memory. It is never written to disk or saved in `runs/`.
- **Local files:** run data, checkpoints and datasets are stored in `app/runs/`, which is gitignored.
- **CPU-only torch on Linux** (optional, saves disk space): `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
- **Tests:** `uv run pytest -q`.
