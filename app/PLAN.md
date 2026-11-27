# SKD Demo App — plan

A demo app for this repo. It does three things:

- submit a knowledge-distillation (KD) training run of `kaggle/streamlined_kd_cifar100.py` to Kaggle;
- track run history, metrics and TensorBoard curves;
- load a teacher or student checkpoint and run predictions.

The app is fully written. A scheduled task fires every 10 minutes for review / polish passes; see *Status* and *Review log*.

## Stack

| Area | Choice |
|---|---|
| Dependencies | `uv`, with `app/pyproject.toml` and `uv.lock` committed. Run with `cd app && uv sync && uv run streamlit run skd_app/main.py`. |
| UI | Streamlit, with 4 tabs: **Train**, **Runs / History**, **TensorBoard**, **Predict**. |
| Kaggle | The official `kaggle` CLI, called via `subprocess`. |
| Plots | `plotly` in the UI, plus TensorBoard event files written with `torch.utils.tensorboard`. |
| Models | `torch` and `torchvision`, CPU build, for predictions. |

## Features

1. **Kaggle credentials.**
   - The user enters a Kaggle username and API key in the UI, or uploads `kaggle.json`.
   - Credentials are held only in `st.session_state`.
   - They reach the CLI only through the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables of the subprocess.
   - They are **never** written to disk, logs or run records. A test enforces this.
   - A button checks the credentials with `kaggle kernels list --mine --page-size 1` and shows quota where available.
2. **Train form.**
   - **Teacher:** `resnet56` (default) or `resnet110`. These are mdistiller checkpoints already known to the script.
   - **Student:** `resnet20` (default) or `resnet32`.
   - **Dataset:** CIFAR-100. It is the only dataset with pretrained teachers; other options are shown disabled with a note.
   - **Hyperparameters:**
     - epochs, learning rate, LR milestones, batch size, temperature, seed;
     - CE weight and KD weight;
     - recipe: `E4` (SKD reproduction) or `F1b` (`SAMPLE_WEIGHT_MODE`/`SW_*`, KD 0.6 when the teacher is correct, else 0.5);
     - a "smoke test" toggle that sets `LIMIT_TRAIN_BATCHES` and `LIMIT_EVAL_BATCHES`.
   - Every option maps to the script's existing environment variables. The script itself is not changed.
3. **Submit.**
   - Build a kernel folder in `app/runs/<run_id>/kernel/`. It contains the training script and a notebook whose first cell sets the chosen environment variables. Reuse `kaggle/make_variant_notebook.py` / `build_notebook.py` where possible.
   - Write `kernel-metadata.json` with GPU on, internet on, and the **pinned docker image from `kaggle/kernel-metadata.template.json` (never changed)**. The slug is `skd-app-<teacher>-<student>-<recipe>-<timestamp>`.
   - Run `kaggle kernels push -p <dir>`.
4. **Runs / History.**
   - Run records live in `app/runs/runs.json`, which is gitignored. Each record holds id, slug, Kaggle username, config, submitted_at, status and metrics.
   - A refresh button calls `kaggle kernels status`.
   - Logs come from `kaggle kernels logs`, or from the downloaded output log.
   - When a run is complete, `kaggle kernels output` downloads the log, history and checkpoint into `app/runs/<run_id>/output/`.
   - Per-epoch metrics are parsed from the `epoch NNN/EEE | lr=... | ...` lines or the saved `history`.
   - The tab shows **best, final and TB10** (mean top-1 over the last 10 epochs), plus curves for loss, top-1 and LR, and a comparison table across runs.
5. **TensorBoard.**
   - Convert each run's history to event files in `app/runs/<run_id>/tb/`.
   - Show the command `uv run tensorboard --logdir app/runs`, with an optional embedded iframe when it is running locally.
6. **Predict.**
   - Choose a model: a teacher (mdistiller checkpoint path or uploaded `.pth`) or a student (checkpoint from a run's output, or an uploaded `.pth`).
   - Upload an image. It is resized to 32×32 and normalized with CIFAR-100 statistics, and the top-5 classes with probabilities are returned.
   - A "random CIFAR-100 test images" mode compares teacher and student predictions side by side.
   - Model builders are imported from, or copied minimally from, `kaggle/streamlined_kd_cifar100.py`, so architectures and state_dict keys match.

## Status

The whole app was written in one pass on 2026-11-27, at the user's request; milestones M1–M7 were collapsed into it.

Checks run on that pass:

- 19 tests pass (`uv run pytest -q`).
- Headless start returns HTTP 200.
- A Playwright check of the Train, History (import log) and Predict tabs (real mdistiller ResNet56 teacher) found no exceptions.

Later scheduled passes are **review / polish only**: fix bugs, smooth UX rough edges, or add ONE small item from *Ideas*. Keep it demo-friendly.

## Review log

(one line per scheduled review pass; the series stops after 6 entries)

## Ideas (small, demo-friendly)

- Comparison chart in History: the y-axis lower bound is ignored (`yaxis_range=[min, None]`), so bars start at 0. Zoom to the data range.
- "Demo data" button that imports the bundled `tests/fixtures/sample_kaggle_log.json` in one click.
- Auto-refresh toggle in History (poll status every N minutes while a run is running).
- Show Kaggle GPU quota in the sidebar after the credential check (`kaggle` CLI does not expose it; skip if not possible).
- Predict: grid of 8 random CIFAR-100 test images with teacher-vs-student agreement.
- Export a run's metrics and config as a small Markdown report for slides.

## Rules for the scheduled task

- Never print, commit or write any Kaggle credentials. Tests use fake values only.
- **Do not submit real Kaggle trainings** from the scheduled task. Only the user does that from the app.
- Do not change the pinned docker image or the training script's defaults.
- `git add` specific files only. Never commit `.pth`, `app/runs/` or `.venv/`.
- If a milestone fails, note the blocker in this file, commit, and report it. Do not skip ahead silently.
