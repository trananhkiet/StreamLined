"""SKD demo app — run with:  cd app && uv run streamlit run skd_app/main.py"""

from __future__ import annotations

import hashlib
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skd_app import kaggle_client as kc  # noqa: E402
from skd_app import metrics, models  # noqa: E402
from skd_app.config import (  # noqa: E402
    DATASETS, RECIPES, REFERENCE, RUNS_DIR, STUDENTS, TEACHERS, TrainConfig, estimated_hours,
)
from skd_app.runs import RunStore, now_iso  # noqa: E402

st.set_page_config(page_title="SKD Demo", page_icon="🧪", layout="wide")
store = RunStore()
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


# ---------------------------------------------------------------- helpers


def creds() -> kc.Credentials | None:
    c = st.session_state.get("creds")
    return c if c and c.is_complete() else None


def run_label(r: dict) -> str:
    cfg = r.get("config", {})
    return (f"{r['run_id']} · {cfg.get('teacher', '?')}→{cfg.get('student', '?')} · {cfg.get('recipe', '?')} "
            f"· seed {cfg.get('seed', '?')} · {r.get('status', '?')}")


def refresh_metrics(run: dict, log_text: str | None = None) -> tuple[pd.DataFrame, dict]:
    out_dir = store.run_dir(run["run_id"]) / "output"
    if log_text is None:
        log_file = store.run_dir(run["run_id"]) / "kernel_log.txt"
        log_text = log_file.read_text(errors="ignore") if log_file.exists() else None
    df, info = metrics.load_run_metrics(out_dir, log_text)
    summary = metrics.summarize(df)
    if summary or info:
        store.update(run["run_id"], metrics={**summary, **{k: v for k, v in info.items() if v is not None}})
    return df, summary


def curve_figure(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if "test_top1" in df:
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["test_top1"], name="test top-1", line=dict(width=2.5)))
    if "train_top1" in df:
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_top1"], name="train top-1", line=dict(dash="dot")))
    if "lr" in df:
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["lr"], name="learning rate", yaxis="y2",
                                 line=dict(color="gray", width=1)))
    fig.update_layout(
        title=title, height=420, xaxis_title="epoch", yaxis_title="top-1 (%)",
        yaxis2=dict(title="lr", overlaying="y", side="right", type="log", showgrid=False),
        legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10),
    )
    return fig


def metric_cards(summary: dict, info: dict, cfg: dict) -> None:
    ref = REFERENCE.get((cfg.get("teacher"), cfg.get("student")), {})
    c = st.columns(5)
    c[0].metric("Best top-1", f"{summary.get('best', '–')}", help=f"epoch {summary.get('best_epoch', '–')}")
    c[1].metric("Final", f"{summary.get('final', '–')}")
    c[2].metric("TB10", f"{summary.get('tb10', '–')}", help="Trung bình top-1 của 10 epoch cuối")
    c[3].metric("Teacher top-1", f"{info.get('teacher_top1', '–')}")
    if ref and "best" in summary:
        c[4].metric("So với paper SKD", f"{summary['best']:.2f}", f"{summary['best'] - ref['paper_skd']:+.2f} vs {ref['paper_skd']}")
    st.caption(f"Tiến độ: {summary.get('epochs_done', 0)}/{summary.get('epochs_total', cfg.get('epochs', '?'))} epoch")


# ---------------------------------------------------------------- sidebar: credentials


with st.sidebar:
    st.header("🔑 Kaggle API")
    st.caption("Key chỉ giữ trong bộ nhớ phiên này — không ghi ra file, không lưu vào lịch sử.")
    mode = st.radio("Cách nhập", ["Username + key", "Upload kaggle.json"], horizontal=True, label_visibility="collapsed")
    if mode == "Username + key":
        username = st.text_input("Username", value=getattr(st.session_state.get("creds"), "username", ""))
        key = st.text_input("API key", type="password")
        if st.button("Lưu", use_container_width=True) and username and key:
            st.session_state.creds = kc.Credentials(username.strip(), key.strip())
    else:
        up = st.file_uploader("kaggle.json", type="json")
        if up is not None:
            try:
                st.session_state.creds = kc.Credentials.from_kaggle_json(up.getvalue())
            except (KeyError, ValueError):
                st.error("File kaggle.json không hợp lệ")
    if creds():
        st.success(f"Đang dùng tài khoản **{creds().username}**")
        col_a, col_b = st.columns(2)
        if col_a.button("Kiểm tra", use_container_width=True):
            res = kc.verify(creds())
            (st.success("Kết nối OK") if res.ok else st.error(res.output[-500:]))
        if col_b.button("Xoá key", use_container_width=True):
            st.session_state.pop("creds", None)
            st.rerun()
    else:
        st.info("Chưa có key — vẫn xem lịch sử, import log và predict được.")
    st.divider()
    st.caption("Repo: Streamlined KD (CVPR 2026) · CIFAR-100")

st.title("🧪 Streamlined KD — demo")
tab_train, tab_runs, tab_tb, tab_pred = st.tabs(["🚀 Train", "📜 Lịch sử & kết quả", "📈 TensorBoard", "🔮 Predict"])


# ---------------------------------------------------------------- Train


with tab_train:
    st.subheader("Cấu hình một lần train trên Kaggle (GPU T4)")
    left, right = st.columns([3, 2])
    with left:
        c1, c2, c3 = st.columns(3)
        teacher = c1.selectbox("Teacher", list(TEACHERS), format_func=TEACHERS.get)
        student = c2.selectbox("Student", list(STUDENTS), format_func=STUDENTS.get)
        dataset = c3.selectbox("Dataset", list(DATASETS), format_func=lambda d: DATASETS[d][0])
        if not DATASETS[dataset][1]:
            st.warning("Dataset này chưa có teacher pretrained — hiện chỉ CIFAR-100 chạy được.")
        recipe = st.radio("Recipe (loss)", list(RECIPES), format_func=RECIPES.get, index=1, horizontal=True)
        c4, c5, c6 = st.columns(3)
        epochs = c4.number_input("Epochs", 1, 400, 300)
        lr = c5.number_input("Learning rate", 0.001, 1.0, 0.1, step=0.01, format="%.3f")
        milestones = c6.text_input("LR giảm ×0.1 tại epoch", "100,188,250")
        c7, c8, c9 = st.columns(3)
        batch = c7.selectbox("Batch size", [32, 64, 128, 256], index=1)
        temp = c8.number_input("Temperature T", 0.5, 20.0, 4.0, step=0.5)
        seed = c9.number_input("Seed", 0, 99999, 2022)
        c10, c11, c12 = st.columns(3)
        ce_w = c10.number_input("CE weight", 0.0, 2.0, 0.5, step=0.1, disabled=recipe == "F1b")
        kd_w = c11.number_input("KD weight", 0.0, 2.0, 0.5, step=0.1, disabled=recipe == "F1b",
                                help="F1b tự chọn w = 0.6/0.5 theo từng mẫu")
        smoke = c12.toggle("Smoke test", help="Chỉ 20 batch train / 5 batch eval mỗi epoch — để kiểm tra nhanh pipeline")
        if smoke and epochs > 5:
            st.caption("Smoke test: nên đặt epochs = 2.")

    cfg = TrainConfig(teacher=teacher, student=student, dataset=dataset, recipe=recipe, epochs=int(epochs),
                      learning_rate=float(lr), lr_milestones=milestones, batch_size=int(batch),
                      temperature=float(temp), seed=int(seed), ce_weight=float(ce_w), kd_weight=float(kd_w),
                      smoke_test=smoke)
    errors = cfg.validate()
    with right:
        ref = REFERENCE.get((teacher, student))
        if ref:
            st.markdown(f"**Tham chiếu {teacher}→{student}:** paper SKD **{ref['paper_skd']}** · KD {ref['paper_kd']} · "
                        f"tái lập E4 {ref['ours_e4']} · F1b {ref['ours_f1b']}")
        st.markdown(f"**Ước tính thời gian:** ~{estimated_hours(cfg):.1f} giờ GPU T4 (quota Kaggle 30 giờ/tuần)")
        st.markdown("**Biến môi trường gửi cho script:**")
        st.dataframe(pd.DataFrame(cfg.to_env().items(), columns=["ENV", "value"]), hide_index=True, height=300)

    for e in errors:
        st.error(e)
    b1, b2 = st.columns(2)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M")
    owner = creds().username if creds() else "your-kaggle-username"
    slug = cfg.slug(stamp)

    if b2.button("📝 Chỉ tạo notebook (không chạy)", disabled=bool(errors), use_container_width=True):
        run_id = f"{stamp}-{recipe.lower()}-dry"
        kdir = kc.build_kernel_dir(store.run_dir(run_id) / "kernel", owner, slug, cfg)
        st.download_button("⬇️ Tải notebook", (kdir / kc.NOTEBOOK_NAME).read_bytes(), file_name=f"{slug}.ipynb")
        st.download_button("⬇️ Tải kernel-metadata.json", (kdir / "kernel-metadata.json").read_bytes(),
                           file_name="kernel-metadata.json")
        st.caption(f"Đã tạo tại `{kdir}` — có thể tự chạy: `kaggle kernels push -p {kdir}`")

    if b1.button("🚀 Train trên Kaggle", type="primary", disabled=bool(errors) or not creds(), use_container_width=True):
        run_id = f"{stamp}-{recipe.lower()}-s{seed}"
        kdir = kc.build_kernel_dir(store.run_dir(run_id) / "kernel", owner, slug, cfg)
        with st.spinner("Đang push kernel lên Kaggle..."):
            res = kc.push(kdir, creds())
        info = kc.parse_push(res.output)
        record = {
            "run_id": run_id, "source": "kaggle", "kaggle_user": owner, "kernel_ref": f"{owner}/{slug}",
            "url": info.get("url", f"https://www.kaggle.com/code/{owner}/{slug}"), "version": info.get("version"),
            "config": {**cfg.__dict__, "env": cfg.to_env()}, "status": "queued" if res.ok else "push_failed",
            "push_output": res.output[-2000:],
        }
        store.add(record)
        if res.ok:
            st.success(f"Đã gửi! Theo dõi ở tab **Lịch sử** hoặc trên Kaggle: {record['url']}")
            st.balloons()
        else:
            st.error("Push thất bại:")
            st.code(res.output[-2000:])
    if not creds():
        st.caption("👈 Nhập Kaggle API key ở thanh bên để bật nút Train.")


# ---------------------------------------------------------------- Runs / history


with tab_runs:
    runs = store.list()
    with st.expander("📥 Import log / history.csv có sẵn (demo không cần Kaggle)", expanded=not runs):
        up = st.file_uploader("Kaggle log (.log/.json/.txt) hoặc history.csv", type=["log", "json", "txt", "csv"])
        c1, c2, c3 = st.columns(3)
        imp_teacher = c1.selectbox("Teacher", list(TEACHERS), key="imp_t")
        imp_student = c2.selectbox("Student", list(STUDENTS), key="imp_s")
        imp_recipe = c3.selectbox("Recipe", list(RECIPES), key="imp_r")
        if st.button("Import") and up is not None:
            run_id = f"import-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}"
            out_dir = store.run_dir(run_id) / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = "history.csv" if up.name.endswith(".csv") else "imported.log"
            (out_dir / name).write_bytes(up.getvalue())
            store.add({"run_id": run_id, "source": "import", "status": "complete", "file": up.name,
                       "config": {"teacher": imp_teacher, "student": imp_student, "recipe": imp_recipe}})
            refresh_metrics(store.get(run_id))
            st.rerun()

    if not runs:
        st.info("Chưa có run nào. Hãy train ở tab đầu hoặc import một log.")
    else:
        table = pd.DataFrame([{
            "run": r["run_id"], "teacher": r.get("config", {}).get("teacher"), "student": r.get("config", {}).get("student"),
            "recipe": r.get("config", {}).get("recipe"), "seed": r.get("config", {}).get("seed"),
            "status": r.get("status"), "best": r.get("metrics", {}).get("best"), "final": r.get("metrics", {}).get("final"),
            "TB10": r.get("metrics", {}).get("tb10"), "epochs": r.get("metrics", {}).get("epochs_done"),
            "submitted": r.get("submitted_at"),
        } for r in runs])
        st.dataframe(table, hide_index=True, use_container_width=True)

        scored = table.dropna(subset=["best"])
        if len(scored) >= 1:
            fig = go.Figure()
            for col in ("best", "final", "TB10"):
                fig.add_trace(go.Bar(x=scored["run"], y=scored[col], name=col))
            for (t, s), ref in REFERENCE.items():
                if ((scored["teacher"] == t) & (scored["student"] == s)).any():
                    fig.add_hline(y=ref["paper_skd"], line_dash="dash", annotation_text=f"paper SKD {t}→{s}: {ref['paper_skd']}")
            fig.update_layout(barmode="group", height=360, yaxis_range=[max(0, scored[["best", "final", "TB10"]].min().min() - 2), None],
                              title="So sánh các run", margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        chosen = st.selectbox("Chi tiết run", runs, format_func=run_label)
        run = store.get(chosen["run_id"])
        cfg_d = run.get("config", {})
        if run.get("url"):
            st.markdown(f"🔗 [{run.get('kernel_ref')}]({run['url']})")
        is_kaggle = run.get("source") == "kaggle"
        a1, a2, a3, a4 = st.columns(4)
        if a1.button("🔄 Cập nhật trạng thái", disabled=not (is_kaggle and creds()), use_container_width=True):
            state, res = kc.status(run["kernel_ref"], creds())
            store.update(run["run_id"], status=state, status_checked_at=now_iso())
            (st.toast(f"Trạng thái: {state}") if res.ok else st.error(res.output[-500:]))
            run = store.get(run["run_id"])
        if a2.button("📄 Lấy log", disabled=not (is_kaggle and creds()), use_container_width=True):
            with st.spinner("Đang tải log..."):
                res = kc.logs(run["kernel_ref"], creds())
            if res.ok:
                (store.run_dir(run["run_id"]) / "kernel_log.txt").write_text(res.output)
            else:
                st.error(res.output[-500:])
        if a3.button("⬇️ Tải output", disabled=not (is_kaggle and creds()), use_container_width=True,
                     help="history.csv, summary.json, checkpoint best của student"):
            with st.spinner("Đang tải output từ Kaggle (có thể vài phút)..."):
                res = kc.download_output(run["kernel_ref"], store.run_dir(run["run_id"]) / "output", creds())
            (st.success("Đã tải output") if res.ok else st.error(res.output[-800:]))
        if a4.button("🗑️ Xoá khỏi lịch sử", use_container_width=True):
            store.delete(run["run_id"])
            st.rerun()

        df, summary = refresh_metrics(run)
        run = store.get(run["run_id"])
        metric_cards(summary, run.get("metrics", {}), cfg_d)
        if not df.empty:
            st.plotly_chart(curve_figure(df, f"{cfg_d.get('teacher')} → {cfg_d.get('student')} · {cfg_d.get('recipe')}"),
                            use_container_width=True)
            extra = [c for c in df.columns if c.startswith("train_") and c not in ("train_top1",)]
            if extra:
                pick = st.multiselect("Thêm chỉ số từ history.csv", extra, default=[c for c in extra if "loss" in c][:3])
                if pick:
                    st.line_chart(df.set_index("epoch")[pick], height=260)
            st.download_button("⬇️ Tải bảng metric (CSV)", df.to_csv(index=False).encode(), f"{run['run_id']}_metrics.csv")
        else:
            st.info("Chưa có metric — bấm **Lấy log** (khi đang chạy) hoặc **Tải output** (khi đã xong).")
        with st.expander("Cấu hình & log thô"):
            st.json({k: v for k, v in run.items() if k != "push_output"})
            log_file = store.run_dir(run["run_id"]) / "kernel_log.txt"
            if log_file.exists():
                text = metrics._log_text(log_file.read_text(errors="ignore"))
                st.code(text[-4000:] or "(trống)")


# ---------------------------------------------------------------- TensorBoard


@st.cache_resource
def tb_process() -> dict:
    return {}


def free_port(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


with tab_tb:
    st.subheader("TensorBoard")
    st.markdown("Xuất lịch sử train của mọi run thành event file trong `app/runs/<run>/tb/`, rồi mở TensorBoard.")
    if st.button("Xuất TensorBoard cho tất cả run"):
        total = 0
        for r in store.list():
            df, _ = refresh_metrics(r)
            if not df.empty:
                total += metrics.export_tensorboard(df, store.run_dir(r["run_id"]) / "tb")
        st.success(f"Đã ghi {total} scalar.")
    proc_box = tb_process()
    port = st.number_input("Port", 1024, 65535, 6006)
    c1, c2 = st.columns(2)
    if c1.button("▶️ Khởi động TensorBoard"):
        if proc_box.get("proc") and proc_box["proc"].poll() is None:
            st.info("TensorBoard đang chạy.")
        elif not free_port(int(port)):
            st.warning(f"Port {port} đang bận — có thể TensorBoard đã chạy sẵn.")
        else:
            proc_box["proc"] = subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main", "--logdir", str(RUNS_DIR), "--port", str(int(port)),
                 "--host", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.success(f"Đã khởi động ở http://localhost:{port}")
    if c2.button("⏹️ Dừng"):
        if proc_box.get("proc"):
            proc_box["proc"].terminate()
            proc_box.pop("proc")
    st.code(f"uv run tensorboard --logdir {RUNS_DIR.relative_to(RUNS_DIR.parent.parent)}", language="bash")
    if not free_port(int(port)):
        components.iframe(f"http://localhost:{int(port)}", height=750, scrolling=True)


# ---------------------------------------------------------------- Predict


@st.cache_resource(show_spinner="Đang nạp model...")
def cached_model(digest: str, _source, arch: str | None):
    return models.load_model(_source, arch)


def model_picker(role: str) -> tuple | None:
    st.markdown(f"#### {'👨‍🏫 Teacher' if role == 'teacher' else '🧑‍🎓 Student'}")
    options = ["(không dùng)", "Upload .pth"]
    if role == "teacher":
        options.insert(1, "mdistiller pretrained")
    trained = [r for r in store.list() if "best_checkpoint" in metrics.find_artifacts(store.run_dir(r["run_id"]) / "output")]
    if role == "student" and trained:
        options.insert(1, "Từ một run đã train")
    src = st.radio("Nguồn", options, key=f"src_{role}", index=1 if len(options) > 1 else 0)
    try:
        if src == "mdistiller pretrained":
            arch = st.selectbox("Kiến trúc", list(TEACHERS), format_func=TEACHERS.get, key="t_arch")
            path = models.teacher_path(arch)
            if not path.exists():
                st.caption("Chưa có file — tải từ GitHub mdistiller (stream tar 364 MB, chỉ giữ 1 checkpoint).")
                if st.button("⬇️ Tải teacher", key="dl_teacher"):
                    with st.spinner("Đang tải..."):
                        models.download_teacher(arch)
                    st.rerun()
                return None
            return cached_model(f"{path}:{path.stat().st_mtime}", path, arch)
        if src == "Từ một run đã train":
            r = st.selectbox("Run", trained, format_func=run_label, key="s_run")
            path = metrics.find_artifacts(store.run_dir(r["run_id"]) / "output")["best_checkpoint"]
            return cached_model(f"{path}:{path.stat().st_mtime}", path, r.get("config", {}).get("student"))
        if src == "Upload .pth":
            up = st.file_uploader("Checkpoint (.pth/.pt)", type=["pth", "pt"], key=f"up_{role}")
            if up is not None:
                data = up.getvalue()
                return cached_model(hashlib.sha256(data).hexdigest(), data, None)
    except Exception as exc:  # noqa: BLE001 — show any load error in the UI
        st.error(f"Không nạp được model: {exc}")
    return None


@st.cache_resource(show_spinner="Đang tải CIFAR-100 test (~160 MB lần đầu)...")
def cifar_test():
    return models.load_cifar100_test()


def prob_chart(preds: list[tuple[str, float]], title: str, truth: str | None) -> go.Figure:
    labels = [p[0] for p in preds][::-1]
    colors = ["#2e7d32" if l == truth else "#1a56db" for l in labels]
    fig = go.Figure(go.Bar(x=[p[1] * 100 for p in preds][::-1], y=labels, orientation="h", marker_color=colors,
                           text=[f"{p[1] * 100:.1f}%" for p in preds][::-1], textposition="auto"))
    fig.update_layout(title=title, height=300, xaxis_title="xác suất (%)", xaxis_range=[0, 100], margin=dict(t=40, b=10))
    return fig


with tab_pred:
    st.subheader("Dự đoán với teacher / student (CIFAR-100, 100 lớp)")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        teacher_m = model_picker("teacher")
    with mcol2:
        student_m = model_picker("student")
    loaded = [(f"Teacher {m[1]}", m[0]) for m in [teacher_m] if m] + [(f"Student {m[1]}", m[0]) for m in [student_m] if m]
    st.divider()
    img_src = st.radio("Ảnh", ["Upload ảnh", "Ảnh ngẫu nhiên từ CIFAR-100 test"], horizontal=True)
    image, truth = None, None
    if img_src == "Upload ảnh":
        up_img = st.file_uploader("Ảnh (jpg/png)", type=["jpg", "jpeg", "png", "webp"])
        if up_img is not None:
            image = Image.open(up_img)
    else:
        if st.button("🎲 Lấy ảnh khác") or "cifar_idx" not in st.session_state:
            st.session_state.cifar_idx = random.randrange(10000)
        try:
            ds = cifar_test()
            image, label = ds[st.session_state.cifar_idx]
            truth = models.CIFAR100_CLASSES[label]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Không tải được CIFAR-100: {exc}")
    if image is not None:
        icol, *pcols = st.columns([1] + [2] * max(1, len(loaded)))
        icol.image(image.resize((160, 160), Image.NEAREST) if image.size[0] <= 64 else image,
                   caption=f"Nhãn thật: {truth}" if truth else "Ảnh đầu vào", use_container_width=True)
        if not loaded:
            pcols[0].info("Chọn ít nhất một model ở trên.")
        for col, (name, model) in zip(pcols, loaded):
            preds = models.predict(model, image)
            col.plotly_chart(prob_chart(preds, name, truth), use_container_width=True)
            if truth:
                col.markdown("✅ Đúng" if preds[0][0] == truth else f"❌ Sai (đoán **{preds[0][0]}**)")
        st.caption("Ảnh được resize về 32×32 và chuẩn hoá theo CIFAR-100 như lúc train.")
