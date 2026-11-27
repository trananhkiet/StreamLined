/* SKD Studio — vanilla JS single-page dashboard. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const MODEL_META = {
  resnet20: { name: "ResNet20", params: "0.28M tham số" },
  resnet32: { name: "ResNet32", params: "0.47M tham số" },
  resnet56: { name: "ResNet56", params: "0.86M tham số", acc: 72.41 },
  resnet110: { name: "ResNet110", params: "1.74M tham số", acc: 74.31 },
};
const RECIPE_DESC = {
  E4: "Tái lập SKD gốc: 0.5·CE + 0.5·KD + 0.5·BC, KD như nhau cho mọi mẫu.",
  F1b: "Cải tiến của bạn: KD = 0.6 khi teacher đoán đúng nhãn, 0.5 khi sai (CE = 1 − w).",
};
const STATUS_TEXT = { queued: "Đang chờ", running: "Đang chạy", complete: "Hoàn thành", error: "Lỗi", push_failed: "Push lỗi", draft: "Bản nháp", cancelacknowledged: "Đã huỷ", unknown: "Không rõ" };

const state = {
  opts: null,
  session: { connected: false, username: null },
  runs: [],
  form: null,
  selectedRun: null,
  charts: {},
  autoRefresh: null,
  predict: { teacher: "", student: "", file: null, cifarIndex: null, previewUrl: null },
};

if (window.Chart) {
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.color = "#7a8296";
  Chart.defaults.borderColor = "#eceef5";
}

/* ------------------------------------------------------------------ infra */

async function api(path, { method = "GET", json, form, quiet = false } = {}) {
  const opts = { method, headers: {}, credentials: "same-origin" };
  if (json !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(json); }
  if (form) opts.body = form;
  const res = await fetch(path, opts);
  const isJson = (res.headers.get("content-type") || "").includes("json");
  const data = isJson ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = (data && data.detail) ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : `HTTP ${res.status}`;
    if (!quiet) toast(msg, "err");
    throw new Error(msg);
  }
  return data;
}

function toast(msg, kind = "info", ms = 4200) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 300); }, ms);
}

async function waitJob(id, onTick) {
  for (;;) {
    const job = await api(`/api/jobs/${id}`, { quiet: true });
    if (onTick) onTick(job);
    if (job.state === "done") return job;
    if (job.state === "error") throw new Error(job.message || "Job lỗi");
    await sleep(1000);
  }
}

function busy(btn, on, label) {
  if (!btn) return;
  if (on) { btn.dataset.label = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spin"></span>${label || ""}`; }
  else { btn.disabled = false; if (btn.dataset.label) btn.innerHTML = btn.dataset.label; }
}

function destroyCharts() { Object.values(state.charts).forEach((c) => c.destroy()); state.charts = {}; }

function badge(status) { const s = status || "unknown"; return `<span class="badge ${esc(s)}">${esc(STATUS_TEXT[s] || s)}</span>`; }

const refKey = (t, s) => `${t}|${s}`;

/* ------------------------------------------------------------------ session */

async function loadSession() {
  state.session = await api("/api/session", { quiet: true });
  $("#account-dot").classList.toggle("on", state.session.connected);
  $("#account-label").textContent = state.session.connected ? state.session.username : "Kết nối Kaggle";
  $("#cred-logout").hidden = !state.session.connected;
}

function openModal() {
  $("#cred-user").value = state.session.username || "";
  $("#cred-key").value = "";
  $("#modal").hidden = false;
  setTimeout(() => (state.session.username ? $("#cred-key") : $("#cred-user")).focus(), 50);
}
function closeModal() { $("#modal").hidden = true; $("#cred-key").value = ""; }

async function saveCreds(username, key) {
  const btn = $("#cred-save");
  busy(btn, true, "Đang kiểm tra");
  try {
    await api("/api/session", { method: "POST", json: { username, key } });
    const v = await api("/api/session/verify", { method: "POST", quiet: true }).catch((e) => ({ ok: false, message: e.message }));
    await loadSession();
    closeModal();
    toast(v.ok ? `Đã kết nối Kaggle: ${username}` : `Đã lưu key, nhưng kiểm tra lỗi: ${v.message}`, v.ok ? "ok" : "err", 6000);
    route();
  } finally { busy(btn, false); }
}

function initModal() {
  $("#account-btn").onclick = openModal;
  $$("[data-close]").forEach((b) => (b.onclick = closeModal));
  $("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
  $("#cred-save").onclick = () => {
    const u = $("#cred-user").value.trim(), k = $("#cred-key").value.trim();
    if (!u || !k) return toast("Nhập đủ username và API key", "err");
    saveCreds(u, k);
  };
  $("#cred-key").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#cred-save").click(); });
  $("#cred-logout").onclick = async () => { await api("/api/session", { method: "DELETE" }); await loadSession(); closeModal(); toast("Đã ngắt kết nối Kaggle"); route(); };
  const drop = $("#kaggle-json-drop"), input = $("#kaggle-json-input");
  const readJson = async (file) => {
    try { const d = JSON.parse(await file.text()); if (!d.username || !d.key) throw new Error(); saveCreds(String(d.username), String(d.key)); }
    catch { toast("kaggle.json không hợp lệ", "err"); }
  };
  $("#kaggle-json-pick").onclick = (e) => { e.preventDefault(); input.click(); };
  drop.onclick = (e) => { if (e.target.tagName !== "A") input.click(); };
  input.onchange = () => input.files[0] && readJson(input.files[0]);
  dropHandlers(drop, (f) => readJson(f));
}

function dropHandlers(el, onFile) {
  el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("over"); });
  el.addEventListener("dragleave", () => el.classList.remove("over"));
  el.addEventListener("drop", (e) => { e.preventDefault(); el.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) onFile(f); });
}

/* ------------------------------------------------------------------ router */

const PAGES = {
  overview: { title: "Tổng quan", sub: "Paper gốc, kết quả tái lập và các thí nghiệm của bạn", render: renderOverview },
  train: { title: "Huấn luyện", sub: "Cấu hình một lần distillation và chạy trên GPU Kaggle (T4)", render: renderTrain },
  runs: { title: "Thí nghiệm", sub: "Lịch sử train, trạng thái, log và kết quả", render: renderRuns },
  tensorboard: { title: "TensorBoard", sub: "Xuất đường cong huấn luyện và mở TensorBoard", render: renderTensorboard },
  predict: { title: "Dự đoán", sub: "So sánh dự đoán của teacher và student trên CIFAR-100", render: renderPredict },
};

async function route() {
  const name = (location.hash.replace("#/", "").split("/")[0]) || "overview";
  const page = PAGES[name] || PAGES.overview;
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === name));
  $("#page-title").textContent = page.title;
  $("#page-sub").textContent = page.sub;
  destroyCharts();
  if (state.autoRefresh && name !== "runs") { clearInterval(state.autoRefresh); state.autoRefresh = null; }
  await page.render($("#view"));
}

async function loadRuns() {
  state.runs = await api("/api/runs", { quiet: true }).catch(() => []);
  $("#nav-runs-count").textContent = state.runs.length || "";
  return state.runs;
}

/* ------------------------------------------------------------------ overview */

async function renderOverview(view) {
  await loadRuns();
  const ref = state.opts.reference;
  const r20 = ref[refKey("resnet56", "resnet20")], r32 = ref[refKey("resnet110", "resnet32")];
  const scored = state.runs.filter((r) => r.metrics && r.metrics.best != null);
  const yourBest = scored.length ? Math.max(...scored.map((r) => r.metrics.best)) : null;
  const running = state.runs.filter((r) => ["running", "queued"].includes(r.status)).length;
  const delta = (a, b) => { const d = a - b; return `<div class="delta ${d > 0.005 ? "up" : d < -0.005 ? "down" : "flat"}">${d >= 0 ? "+" : ""}${d.toFixed(2)}</div>`; };
  view.innerHTML = `
  <div class="stack">
    <div class="hero">
      <div>
        <span class="tag" style="background:rgba(255,255,255,.16);color:#fff">CIFAR-100 · Knowledge Distillation</span>
        <h2 style="margin-top:12px">Streamlined Knowledge Distillation</h2>
        <p>Train student nhỏ (ResNet20/32) học từ teacher lớn (ResNet56/110) trên GPU Kaggle, theo dõi kết quả, so với paper và thử dự đoán trực tiếp — tất cả trong một giao diện.</p>
        <div class="row">
          <a class="btn white" href="#/train">Bắt đầu train</a>
          <a class="btn glass" href="#/runs">Xem thí nghiệm</a>
          <a class="btn glass" href="#/predict">Thử dự đoán</a>
        </div>
      </div>
      <div class="hero-stats">
        <div class="hero-stat"><div class="l">Paper SKD · R56→R20</div><div class="v">${fmt(r20.paper_skd)}</div><div class="s">KD thường: ${fmt(r20.paper_kd)}</div></div>
        <div class="hero-stat"><div class="l">Paper SKD · R110→R32</div><div class="v">${fmt(r32.paper_skd)}</div><div class="s">KD thường: ${fmt(r32.paper_kd)}</div></div>
        <div class="hero-stat"><div class="l">Best trong app</div><div class="v">${yourBest == null ? "–" : fmt(yourBest)}</div><div class="s">${scored.length} run có kết quả</div></div>
        <div class="hero-stat"><div class="l">Đang chạy</div><div class="v">${running}</div><div class="s">${state.runs.length} run tổng cộng</div></div>
      </div>
    </div>

    <div>
      <div class="section-title">Kết quả đã tái lập (top-1 %, best epoch)</div>
      <div class="grid g4">
        <div class="card kpi"><div class="label">Tái lập E4 · R56→R20</div><div class="value">${fmt(r20.ours_e4)}</div>${delta(r20.ours_e4, r20.paper_skd)}<div class="muted small">so với paper ${fmt(r20.paper_skd)}</div></div>
        <div class="card kpi accent"><div class="label">F1b (của bạn) · R56→R20</div><div class="value">${fmt(r20.ours_f1b)}</div><div class="delta">+${(r20.ours_f1b - r20.ours_e4).toFixed(2)} so với E4 · TB 2 seed</div></div>
        <div class="card kpi"><div class="label">Tái lập E4 · R110→R32</div><div class="value">${fmt(r32.ours_e4)}</div>${delta(r32.ours_e4, r32.paper_skd)}<div class="muted small">so với paper ${fmt(r32.paper_skd)}</div></div>
        <div class="card kpi"><div class="label">F1b · R110→R32</div><div class="value">${fmt(r32.ours_f1b)}</div>${delta(r32.ours_f1b, r32.ours_e4)}<div class="muted small">so với E4 · 1 seed</div></div>
      </div>
    </div>

    <div>
      <div class="section-title">Quy trình demo</div>
      <div class="steps">
        <div class="card step"><div class="num-c">1</div><h4>Kết nối Kaggle</h4><p>Nhập username + API key (góc trên bên phải). Key chỉ nằm trong bộ nhớ.</p></div>
        <div class="card step"><div class="num-c">2</div><h4>Cấu hình & train</h4><p>Chọn teacher/student, recipe E4/F1b, siêu tham số rồi đẩy lên GPU Kaggle.</p></div>
        <div class="card step"><div class="num-c">3</div><h4>Theo dõi & so sánh</h4><p>Trạng thái, log, best/final/TB10, đường cong và so với paper.</p></div>
        <div class="card step"><div class="num-c">4</div><h4>Dự đoán</h4><p>Nạp teacher/student, đưa ảnh vào và so top-5 của hai model.</p></div>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><div><h3>Run gần đây</h3><div class="sub">5 run mới nhất</div></div><a class="btn sm" href="#/runs">Tất cả</a></div>
      ${runsTable(state.runs.slice(0, 5), false)}
    </div>
  </div>`;
  $$("tbody tr[data-run]", view).forEach((tr) => (tr.onclick = () => { state.selectedRun = tr.dataset.run; location.hash = "#/runs"; }));
}

function runsTable(runs, selectable = true) {
  if (!runs.length) {
    return `<div class="empty"><div class="big">Chưa có run nào</div>Train một run mới hoặc nạp log mẫu ở trang Thí nghiệm.</div>`;
  }
  const rows = runs.map((r) => {
    const c = r.config || {}, m = r.metrics || {};
    const title = r.label || `${MODEL_META[c.teacher]?.name || c.teacher} → ${MODEL_META[c.student]?.name || c.student}`;
    return `<tr data-run="${esc(r.run_id)}" class="${selectable && r.run_id === state.selectedRun ? "sel" : ""}">
      <td><div class="run-name">${esc(title)}</div><div class="run-sub">${esc(r.run_id)}</div></td>
      <td><span class="tag">${esc(c.recipe || "–")}</span></td>
      <td>${esc(c.seed ?? "–")}</td>
      <td>${badge(r.status)}</td>
      <td class="num"><b>${fmt(m.best)}</b></td>
      <td class="num">${fmt(m.final)}</td>
      <td class="num">${fmt(m.tb10)}</td>
      <td class="num">${m.epochs_done ? `${m.epochs_done}/${m.epochs_total || c.epochs || "?"}` : "–"}</td>
      <td class="muted">${esc((r.submitted_at || "").replace("T", " ").replace("Z", ""))}</td>
    </tr>`;
  }).join("");
  return `<div style="overflow:auto"><table class="table">
    <thead><tr><th>Run</th><th>Recipe</th><th>Seed</th><th>Trạng thái</th><th class="num">Best</th><th class="num">Final</th><th class="num">TB10</th><th class="num">Epoch</th><th>Thời gian (UTC)</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

/* ------------------------------------------------------------------ train */

function choiceCards(name, items, selected, { disabled = {}, sub = {} } = {}) {
  return Object.entries(items).map(([key, label]) => `
    <button type="button" class="choice ${key === selected ? "selected" : ""} ${disabled[key] ? "disabled" : ""}" data-name="${name}" data-value="${esc(key)}" ${disabled[key] ? "disabled" : ""}>
      <div class="c-title"><span>${esc(label)}</span><span class="check"></span></div>
      ${sub[key] ? `<div class="c-sub">${esc(sub[key])}</div>` : ""}
    </button>`).join("");
}

async function renderTrain(view) {
  const o = state.opts;
  const f = (state.form ||= { ...o.defaults });
  const teacherLabels = Object.fromEntries(Object.keys(o.teachers).map((k) => [k, MODEL_META[k].name]));
  const studentLabels = Object.fromEntries(Object.keys(o.students).map((k) => [k, MODEL_META[k].name]));
  const tSub = Object.fromEntries(Object.keys(o.teachers).map((k) => [k, `${MODEL_META[k].params} · top-1 ${MODEL_META[k].acc}% (mdistiller)`]));
  const sSub = Object.fromEntries(Object.keys(o.students).map((k) => [k, MODEL_META[k].params]));
  const dsLabels = Object.fromEntries(Object.entries(o.datasets).map(([k, v]) => [k, v.label.split(" (")[0]]));
  const dsSub = Object.fromEntries(Object.entries(o.datasets).map(([k, v]) => [k, v.enabled ? "50k train · 10k test · 100 lớp" : "Chưa có teacher pretrained"]));
  const dsDisabled = Object.fromEntries(Object.entries(o.datasets).map(([k, v]) => [k, !v.enabled]));
  const recipeLabels = { E4: "E4 · Tái lập SKD", F1b: "F1b · Teacher-aware KD" };

  view.innerHTML = `
  <div class="train-layout">
    <div class="card card-pad">
      <div class="form-block">
        <div class="section-title">Teacher</div>
        <div class="choice-grid">${choiceCards("teacher", teacherLabels, f.teacher, { sub: tSub })}</div>
      </div>
      <div class="form-block">
        <div class="section-title">Student</div>
        <div class="choice-grid">${choiceCards("student", studentLabels, f.student, { sub: sSub })}</div>
      </div>
      <div class="form-block">
        <div class="section-title">Dataset</div>
        <div class="grid g3">${choiceCards("dataset", dsLabels, f.dataset, { sub: dsSub, disabled: dsDisabled })}</div>
      </div>
      <div class="form-block">
        <div class="section-title">Recipe (hàm loss)</div>
        <div class="choice-grid">${choiceCards("recipe", recipeLabels, f.recipe, { sub: RECIPE_DESC })}</div>
      </div>
      <div class="form-block">
        <div class="section-title">Siêu tham số</div>
        <div class="params">
          ${numField("epochs", "Epochs", 1, 400, 1)}
          ${numField("learning_rate", "Learning rate", 0.001, 1, 0.001)}
          <label class="field"><span>LR giảm ×0.1 tại epoch</span><input data-field="lr_milestones" value="${esc(f.lr_milestones)}"><span class="hint">vd. 100,188,250</span></label>
          <label class="field"><span>Batch size</span><select data-field="batch_size">${[32, 64, 128, 256].map((b) => `<option ${b === f.batch_size ? "selected" : ""}>${b}</option>`).join("")}</select></label>
          ${numField("temperature", "Temperature T", 0.5, 20, 0.5)}
          ${numField("seed", "Seed", 0, 99999, 1)}
          ${numField("ce_weight", "CE weight", 0, 2, 0.1, f.recipe === "F1b", "F1b tự đặt 1 − w")}
          ${numField("kd_weight", "KD weight", 0, 2, 0.1, f.recipe === "F1b", "F1b: w = 0.6 / 0.5")}
        </div>
        <div class="switch-row" style="margin-top:18px">
          <div><b>Smoke test</b><div class="muted small">Chỉ 20 batch train / 5 batch eval mỗi epoch — kiểm tra pipeline trong vài phút.</div></div>
          <label class="switch"><input type="checkbox" data-field="smoke_test" ${f.smoke_test ? "checked" : ""}><span></span></label>
        </div>
      </div>
    </div>

    <div class="summary stack">
      <div class="card card-pad stack" style="gap:16px">
        <div class="pipeline">
          <div class="node"><div class="n-role">Teacher</div><div class="n-name" id="sum-t"></div><div class="n-meta" id="sum-tm"></div></div>
          <div class="arrow">→</div>
          <div class="node"><div class="n-role">Student</div><div class="n-name" id="sum-s"></div><div class="n-meta" id="sum-sm"></div></div>
        </div>
        <dl class="kv" id="sum-kv"></dl>
        <div id="sum-errors"></div>
        <details><summary>Biến môi trường gửi cho script</summary><div class="codebox" id="sum-env" style="margin-top:10px"></div></details>
        ${state.session.connected ? "" : `<div class="notice">🔑 Kết nối Kaggle (góc trên phải) để bật nút Train. Vẫn có thể tạo notebook nháp.</div>`}
        <button class="btn primary block" id="btn-train" ${state.session.connected ? "" : "disabled"}>
          <svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>Train trên Kaggle</button>
        <button class="btn block" id="btn-draft">Chỉ tạo notebook (không chạy)</button>
      </div>
    </div>
  </div>`;

  $$(".choice", view).forEach((b) => (b.onclick = () => {
    f[b.dataset.name] = b.dataset.value;
    if (b.dataset.name === "teacher") f.student = { resnet56: "resnet20", resnet110: "resnet32" }[b.dataset.value];  // paper pairs
    if (b.dataset.name === "recipe" || b.dataset.name === "teacher" || b.dataset.name === "student") return renderTrain(view);
    $$(`.choice[data-name="${b.dataset.name}"]`, view).forEach((x) => x.classList.toggle("selected", x === b));
    updatePreview();
  }));
  $$("[data-field]", view).forEach((el) => {
    const handler = () => {
      const k = el.dataset.field;
      f[k] = el.type === "checkbox" ? el.checked : (el.type === "number" || k === "batch_size") ? Number(el.value) : el.value;
      updatePreview();
    };
    el.addEventListener("input", handler);
    el.addEventListener("change", handler);
  });
  $("#btn-train").onclick = () => submitRun(false, $("#btn-train"));
  $("#btn-draft").onclick = () => submitRun(true, $("#btn-draft"));
  updatePreview();

  function numField(key, label, min, max, step, disabled = false, hint = "") {
    return `<label class="field"><span>${label}</span><input type="number" data-field="${key}" min="${min}" max="${max}" step="${step}" value="${esc(f[key])}" ${disabled ? "disabled" : ""}>${hint ? `<span class="hint">${hint}</span>` : ""}</label>`;
  }
}

let previewTimer = null;
function updatePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    const f = state.form;
    const p = await api("/api/preview", { method: "POST", json: f, quiet: true }).catch(() => null);
    if (!p || !$("#sum-kv")) return;
    const ref = state.opts.reference[refKey(f.teacher, f.student)];
    $("#sum-t").textContent = MODEL_META[f.teacher].name;
    $("#sum-tm").textContent = `${MODEL_META[f.teacher].acc}%`;
    $("#sum-s").textContent = MODEL_META[f.student].name;
    $("#sum-sm").textContent = MODEL_META[f.student].params.replace(" tham số", "");
    $("#sum-kv").innerHTML = `
      <dt>Recipe</dt><dd>${esc(f.recipe)}${f.smoke_test ? " · smoke" : ""}</dd>
      <dt>Ước tính GPU T4</dt><dd>~${p.hours} giờ</dd>
      ${ref ? `<dt>Paper SKD</dt><dd>${fmt(ref.paper_skd)}</dd><dt>Tái lập E4 / F1b</dt><dd>${fmt(ref.ours_e4)} / ${fmt(ref.ours_f1b)}</dd>`
            : `<dt>Paper</dt><dd class="muted">cặp này không có trong paper</dd>`}
      <dt>Kernel</dt><dd style="font-family:var(--mono);font-size:11.5px;font-weight:500">${esc(p.slug)}</dd>`;
    $("#sum-env").innerHTML = Object.entries(p.env).map(([k, v]) => `<span class="k">${esc(k)}</span>=<span class="v">${esc(v)}</span>`).join("<br>");
    $("#sum-errors").innerHTML = p.errors.length ? `<div class="errors">${p.errors.map(esc).join("<br>")}</div>` : "";
    $("#btn-train").disabled = !state.session.connected || p.errors.length > 0;
    $("#btn-draft").disabled = p.errors.length > 0;
  }, 150);
}

async function submitRun(dry, btn) {
  busy(btn, true, dry ? "Đang tạo notebook" : "Đang đẩy lên Kaggle");
  try {
    const rec = await api(`/api/runs${dry ? "?dry_run=true" : ""}`, { method: "POST", json: state.form });
    if (rec.status === "push_failed") toast("Push lên Kaggle thất bại — xem chi tiết trong run", "err", 7000);
    else toast(dry ? "Đã tạo notebook nháp" : "Đã gửi run lên Kaggle 🚀", "ok");
    state.selectedRun = rec.run_id;
    location.hash = "#/runs";
  } catch { /* toast shown */ } finally { busy(btn, false); }
}

/* ------------------------------------------------------------------ runs */

async function renderRuns(view) {
  await loadRuns();
  if (!state.selectedRun && state.runs.length) state.selectedRun = state.runs[0].run_id;
  const scored = state.runs.filter((r) => r.metrics && r.metrics.best != null);
  view.innerHTML = `
  <div class="stack">
    <div class="row">
      <button class="btn" id="btn-import"><svg viewBox="0 0 24 24"><path d="M12 4v12m0 0l-4-4m4 4l4-4M4 20h16"/></svg>Import log / history.csv</button>
      <button class="btn" id="btn-sample">Nạp log mẫu</button>
      <span class="spacer"></span>
      <a class="btn primary" href="#/train">+ Run mới</a>
    </div>
    <div class="card card-pad" id="import-panel" hidden>
      <div class="grid g4" style="align-items:end">
        <label class="field"><span>File</span><input type="file" id="imp-file" accept=".log,.json,.txt,.csv"></label>
        <label class="field"><span>Teacher</span><select id="imp-t">${Object.keys(state.opts.teachers).map((k) => `<option value="${k}">${MODEL_META[k].name}</option>`).join("")}</select></label>
        <label class="field"><span>Student</span><select id="imp-s">${Object.keys(state.opts.students).map((k) => `<option value="${k}">${MODEL_META[k].name}</option>`).join("")}</select></label>
        <label class="field"><span>Recipe</span><select id="imp-r"><option>E4</option><option>F1b</option></select></label>
      </div>
      <div class="row" style="margin-top:14px"><span class="muted small">Nhận log Kaggle (JSON hoặc text) hoặc history.csv do script ghi ra.</span><span class="spacer"></span><button class="btn primary sm" id="imp-go">Import</button></div>
    </div>
    ${scored.length ? `<div class="card"><div class="card-head"><div><h3>So sánh các run</h3><div class="sub">Top-1 (%) · vạch đứt = paper SKD của cặp tương ứng</div></div></div><div class="card-pad"><div class="chart-box sm"><canvas id="cmp-chart"></canvas></div></div></div>` : ""}
    <div class="card">${runsTable(state.runs)}</div>
    <div id="run-detail"></div>
  </div>`;

  $("#btn-import").onclick = () => ($("#import-panel").hidden = !$("#import-panel").hidden);
  $("#btn-sample").onclick = async (e) => {
    busy(e.currentTarget, true);
    try { const r = await api("/api/runs/import-sample", { method: "POST" }); state.selectedRun = r.run_id; toast("Đã nạp log mẫu", "ok"); route(); }
    finally { busy(e.currentTarget, false); }
  };
  $("#imp-go").onclick = async (e) => {
    const file = $("#imp-file").files[0];
    if (!file) return toast("Chọn file trước", "err");
    const form = new FormData();
    form.append("file", file); form.append("teacher", $("#imp-t").value); form.append("student", $("#imp-s").value); form.append("recipe", $("#imp-r").value);
    busy(e.currentTarget, true);
    try { const r = await api("/api/runs/import", { method: "POST", form }); state.selectedRun = r.run_id; toast("Import thành công", "ok"); route(); }
    finally { busy(e.currentTarget, false); }
  };
  $$("tbody tr[data-run]", view).forEach((tr) => (tr.onclick = () => {
    state.selectedRun = tr.dataset.run;
    $$("tbody tr", view).forEach((x) => x.classList.toggle("sel", x === tr));
    renderRunDetail();
  }));
  if (scored.length) drawCompare(scored);
  if (state.selectedRun) renderRunDetail();
}

function drawCompare(runs) {
  const labels = runs.map((r) => r.label || r.run_id);
  const papers = runs.map((r) => state.opts.reference[refKey(r.config?.teacher, r.config?.student)]?.paper_skd ?? null);
  const vals = runs.flatMap((r) => [r.metrics.best, r.metrics.final, r.metrics.tb10]).concat(papers.filter((v) => v != null));
  state.charts.cmp = new Chart($("#cmp-chart"), {
    data: {
      labels,
      datasets: [
        { type: "bar", label: "Best", data: runs.map((r) => r.metrics.best), backgroundColor: "#4f46e5", borderRadius: 6, maxBarThickness: 34 },
        { type: "bar", label: "Final", data: runs.map((r) => r.metrics.final), backgroundColor: "#a5b4fc", borderRadius: 6, maxBarThickness: 34 },
        { type: "bar", label: "TB10", data: runs.map((r) => r.metrics.tb10), backgroundColor: "#c4b5fd", borderRadius: 6, maxBarThickness: 34 },
        { type: "line", label: "Paper SKD", data: papers, borderColor: "#f59e0b", borderDash: [6, 5], pointRadius: 0, borderWidth: 2, stepped: "middle" },
      ],
    },
    options: {
      maintainAspectRatio: false,
      scales: { y: { min: Math.max(0, Math.floor(Math.min(...vals) - 5)), max: Math.ceil(Math.max(...vals) + 0.5), title: { display: true, text: "top-1 (%)" } }, x: { grid: { display: false } } },
      plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8 } } },
    },
  });
}

async function renderRunDetail() {
  const box = $("#run-detail");
  if (!box || !state.selectedRun) return;
  let d;
  try { d = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}`, { quiet: true }); }
  catch { box.innerHTML = ""; return; }
  const r = d.run, c = r.config || {}, s = d.summary || {};
  const ref = state.opts.reference[refKey(c.teacher, c.student)];
  const isKaggle = r.source === "kaggle";
  const can = isKaggle && state.session.connected;
  const total = s.epochs_total || c.epochs || 0, done = s.epochs_done || 0;
  const pct = total ? Math.min(100, (100 * done) / total) : 0;
  const dPaper = ref && s.best != null ? s.best - ref.paper_skd : null;
  ["detail", "detailLr"].forEach((k) => state.charts[k] && (state.charts[k].destroy(), delete state.charts[k]));

  box.innerHTML = `
  <div class="card">
    <div class="card-head detail-head">
      <div>
        <div class="row"><h3>${esc(r.label || `${MODEL_META[c.teacher]?.name} → ${MODEL_META[c.student]?.name}`)}</h3>${badge(r.status)}<span class="tag">${esc(c.recipe || "")}</span></div>
        <div class="sub">${esc(r.run_id)}${r.url && isKaggle ? ` · <a href="${esc(r.url)}" target="_blank" rel="noopener">mở trên Kaggle ↗</a>` : ""}${r.status_checked_at ? ` · cập nhật ${esc(r.status_checked_at.replace("T", " ").replace("Z", ""))}` : ""}</div>
      </div>
      <div class="row">
        ${isKaggle ? `
          <button class="btn sm" data-act="status" ${can ? "" : "disabled"}>Cập nhật trạng thái</button>
          <button class="btn sm" data-act="logs" ${can ? "" : "disabled"}>Lấy log</button>
          <button class="btn sm" data-act="output" ${can ? "" : "disabled"}>Tải output</button>
          <label class="row small muted" title="Mỗi 60 giây cập nhật trạng thái + log"><label class="switch" style="transform:scale(.8)"><input type="checkbox" id="auto-ref" ${state.autoRefresh ? "checked" : ""} ${can ? "" : "disabled"}><span></span></label>Tự động</label>` : ""}
        ${d.has_notebook ? `<a class="btn sm" href="/api/runs/${encodeURIComponent(r.run_id)}/notebook">Notebook</a>` : ""}
        ${d.series.epoch ? `<a class="btn sm" href="/api/runs/${encodeURIComponent(r.run_id)}/metrics.csv">CSV</a>` : ""}
        <button class="btn sm ghost danger" data-act="delete">Xoá</button>
      </div>
    </div>
    <div class="card-pad stack">
      ${isKaggle && !state.session.connected ? `<div class="notice">Kết nối Kaggle để cập nhật trạng thái, lấy log và tải output của run này.</div>` : ""}
      ${r.status === "push_failed" ? `<div class="errors">${esc(r.push_output || "Push thất bại")}</div>` : ""}
      ${r.status === "draft" ? `<div class="notice">Bản nháp: notebook + kernel-metadata đã được tạo trong <code>app/runs/${esc(r.run_id)}/kernel</code>. Tự chạy bằng <code>kaggle kernels push -p &lt;thư mục&gt;</code> hoặc kết nối Kaggle và bấm Train.</div>` : ""}
      <div class="grid g4">
        <div class="card kpi accent"><div class="label">Best top-1</div><div class="value">${fmt(s.best)}</div><div class="delta">${s.best_epoch ? `epoch ${s.best_epoch}` : "&nbsp;"}</div></div>
        <div class="card kpi"><div class="label">Final</div><div class="value">${fmt(s.final)}</div><div class="delta flat">epoch cuối</div></div>
        <div class="card kpi"><div class="label">TB10</div><div class="value">${fmt(s.tb10)}</div><div class="delta flat">TB 10 epoch cuối</div></div>
        <div class="card kpi"><div class="label">So với paper SKD</div><div class="value">${dPaper == null ? "–" : (dPaper >= 0 ? "+" : "") + dPaper.toFixed(2)}</div><div class="delta flat">${ref ? `paper ${fmt(ref.paper_skd)}` : "&nbsp;"}${s.teacher_top1 ? ` · teacher ${fmt(s.teacher_top1)}` : ""}</div></div>
      </div>
      <div><div class="row small" style="justify-content:space-between;margin-bottom:6px"><span class="muted">Tiến độ</span><b>${done}/${total || "?"} epoch</b></div><div class="progress"><div style="width:${pct}%"></div></div></div>
      ${d.series.epoch ? `<div class="chart-box"><canvas id="detail-chart"></canvas></div>`
        : `<div class="empty"><div class="big">Chưa có số liệu</div>${isKaggle ? "Bấm <b>Lấy log</b> khi đang chạy hoặc <b>Tải output</b> khi đã xong." : "Run này chưa có epoch nào."}</div>`}
      <div class="grid g2">
        <div><div class="section-title">Log</div><div class="log" id="log-view">${d.log_tail ? highlightLog(d.log_tail) : '<span style="color:#7d86a1">(chưa có log)</span>'}</div></div>
        <div><div class="section-title">Cấu hình</div><div class="codebox" style="max-height:300px;height:300px">${Object.entries(c.env || c).filter(([k]) => k !== "env").map(([k, v]) => `<span class="k">${esc(k)}</span>=<span class="v">${esc(typeof v === "object" ? JSON.stringify(v) : v)}</span>`).join("<br>")}</div></div>
      </div>
    </div>
  </div>`;
  const lv = $("#log-view"); if (lv) lv.scrollTop = lv.scrollHeight;

  if (d.series.epoch) {
    const S = d.series;
    state.charts.detail = new Chart($("#detail-chart"), {
      type: "line",
      data: {
        labels: S.epoch,
        datasets: [
          { label: "Test top-1", data: S.test_top1, borderColor: "#4f46e5", backgroundColor: "rgba(79,70,229,.08)", fill: true, tension: .25, pointRadius: 0, borderWidth: 2.2 },
          ...(S.train_top1 ? [{ label: "Train top-1", data: S.train_top1, borderColor: "#a78bfa", borderDash: [5, 4], tension: .25, pointRadius: 0, borderWidth: 1.6 }] : []),
          ...(ref ? [{ label: "Paper SKD", data: S.epoch.map(() => ref.paper_skd), borderColor: "#f59e0b", borderDash: [6, 5], pointRadius: 0, borderWidth: 1.5 }] : []),
          ...(S.lr ? [{ label: "Learning rate", data: S.lr, borderColor: "#94a3b8", pointRadius: 0, borderWidth: 1.2, yAxisID: "lr", stepped: true }] : []),
        ],
      },
      options: {
        maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        scales: {
          y: { title: { display: true, text: "top-1 (%)" } },
          lr: { position: "right", type: "logarithmic", grid: { display: false }, title: { display: true, text: "lr" },
                ticks: { callback: (v) => ([1, 0.1, 0.01, 0.001, 0.0001].some((p) => Math.abs(v - p) < p * 1e-6) ? v : "") } },
          x: { title: { display: true, text: "epoch" }, ticks: { maxTicksLimit: 12 } },
        },
        plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8 } } },
      },
    });
  }

  $$("[data-act]", box).forEach((b) => (b.onclick = () => runAction(b.dataset.act, r, b)));
  const auto = $("#auto-ref");
  if (auto) auto.onchange = () => {
    clearInterval(state.autoRefresh); state.autoRefresh = null;
    if (auto.checked) {
      state.autoRefresh = setInterval(async () => {
        try { await api(`/api/runs/${encodeURIComponent(r.run_id)}/status`, { method: "POST", quiet: true }); await api(`/api/runs/${encodeURIComponent(r.run_id)}/logs`, { method: "POST", quiet: true }); } catch {}
        if (location.hash.startsWith("#/runs") && state.selectedRun === r.run_id) renderRunDetail();
      }, 60000);
      toast("Tự động cập nhật mỗi 60 giây", "info");
    }
  };
}

function highlightLog(text) {
  return esc(text).replace(/^(epoch \d+\/\d+.*)$/gm, '<span class="ep">$1</span>');
}

async function runAction(act, run, btn) {
  const id = encodeURIComponent(run.run_id);
  try {
    if (act === "delete") {
      if (!confirm(`Xoá run ${run.run_id} khỏi lịch sử? (không xoá trên Kaggle)`)) return;
      await api(`/api/runs/${id}`, { method: "DELETE" });
      state.selectedRun = null; toast("Đã xoá", "ok"); return route();
    }
    busy(btn, true);
    if (act === "status") { const r = await api(`/api/runs/${id}/status`, { method: "POST" }); toast(`Trạng thái: ${STATUS_TEXT[r.status] || r.status}`, "ok"); }
    if (act === "logs") { await api(`/api/runs/${id}/logs`, { method: "POST" }); toast("Đã cập nhật log", "ok"); }
    if (act === "output") {
      const { job } = await api(`/api/runs/${id}/output`, { method: "POST" });
      toast("Đang tải output từ Kaggle…", "info");
      await waitJob(job);
      toast("Đã tải output (history, checkpoint)", "ok");
    }
    await loadRuns();
    route();
  } catch (e) { if (act === "output") toast(e.message, "err", 8000); }
  finally { busy(btn, false); }
}

/* ------------------------------------------------------------------ tensorboard */

async function renderTensorboard(view) {
  const tb = await api("/api/tensorboard", { quiet: true });
  view.innerHTML = `
  <div class="stack">
    <div class="grid g3">
      <div class="card step"><div class="num-c">1</div><h4>Xuất event file</h4><p>Chuyển lịch sử train của mọi run sang <code>runs/&lt;run&gt;/tb/</code>.</p><button class="btn sm" id="tb-export" style="margin-top:12px">Xuất cho tất cả run</button></div>
      <div class="card step"><div class="num-c">2</div><h4>Khởi động TensorBoard</h4><p>Chạy local ở cổng 6006, logdir = <code>runs/</code>.</p>
        <div class="row" style="margin-top:12px"><button class="btn sm primary" id="tb-start">Khởi động</button><button class="btn sm" id="tb-stop" ${tb.running ? "" : "disabled"}>Dừng</button></div></div>
      <div class="card step"><div class="num-c">3</div><h4>Hoặc tự chạy</h4><p>Trong thư mục <code>app/</code>:</p><div class="codebox" style="margin-top:8px">uv run tensorboard --logdir runs</div></div>
    </div>
    <div class="card">
      <div class="card-head"><div><h3>TensorBoard</h3><div class="sub">${tb.url ? `Đang chạy tại <a href="${tb.url}" target="_blank" rel="noopener">${tb.url} ↗</a>` : "Chưa chạy"}</div></div></div>
      ${tb.url ? `<iframe class="tb" src="${tb.url}"></iframe>` : `<div class="empty"><div class="big">TensorBoard chưa chạy</div>Xuất event file rồi bấm Khởi động.</div>`}
    </div>
  </div>`;
  $("#tb-export").onclick = async (e) => { busy(e.currentTarget, true); try { const r = await api("/api/tensorboard/export", { method: "POST" }); toast(`Đã xuất ${r.runs} run (${r.scalars} scalar)`, "ok"); } finally { busy(e.currentTarget, false); } };
  $("#tb-start").onclick = async (e) => { busy(e.currentTarget, true); try { const r = await api("/api/tensorboard/start", { method: "POST" }); if (r.note) toast(r.note, "info"); await sleep(2500); route(); } finally { busy(e.currentTarget, false); } };
  $("#tb-stop").onclick = async () => { await api("/api/tensorboard/stop", { method: "POST" }); route(); };
}

/* ------------------------------------------------------------------ predict */

async function renderPredict(view) {
  const m = await api("/api/models", { quiet: true });
  const P = state.predict;
  const studentOpts = [...m.students, ...m.uploads];
  view.innerHTML = `
  <div class="predict-layout">
    <div class="stack">
      <div class="card">
        <div class="card-head"><div><h3>Model</h3><div class="sub">Chọn một hoặc cả hai để so sánh</div></div></div>
        <div class="card-pad stack" style="gap:16px">
          <label class="field"><span>Teacher</span>
            <select id="sel-teacher"><option value="">— không dùng —</option>${m.teachers.map((t) => `<option value="${t.spec}" ${P.teacher === t.spec ? "selected" : ""}>${esc(t.label)}${t.ready ? "" : " · chưa tải"}</option>`).join("")}</select></label>
          <div id="teacher-dl"></div>
          <label class="field"><span>Student / checkpoint</span>
            <select id="sel-student"><option value="">— không dùng —</option>${studentOpts.map((s) => `<option value="${esc(s.spec)}" ${P.student === s.spec ? "selected" : ""}>${esc(s.label)}${s.best ? ` · best ${fmt(s.best)}` : ""}</option>`).join("")}</select>
            <span class="hint">${m.students.length ? "Checkpoint best của các run đã tải output." : "Chưa có run nào có checkpoint — có thể upload file .pth."}</span></label>
          <input type="file" id="ckpt-input" accept=".pth,.pt" hidden>
          <button class="btn" id="ckpt-btn"><svg viewBox="0 0 24 24"><path d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16"/></svg>Upload checkpoint (.pth)</button>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><div><h3>Ảnh đầu vào</h3><div class="sub">Ảnh được resize về 32×32 như CIFAR-100</div></div></div>
        <div class="card-pad stack" style="gap:14px">
          <div class="dropzone" id="img-drop"><svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 17l-5-5-9 8"/></svg>
            Kéo thả ảnh vào đây hoặc <b>bấm để chọn</b><input type="file" id="img-input" accept="image/*" hidden></div>
          <button class="btn" id="cifar-btn">🎲 Ảnh ngẫu nhiên từ CIFAR-100 test</button>
        </div>
      </div>
    </div>
    <div class="stack" id="pred-out">
      <div class="card"><div class="empty"><div class="big">Chưa có dự đoán</div>Chọn model và đưa một ảnh vào để xem top-5.</div></div>
    </div>
  </div>`;

  const teacherReady = (spec) => m.teachers.find((t) => t.spec === spec)?.ready;
  const renderDl = () => {
    const t = m.teachers.find((x) => x.spec === P.teacher);
    $("#teacher-dl").innerHTML = t && !t.ready ? `<div class="notice" style="align-items:center"><span style="flex:1">Checkpoint ${esc(MODEL_META[t.arch].name)} chưa có. Tải từ GitHub mdistiller (stream ~364 MB, chỉ giữ 1 file).</span><button class="btn sm primary" id="dl-btn">Tải về</button></div>` : "";
    const b = $("#dl-btn");
    if (b) b.onclick = async () => {
      busy(b, true, "Đang tải");
      try { const { job } = await api(`/api/teachers/${t.arch}/download`, { method: "POST" }); await waitJob(job); toast("Đã tải teacher", "ok"); renderPredict(view); }
      catch (e) { toast(e.message, "err", 8000); busy(b, false); }
    };
  };
  renderDl();
  $("#sel-teacher").onchange = (e) => { P.teacher = e.target.value; renderDl(); runPredict(teacherReady); };
  $("#sel-student").onchange = (e) => { P.student = e.target.value; runPredict(teacherReady); };
  $("#ckpt-btn").onclick = () => $("#ckpt-input").click();
  $("#ckpt-input").onchange = async (e) => {
    const file = e.target.files[0]; if (!file) return;
    const form = new FormData(); form.append("file", file);
    busy($("#ckpt-btn"), true, "Đang nạp");
    try { const meta = await api("/api/checkpoints", { method: "POST", form }); P.student = meta.spec; toast(`Đã nạp ${meta.label}`, "ok"); renderPredict(view); }
    finally { busy($("#ckpt-btn"), false); }
  };
  const drop = $("#img-drop"), input = $("#img-input");
  const setFile = (f) => { P.file = f; P.cifarIndex = null; P.previewUrl && URL.revokeObjectURL(P.previewUrl); P.previewUrl = URL.createObjectURL(f); runPredict(teacherReady); };
  drop.onclick = () => input.click();
  input.onchange = () => input.files[0] && setFile(input.files[0]);
  dropHandlers(drop, setFile);
  $("#cifar-btn").onclick = async (e) => {
    const b = e.currentTarget;
    busy(b, true, "Đang chuẩn bị CIFAR-100");
    try {
      const l = await api("/api/cifar/load", { method: "POST" });
      if (l.job) { toast("Đang tải CIFAR-100 test (~160 MB lần đầu)…", "info"); await waitJob(l.job); }
      const s = await api("/api/cifar/sample");
      P.file = null; P.cifarIndex = s.index; P.previewUrl = s.image;
      runPredict(teacherReady);
    } catch (err) { toast(err.message, "err", 8000); }
    finally { busy(b, false); }
  };
  if (P.file || P.cifarIndex != null) runPredict(teacherReady);
}

async function runPredict(teacherReady) {
  const P = state.predict, out = $("#pred-out");
  const specs = [P.teacher && teacherReady(P.teacher) ? P.teacher : "", P.student].filter(Boolean);
  if (!out || (!P.file && P.cifarIndex == null)) return;
  if (!specs.length) { out.innerHTML = `<div class="card"><div class="empty"><div class="big">Chưa chọn model</div>Chọn teacher (đã tải) hoặc student ở cột bên trái.</div></div>`; return; }
  out.innerHTML = `<div class="card"><div class="empty"><span class="spin" style="color:var(--accent)"></span><div style="margin-top:10px">Đang dự đoán…</div></div></div>`;
  const form = new FormData();
  form.append("specs", JSON.stringify(specs));
  if (P.file) form.append("file", P.file); else form.append("cifar_index", P.cifarIndex);
  let d;
  try { d = await api("/api/predict", { method: "POST", form }); }
  catch (e) { out.innerHTML = `<div class="card card-pad"><div class="errors">${esc(e.message)}</div></div>`; return; }
  const names = { teacher: "Teacher", run: "Student", upload: "Checkpoint" };
  const tops = d.results.map((r) => r.predictions[0].label);
  const agree = d.results.length === 2 ? tops[0] === tops[1] : null;
  out.innerHTML = `
    <div class="card card-pad">
      <div class="input-preview">
        <img class="orig" src="${P.previewUrl}" alt="input">
        <div><div class="muted small">Model nhìn thấy (32×32)</div><img src="${d.input32}" width="132" height="132" alt="32x32"></div>
        <div class="stack" style="gap:6px">
          ${d.truth ? `<div><div class="muted small">Nhãn thật</div><div style="font-size:20px;font-weight:700">${esc(d.truth)}</div></div>` : `<div class="muted small">Ảnh tự upload — không có nhãn thật</div>`}
          ${agree === null ? "" : `<div class="agree ${agree ? "yes" : "no"}">${agree ? "✓ Teacher và student cùng dự đoán" : "≠ Teacher và student dự đoán khác nhau"}</div>`}
        </div>
      </div>
    </div>
    <div class="result-grid">
      ${d.results.map((r) => `
        <div class="card">
          <div class="card-head"><div><h3>${names[r.spec.split(":")[0]] || "Model"} · ${esc(MODEL_META[r.arch]?.name || r.arch)}</h3><div class="sub">${esc(r.spec)}</div></div>
            ${r.correct === null ? "" : `<span class="badge ${r.correct ? "ok" : "error"}">${r.correct ? "Đúng" : "Sai"}</span>`}</div>
          <div class="card-pad">
            ${r.predictions.map((p, i) => `<div class="pred-row ${i === 0 ? "top" : ""} ${p.label === d.truth ? "truth" : ""}">
              <span class="lbl">${esc(p.label.replaceAll("_", " "))}</span><span class="bar"><div style="width:${(p.prob * 100).toFixed(1)}%"></div></span><span class="pct">${(p.prob * 100).toFixed(1)}%</span></div>`).join("")}
          </div>
        </div>`).join("")}
    </div>`;
}

/* ------------------------------------------------------------------ boot */

(async function boot() {
  initModal();
  state.opts = await api("/api/options");
  await Promise.all([loadSession(), loadRuns()]);
  window.addEventListener("hashchange", route);
  route();
})();
