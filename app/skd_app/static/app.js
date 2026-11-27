/* SKD Studio — simple 3-page dashboard: Huấn luyện · Kết quả · Dự đoán. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const NAME = { resnet20: "ResNet20", resnet32: "ResNet32", resnet56: "ResNet56", resnet110: "ResNet110" };
const PAIRS = {
  r56: { teacher: "resnet56", student: "resnet20", title: "ResNet56 → ResNet20", sub: "Teacher 72.41% · ~5 giờ GPU", paper: 72.50 },
  r110: { teacher: "resnet110", student: "resnet32", title: "ResNet110 → ResNet32", sub: "Teacher 74.31% · ~6.5 giờ GPU", paper: 74.84 },
};
const RECIPES = {
  E4: { title: "E4 · SKD gốc", sub: "Tái lập paper: KD 0.5 cho mọi mẫu" },
  F1b: { title: "F1b · Teacher-aware", sub: "KD 0.6 khi teacher đúng, 0.5 khi sai" },
};
const STATUS = { queued: "Đang chờ", running: "Đang chạy", complete: "Hoàn thành", error: "Lỗi", push_failed: "Push lỗi", cancelacknowledged: "Đã huỷ", unknown: "Không rõ" };

const state = { session: { connected: false }, runs: [], selected: null, chart: null, predict: { teacher: "", student: "", file: null, cifarIndex: null, preview: null } };
const form = { pair: "r56", recipe: "F1b", epochs: 300, learning_rate: 0.1, seed: 2022, smoke_test: false, lr_milestones: "100,188,250", batch_size: 64, temperature: 4 };

if (window.Chart) { Chart.defaults.font.family = "Inter, system-ui, sans-serif"; Chart.defaults.color = "#7a8296"; Chart.defaults.borderColor = "#eceef5"; }

/* ------------------------------------------------------------------ helpers */

async function api(path, { method = "GET", json, body, quiet = false } = {}) {
  const opts = { method, headers: {}, credentials: "same-origin" };
  if (json !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(json); }
  if (body) opts.body = body;
  const res = await fetch(path, opts);
  const data = (res.headers.get("content-type") || "").includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = typeof data?.detail === "string" ? data.detail : `Lỗi HTTP ${res.status}`;
    if (!quiet) toast(msg, "err");
    throw new Error(msg);
  }
  return data;
}

function toast(msg, kind = "info", ms = 4500) {
  const el = Object.assign(document.createElement("div"), { className: `toast ${kind}`, textContent: msg });
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function waitJob(id) {
  for (;;) {
    const job = await api(`/api/jobs/${id}`, { quiet: true });
    if (job.state === "done") return job;
    if (job.state === "error") throw new Error(job.message || "Lỗi");
    await sleep(1000);
  }
}

function busy(btn, on, label = "") {
  if (!btn) return;
  if (on) { btn.dataset.html = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spin"></span>${label}`; }
  else { btn.disabled = false; if (btn.dataset.html) btn.innerHTML = btn.dataset.html; }
}

const badge = (s) => `<span class="badge ${esc(s || "unknown")}">${esc(STATUS[s] || s || "–")}</span>`;
const pairOf = (c = {}) => Object.values(PAIRS).find((p) => p.teacher === c.teacher && p.student === c.student);
const runTitle = (r) => `${NAME[r.config?.teacher] || "?"} → ${NAME[r.config?.student] || "?"}`;

function card(choiceKey, value, { title, sub }, selected) {
  return `<button type="button" class="choice ${selected ? "selected" : ""}" data-k="${choiceKey}" data-v="${value}">
    <div class="c-title"><span>${esc(title)}</span><span class="check"></span></div><div class="c-sub">${esc(sub)}</div></button>`;
}

function dropzone(el, onFile) {
  el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("over"); });
  el.addEventListener("dragleave", () => el.classList.remove("over"));
  el.addEventListener("drop", (e) => { e.preventDefault(); el.classList.remove("over"); e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]); });
}

/* ------------------------------------------------------------------ Kaggle account */

async function loadSession() {
  state.session = await api("/api/session", { quiet: true });
  $("#account-dot").classList.toggle("on", state.session.connected);
  $("#account-label").textContent = state.session.connected ? state.session.username : "Kết nối Kaggle";
  $("#cred-logout").hidden = !state.session.connected;
}

function initAccount() {
  const modal = $("#modal");
  const close = () => { modal.hidden = true; $("#cred-key").value = ""; };
  $("#account-btn").onclick = () => { $("#cred-user").value = state.session.username || ""; modal.hidden = false; $("#cred-user").focus(); };
  $$("[data-close]").forEach((b) => (b.onclick = close));
  modal.onclick = (e) => e.target === modal && close();
  document.addEventListener("keydown", (e) => e.key === "Escape" && close());

  const save = async (username, key) => {
    const btn = $("#cred-save");
    busy(btn, true, "Đang kiểm tra");
    try {
      await api("/api/session", { method: "POST", json: { username, key } });
      const v = await api("/api/session/verify", { method: "POST", quiet: true }).catch((e) => ({ ok: false, message: e.message }));
      await loadSession();
      close();
      toast(v.ok ? `Đã kết nối Kaggle: ${username}` : `Kết nối lỗi: ${v.message}`, v.ok ? "ok" : "err", 7000);
      route();
    } finally { busy(btn, false); }
  };
  $("#cred-save").onclick = () => {
    const u = $("#cred-user").value.trim(), k = $("#cred-key").value.trim();
    u && k ? save(u, k) : toast("Nhập đủ username và API key", "err");
  };
  $("#cred-key").onkeydown = (e) => e.key === "Enter" && $("#cred-save").click();
  $("#cred-logout").onclick = async () => { await api("/api/session", { method: "DELETE" }); await loadSession(); close(); route(); };

  const input = $("#kaggle-json-input");
  const readJson = async (f) => {
    try { const d = JSON.parse(await f.text()); save(String(d.username), String(d.key)); } catch { toast("kaggle.json không hợp lệ", "err"); }
  };
  $("#kaggle-json-pick").onclick = (e) => { e.preventDefault(); input.click(); };
  $("#kaggle-json-drop").onclick = (e) => e.target.tagName !== "A" && input.click();
  input.onchange = () => input.files[0] && readJson(input.files[0]);
  dropzone($("#kaggle-json-drop"), readJson);
}

/* ------------------------------------------------------------------ router */

const PAGES = {
  train: { title: "Huấn luyện", sub: "Chọn cấu hình rồi train trên GPU Kaggle", render: renderTrain },
  runs: { title: "Kết quả", sub: "Các lần train và độ chính xác so với paper", render: renderRuns },
  predict: { title: "Dự đoán", sub: "So sánh teacher và student trên ảnh", render: renderPredict },
};

async function route() {
  const name = location.hash.replace("#/", "") || "train";
  const page = PAGES[name] || PAGES.train;
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === name));
  $("#page-title").textContent = page.title;
  $("#page-sub").textContent = page.sub;
  if (state.chart) { state.chart.destroy(); state.chart = null; }
  await page.render($("#view"));
}

async function loadRuns() {
  state.runs = await api("/api/runs", { quiet: true }).catch(() => []);
  $("#nav-runs-count").textContent = state.runs.length || "";
}

/* ------------------------------------------------------------------ Huấn luyện */

function renderTrain(view) {
  const connected = state.session.connected;
  const pair = PAIRS[form.pair];
  view.innerHTML = `
  <div class="train-layout">
    <div class="card card-pad">
      <div class="form-block"><div class="section-title">1 · Cặp model (CIFAR-100)</div>
        <div class="choice-grid">${Object.entries(PAIRS).map(([k, p]) => card("pair", k, p, k === form.pair)).join("")}</div></div>
      <div class="form-block"><div class="section-title">2 · Cách distill</div>
        <div class="choice-grid">${Object.entries(RECIPES).map(([k, r]) => card("recipe", k, r, k === form.recipe)).join("")}</div></div>
      <div class="form-block"><div class="section-title">3 · Tham số</div>
        <div class="params">
          <label class="field"><span>Epochs</span><input type="number" data-f="epochs" min="1" max="400" value="${form.epochs}"></label>
          <label class="field"><span>Learning rate</span><input type="number" data-f="learning_rate" min="0.001" max="1" step="0.01" value="${form.learning_rate}"></label>
          <label class="field"><span>Seed</span><input type="number" data-f="seed" min="0" value="${form.seed}"></label>
        </div>
        <details style="margin-top:16px"><summary>Nâng cao</summary>
          <div class="params" style="margin-top:12px">
            <label class="field"><span>LR giảm ×0.1 tại epoch</span><input data-f="lr_milestones" value="${esc(form.lr_milestones)}"></label>
            <label class="field"><span>Batch size</span><select data-f="batch_size">${[32, 64, 128, 256].map((b) => `<option ${b === form.batch_size ? "selected" : ""}>${b}</option>`).join("")}</select></label>
            <label class="field"><span>Temperature T</span><input type="number" data-f="temperature" min="0.5" max="20" step="0.5" value="${form.temperature}"></label>
          </div></details>
        <div class="switch-row" style="margin-top:16px">
          <div><b>Chạy thử nhanh</b><div class="muted small">Chỉ vài batch mỗi epoch — kiểm tra mọi thứ chạy được trong vài phút.</div></div>
          <label class="switch"><input type="checkbox" data-f="smoke_test" ${form.smoke_test ? "checked" : ""}><span></span></label>
        </div>
      </div>
    </div>

    <div class="summary card card-pad stack" style="gap:16px">
      <div class="pipeline">
        <div class="node"><div class="n-role">Teacher</div><div class="n-name">${NAME[pair.teacher]}</div></div>
        <div class="arrow">→</div>
        <div class="node"><div class="n-role">Student</div><div class="n-name">${NAME[pair.student]}</div></div>
      </div>
      <dl class="kv">
        <dt>Cách distill</dt><dd>${form.recipe}</dd>
        <dt>Paper SKD</dt><dd>${fmt(pair.paper)}%</dd>
        <dt>Thời gian ước tính</dt><dd id="sum-hours">–</dd>
      </dl>
      <div id="sum-errors"></div>
      ${connected ? "" : `<div class="notice"><span>🔑 Bấm <b>Kết nối Kaggle</b> ở góc trên phải để train.</span></div>`}
      <button class="btn primary block" id="btn-train" ${connected ? "" : "disabled"}>Train trên Kaggle</button>
    </div>
  </div>`;

  $$(".choice", view).forEach((b) => (b.onclick = () => { form[b.dataset.k] = b.dataset.v; renderTrain(view); }));
  $$("[data-f]", view).forEach((el) => (el.oninput = el.onchange = () => {
    const k = el.dataset.f;
    form[k] = el.type === "checkbox" ? el.checked : (el.type === "number" || k === "batch_size") ? Number(el.value) : el.value;
    preview();
  }));
  $("#btn-train").onclick = submit;
  preview();
}

function payload() {
  const { pair, ...rest } = form;
  return { ...rest, teacher: PAIRS[pair].teacher, student: PAIRS[pair].student };
}

async function preview() {
  const p = await api("/api/preview", { method: "POST", json: payload(), quiet: true }).catch(() => null);
  if (!p || !$("#sum-hours")) return;
  $("#sum-hours").textContent = `~${p.hours} giờ GPU`;
  $("#sum-errors").innerHTML = p.errors.length ? `<div class="errors">${p.errors.map(esc).join("<br>")}</div>` : "";
  $("#btn-train").disabled = !state.session.connected || p.errors.length > 0;
}

async function submit() {
  const btn = $("#btn-train");
  busy(btn, true, "Đang gửi lên Kaggle");
  try {
    const rec = await api("/api/runs", { method: "POST", json: payload() });
    toast(rec.status === "push_failed" ? "Gửi lên Kaggle thất bại — xem chi tiết ở trang Kết quả" : "Đã gửi lên Kaggle 🚀", rec.status === "push_failed" ? "err" : "ok", 6000);
    state.selected = rec.run_id;
    location.hash = "#/runs";
  } catch { /* toast shown */ } finally { busy(btn, false); }
}

/* ------------------------------------------------------------------ Kết quả */

async function renderRuns(view) {
  await loadRuns();
  if (!state.runs.find((r) => r.run_id === state.selected)) state.selected = state.runs[0]?.run_id || null;
  view.innerHTML = `
  <div class="stack">
    <div class="row">
      <button class="btn" id="btn-import">Import log có sẵn</button>
      <input type="file" id="import-file" accept=".log,.json,.txt,.csv" hidden>
      <button class="btn" id="btn-tb" ${state.runs.length ? "" : "disabled"}>Mở TensorBoard ↗</button>
      <span class="spacer"></span>
      <a class="btn primary" href="#/train">+ Train mới</a>
    </div>
    <div class="card">${table()}</div>
    <div id="detail"></div>
  </div>`;

  $("#btn-import").onclick = () => $("#import-file").click();
  $("#import-file").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const body = new FormData(); body.append("file", f);
    try { const r = await api("/api/runs/import", { method: "POST", body }); state.selected = r.run_id; toast("Đã import", "ok"); route(); } catch {}
  };
  $("#btn-tb").onclick = async (e) => {
    const btn = e.currentTarget;
    busy(btn, true, "Đang chuẩn bị");
    try { const r = await api("/api/tensorboard", { method: "POST" }); await sleep(2500); window.open(r.url, "_blank"); }
    catch {} finally { busy(btn, false); }
  };
  $$("tbody tr", view).forEach((tr) => (tr.onclick = () => {
    state.selected = tr.dataset.run;
    $$("tbody tr", view).forEach((x) => x.classList.toggle("sel", x === tr));
    renderDetail();
  }));
  renderDetail();
}

function table() {
  if (!state.runs.length) return `<div class="empty"><div class="big">Chưa có lần train nào</div>Bấm <b>Train mới</b>, hoặc <b>Import log có sẵn</b> từ một lần chạy trước.</div>`;
  return `<div style="overflow:auto"><table class="table">
    <thead><tr><th>Lần train</th><th>Cách distill</th><th>Trạng thái</th><th class="num">Best</th><th class="num">TB10</th><th class="num">Paper</th></tr></thead>
    <tbody>${state.runs.map((r) => {
      const m = r.metrics || {}, p = pairOf(r.config);
      return `<tr data-run="${esc(r.run_id)}" class="${r.run_id === state.selected ? "sel" : ""}">
        <td><div class="run-name">${esc(runTitle(r))}</div><div class="run-sub">${esc(r.label || r.run_id)}</div></td>
        <td><span class="tag">${esc(r.config?.recipe || "–")}</span></td>
        <td>${badge(r.status)}</td>
        <td class="num"><b>${fmt(m.best)}</b></td><td class="num">${fmt(m.tb10)}</td><td class="num muted">${p ? fmt(p.paper) : "–"}</td></tr>`;
    }).join("")}</tbody></table></div>`;
}

async function renderDetail() {
  const box = $("#detail");
  if (!box || !state.selected) return;
  if (state.chart) { state.chart.destroy(); state.chart = null; }
  const d = await api(`/api/runs/${encodeURIComponent(state.selected)}`, { quiet: true }).catch(() => null);
  if (!d) { box.innerHTML = ""; return; }
  const r = d.run, s = d.summary || {}, p = pairOf(r.config);
  const isKaggle = r.source === "kaggle";
  const done = s.epochs_done || 0, total = s.epochs_total || r.config?.epochs || 0;
  const diff = p && s.best != null ? s.best - p.paper : null;
  box.innerHTML = `
  <div class="card">
    <div class="card-head">
      <div><div class="row"><h3>${esc(runTitle(r))}</h3>${badge(r.status)}<span class="tag">${esc(r.config?.recipe || "")}</span></div>
        <div class="sub">${esc(r.label || r.run_id)}${isKaggle && r.url ? ` · <a href="${esc(r.url)}" target="_blank" rel="noopener">xem trên Kaggle ↗</a>` : ""}</div></div>
      <div class="row">
        ${isKaggle ? `<button class="btn sm primary" id="btn-refresh" ${state.session.connected ? "" : "disabled title='Kết nối Kaggle trước'"}>Cập nhật</button>` : ""}
        <button class="btn sm ghost danger" id="btn-del">Xoá</button>
      </div>
    </div>
    <div class="card-pad stack">
      ${r.status === "push_failed" ? `<div class="errors">${esc(r.push_output || "Gửi lên Kaggle thất bại")}</div>` : ""}
      ${isKaggle && ["queued", "running"].includes(r.status) ? `<div class="notice"><span>Đang train trên Kaggle. Kaggle chỉ trả kết quả khi chạy xong — bấm <b>Cập nhật</b> để kiểm tra; khi hoàn thành app tự tải kết quả về.</span></div>` : ""}
      <div class="grid g4">
        <div class="card kpi accent"><div class="label">Best top-1</div><div class="value">${fmt(s.best)}</div><div class="delta">${s.best_epoch ? `epoch ${s.best_epoch}` : "&nbsp;"}</div></div>
        <div class="card kpi"><div class="label">Final</div><div class="value">${fmt(s.final)}</div><div class="delta flat">epoch cuối</div></div>
        <div class="card kpi"><div class="label">TB10</div><div class="value">${fmt(s.tb10)}</div><div class="delta flat">TB 10 epoch cuối</div></div>
        <div class="card kpi"><div class="label">So với paper</div><div class="value">${diff == null ? "–" : (diff >= 0 ? "+" : "") + diff.toFixed(2)}</div><div class="delta flat">${p ? `paper ${fmt(p.paper)}` : "&nbsp;"}</div></div>
      </div>
      ${d.series.epoch ? `<div class="muted small">${done}/${total || "?"} epoch</div><div class="chart-box"><canvas id="chart"></canvas></div>` : ""}
      ${d.log_tail ? `<details><summary>Log</summary><div class="log" style="margin-top:10px">${esc(d.log_tail)}</div></details>` : ""}
    </div>
  </div>`;

  if (d.series.epoch) {
    const S = d.series;
    state.chart = new Chart($("#chart"), {
      type: "line",
      data: {
        labels: S.epoch,
        datasets: [
          { label: "Test top-1", data: S.test_top1, borderColor: "#4f46e5", backgroundColor: "rgba(79,70,229,.08)", fill: true, tension: .25, pointRadius: 0, borderWidth: 2.2 },
          ...(S.train_top1 ? [{ label: "Train top-1", data: S.train_top1, borderColor: "#a78bfa", borderDash: [5, 4], tension: .25, pointRadius: 0, borderWidth: 1.6 }] : []),
          ...(p ? [{ label: `Paper SKD ${p.paper}`, data: S.epoch.map(() => p.paper), borderColor: "#f59e0b", borderDash: [6, 5], pointRadius: 0, borderWidth: 1.5 }] : []),
        ],
      },
      options: {
        maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        scales: { y: { title: { display: true, text: "top-1 (%)" } }, x: { title: { display: true, text: "epoch" }, ticks: { maxTicksLimit: 12 } } },
        plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8 } } },
      },
    });
  }

  const ref = $("#btn-refresh");
  if (ref) ref.onclick = async () => {
    busy(ref, true);
    try {
      const res = await api(`/api/runs/${encodeURIComponent(r.run_id)}/refresh`, { method: "POST" });
      if (res.job) { toast("Run đã xong — đang tải kết quả…", "info"); await waitJob(res.job); toast("Đã tải kết quả", "ok"); }
      else toast(`Trạng thái: ${STATUS[res.status] || res.status}`, "ok");
      route();
    } catch (e) { toast(e.message, "err", 8000); busy(ref, false); }
  };
  $("#btn-del").onclick = async () => {
    if (!confirm("Xoá lần train này khỏi danh sách? (không xoá trên Kaggle)")) return;
    await api(`/api/runs/${encodeURIComponent(r.run_id)}`, { method: "DELETE" });
    state.selected = null; route();
  };
}

/* ------------------------------------------------------------------ Dự đoán */

async function renderPredict(view) {
  const m = await api("/api/models", { quiet: true });
  const P = state.predict;
  const students = [...m.students, ...m.uploads];
  view.innerHTML = `
  <div class="predict-layout">
    <div class="stack">
      <div class="card card-pad stack" style="gap:16px">
        <label class="field"><span>Teacher</span>
          <select id="sel-t"><option value="">— không dùng —</option>${m.teachers.map((t) => `<option value="${t.spec}" ${P.teacher === t.spec ? "selected" : ""}>${NAME[t.arch]}${t.ready ? "" : " (cần tải)"}</option>`).join("")}</select></label>
        <div id="t-dl"></div>
        <label class="field"><span>Student</span>
          <select id="sel-s"><option value="">— không dùng —</option>${students.map((s) => `<option value="${esc(s.spec)}" ${P.student === s.spec ? "selected" : ""}>${esc(s.label)}</option>`).join("")}</select></label>
        <input type="file" id="ckpt" accept=".pth,.pt" hidden>
        <button class="btn sm" id="btn-ckpt">Hoặc upload checkpoint .pth</button>
      </div>
      <div class="card card-pad stack" style="gap:12px">
        <div class="dropzone" id="drop"><svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 17l-5-5-9 8"/></svg>
          Kéo thả ảnh vào đây hoặc <b>bấm để chọn</b><input type="file" id="img" accept="image/*" hidden></div>
        <button class="btn" id="btn-cifar">🎲 Ảnh ngẫu nhiên CIFAR-100</button>
      </div>
    </div>
    <div class="stack" id="out"><div class="card"><div class="empty"><div class="big">Chọn model và đưa ảnh vào</div>Kết quả top-5 sẽ hiện ở đây.</div></div></div>
  </div>`;

  const ready = (spec) => !spec.startsWith("teacher:") || m.teachers.find((t) => t.spec === spec)?.ready;
  const showDl = () => {
    const t = m.teachers.find((x) => x.spec === P.teacher);
    $("#t-dl").innerHTML = t && !t.ready ? `<button class="btn sm primary" id="btn-dl">Tải teacher ${NAME[t.arch]} (~1 lần)</button>` : "";
    const b = $("#btn-dl");
    if (b) b.onclick = async () => {
      busy(b, true, "Đang tải");
      try { const { job } = await api(`/api/teachers/${t.arch}/download`, { method: "POST" }); await waitJob(job); toast("Đã tải teacher", "ok"); renderPredict(view); }
      catch (e) { toast(e.message, "err", 8000); busy(b, false); }
    };
  };
  const run = () => predict(ready);
  showDl();
  $("#sel-t").onchange = (e) => { P.teacher = e.target.value; showDl(); run(); };
  $("#sel-s").onchange = (e) => { P.student = e.target.value; run(); };
  $("#btn-ckpt").onclick = () => $("#ckpt").click();
  $("#ckpt").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const body = new FormData(); body.append("file", f);
    try { const meta = await api("/api/checkpoints", { method: "POST", body }); P.student = meta.spec; renderPredict(view); } catch {}
  };
  const setFile = (f) => { P.file = f; P.cifarIndex = null; P.preview = URL.createObjectURL(f); run(); };
  $("#drop").onclick = () => $("#img").click();
  $("#img").onchange = (e) => e.target.files[0] && setFile(e.target.files[0]);
  dropzone($("#drop"), setFile);
  $("#btn-cifar").onclick = async (e) => {
    const b = e.currentTarget;
    busy(b, true, "Đang tải CIFAR-100");
    try {
      const l = await api("/api/cifar/load", { method: "POST" });
      if (l.job) await waitJob(l.job);
      const s = await api("/api/cifar/sample");
      Object.assign(P, { file: null, cifarIndex: s.index, preview: s.image });
      run();
    } catch (err) { toast(err.message, "err", 8000); } finally { busy(b, false); }
  };
  if (P.file || P.cifarIndex != null) run();
}

async function predict(ready) {
  const P = state.predict, out = $("#out");
  const specs = [P.teacher, P.student].filter((s) => s && ready(s));
  if (!out || (!P.file && P.cifarIndex == null) || !specs.length) return;
  out.innerHTML = `<div class="card"><div class="empty"><span class="spin" style="color:var(--accent)"></span></div></div>`;
  const body = new FormData();
  body.append("specs", JSON.stringify(specs));
  P.file ? body.append("file", P.file) : body.append("cifar_index", P.cifarIndex);
  let d;
  try { d = await api("/api/predict", { method: "POST", body }); } catch (e) { out.innerHTML = `<div class="card card-pad"><div class="errors">${esc(e.message)}</div></div>`; return; }
  const same = d.results.length === 2 ? d.results[0].predictions[0].label === d.results[1].predictions[0].label : null;
  out.innerHTML = `
    <div class="card card-pad"><div class="input-preview">
      <img class="orig" src="${P.preview}" alt="">
      <div>${d.truth ? `<div class="muted small">Nhãn thật</div><div style="font-size:22px;font-weight:700">${esc(d.truth.replaceAll("_", " "))}</div>` : `<div class="muted small">Ảnh của bạn (không có nhãn)</div>`}
        ${same === null ? "" : `<div class="agree ${same ? "yes" : "no"}" style="margin-top:10px">${same ? "✓ Teacher và student đoán giống nhau" : "≠ Teacher và student đoán khác nhau"}</div>`}</div>
    </div></div>
    <div class="result-grid">${d.results.map((r) => `
      <div class="card"><div class="card-head"><h3>${r.spec.startsWith("teacher:") ? "Teacher" : "Student"} · ${NAME[r.arch] || r.arch}</h3>
        ${r.correct === null ? "" : `<span class="badge ${r.correct ? "ok" : "error"}">${r.correct ? "Đúng" : "Sai"}</span>`}</div>
        <div class="card-pad">${r.predictions.map((p, i) => `<div class="pred-row ${i === 0 ? "top" : ""} ${p.label === d.truth ? "truth" : ""}">
          <span class="lbl">${esc(p.label.replaceAll("_", " "))}</span><span class="bar"><div style="width:${(p.prob * 100).toFixed(1)}%"></div></span><span class="pct">${(p.prob * 100).toFixed(1)}%</span></div>`).join("")}</div></div>`).join("")}
    </div>`;
}

/* ------------------------------------------------------------------ boot */

(async () => {
  initAccount();
  await Promise.all([loadSession(), loadRuns()]);
  window.addEventListener("hashchange", route);
  route();
})();
