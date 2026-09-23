/* ==========================================================================
   Ambiguity-Safe Inverse Decoding — Frontend logic
   Talks to the Flask API served from the same origin (see backend/app.py).
   ========================================================================== */

const API = {
  health:  "/api/health",
  presets: "/api/presets",
  preset:  (k) => `/api/preset/${encodeURIComponent(k)}`,
  analyze: "/api/analyze",
};

const $ = (id) => document.getElementById(id);

// -------------------------------------------------------------------------
// Parsing helpers: textarea (one row per line, comma-separated features)
// -------------------------------------------------------------------------
function parseGrid(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => line.split(",").map((v) => v.trim()));
}

function gridToText(grid) {
  if (!grid) return "";
  return grid.map((row) => row.join(", ")).join("\n");
}

// -------------------------------------------------------------------------
// Theme toggle
// -------------------------------------------------------------------------
function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  const btn = $("themeToggle");
  const icon = btn.querySelector(".theme-icon");
  const sync = () => {
    icon.textContent =
      document.documentElement.getAttribute("data-theme") === "dark" ? "☀" : "☾";
  };
  sync();
  btn.addEventListener("click", () => {
    const next =
      document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    sync();
  });
}

// -------------------------------------------------------------------------
// Health check
// -------------------------------------------------------------------------
async function checkHealth() {
  const el = $("healthStatus");
  try {
    const r = await fetch(API.health);
    const j = await r.json();
    el.textContent = `API ${j.status} · v${j.version}`;
    el.className = "health ok";
  } catch (e) {
    el.textContent = "API offline — start backend/app.py";
    el.className = "health err";
  }
}

// -------------------------------------------------------------------------
// Presets
// -------------------------------------------------------------------------
let PRESET_CACHE = {};

async function loadPresets() {
  try {
    const r = await fetch(API.presets);
    const j = await r.json();
    const sel = $("presetSelect");
    j.presets.forEach((p) => {
      PRESET_CACHE[p.key] = p;
      const opt = document.createElement("option");
      opt.value = p.key;
      opt.textContent = p.name;
      sel.appendChild(opt);
    });
  } catch (e) {
    /* health check surfaces the API error */
  }
}

async function applyPreset(key) {
  $("presetDesc").textContent = "";
  if (!key) return;
  const r = await fetch(API.preset(key));
  const { preset } = await r.json();
  $("presetDesc").textContent = preset.description;
  $("featureNames").value = (preset.feature_names || []).join(", ");
  $("trainData").value = gridToText(preset.train_data);
  $("testData").value = gridToText(preset.test_data);
  $("groundTruth").value = gridToText(preset.ground_truth);
  $("dropSelect").value = preset.drop === null ? "null" : preset.drop;
  $("handleUnknown").value = preset.handle_unknown || "ignore";
}

// -------------------------------------------------------------------------
// Analyze
// -------------------------------------------------------------------------
async function analyze() {
  const btn = $("analyzeBtn");
  const err = $("errorBox");
  err.hidden = true;

  const train = parseGrid($("trainData").value);
  const test = parseGrid($("testData").value);
  const gt = parseGrid($("groundTruth").value);
  const names = $("featureNames").value
    .split(",").map((s) => s.trim()).filter(Boolean);

  if (!train.length || !test.length) {
    return showError("Training data and test data are both required.");
  }

  const body = {
    train_data: train,
    test_data: test,
    drop: $("dropSelect").value,
    handle_unknown: $("handleUnknown").value,
  };
  if (gt.length) body.ground_truth = gt;
  if (names.length) body.feature_names = names;

  btn.disabled = true;
  btn.textContent = "Analyzing…";
  try {
    const r = await fetch(API.analyze, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!j.success) throw new Error(j.error || "Analysis failed");
    render(j);
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
}

function showError(msg) {
  const err = $("errorBox");
  err.textContent = msg;
  err.hidden = false;
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
const STATUS_META = {
  SAFE:      { cls: "safe",      icon: "✓", label: "SAFE" },
  AMBIGUOUS: { cls: "ambiguous", icon: "⚠", label: "AMBIGUOUS" },
  UNKNOWN:   { cls: "unknown",   icon: "?", label: "UNKNOWN" },
};

function statusBadge(status) {
  const m = STATUS_META[status] || STATUS_META.UNKNOWN;
  return `<span class="badge badge-${m.cls}">${m.icon} ${m.label}</span>`;
}

function render(data) {
  $("placeholder").hidden = true;
  $("resultsBody").hidden = false;

  renderKeyFinding(data.summary);
  renderTiles(data.metrics);
  renderStatusBars(data.metrics);
  renderTable(data.results);
  renderMetadata(data.metadata, data.config);
}

function renderKeyFinding(summary) {
  $("keyFinding").innerHTML =
    `<strong>Key finding.</strong> ${escapeHtml(summary.key_finding)}`;
}

function renderTiles(m) {
  const tiles = [
    { value: m.total_samples,   label: "Test samples",       cls: "" },
    { value: m.safe_count,      label: "SAFE",               cls: "good" },
    { value: m.ambiguous_count, label: "AMBIGUOUS",          cls: "critical" },
    { value: m.unknown_count,   label: "UNKNOWN",            cls: "warning" },
    { value: m.baseline_incorrect, label: "Silent errors (baseline)", cls: "critical" },
    { value: m.detection_rate + "%", label: "Detection rate", cls: "good" },
  ];
  $("tiles").innerHTML = tiles.map((t) => `
    <div class="tile ${t.cls}">
      <div class="tile-value">${t.value}</div>
      <div class="tile-label">${t.label}</div>
    </div>`).join("");
}

function renderStatusBars(m) {
  const total = m.total_samples || 1;
  const rows = [
    { name: "SAFE",      count: m.safe_count,      dot: "dot-safe",      fill: "fill-safe",      icon: "✓" },
    { name: "AMBIGUOUS", count: m.ambiguous_count, dot: "dot-ambiguous", fill: "fill-ambiguous", icon: "⚠" },
    { name: "UNKNOWN",   count: m.unknown_count,   dot: "dot-unknown",   fill: "fill-unknown",   icon: "?" },
  ];
  $("statusBars").innerHTML = rows.map((r) => {
    const pct = (r.count / total) * 100;
    return `
      <div class="status-row">
        <span class="status-name"><span class="status-dot ${r.dot}"></span>${r.icon} ${r.name}</span>
        <div class="status-track"><div class="status-fill ${r.fill}" style="width:${pct}%"></div></div>
        <span class="status-count">${r.count}</span>
      </div>`;
  }).join("");
}

function renderTable(results) {
  const tbody = $("resultsTable").querySelector("tbody");
  tbody.innerHTML = results.map((r) => {
    const input = r.input ? r.input.join(", ") : "—";
    const encoded = `[${r.encoded.join(", ")}]`;
    const baseline = r.baseline_decoded.map((v) => v == null ? "∅" : v).join(", ");
    let baselineCls = "";
    if (r.baseline_correct === true) baselineCls = "cell-right";
    else if (r.baseline_correct === false) baselineCls = "cell-wrong";
    const baselineMark = r.baseline_correct === false ? " ✗" : (r.baseline_correct === true ? " ✓" : "");

    const safe = r.safe_output
      .map((v) => v == null ? `<span class="cell-muted">withheld</span>` : escapeHtml(v))
      .join(", ");
    const possible = r.possible_values
      .map((pv) => `[${pv.map(escapeHtml).join(", ")}]`).join(" ");

    return `
      <tr>
        <td>${r.sample_index}</td>
        <td>${escapeHtml(input)}</td>
        <td><code>${encoded}</code></td>
        <td class="${baselineCls}">${escapeHtml(baseline)}${baselineMark}</td>
        <td>${statusBadge(r.row_status)}</td>
        <td>${safe}</td>
        <td class="cell-muted">${possible}</td>
      </tr>`;
  }).join("");
}

function renderMetadata(meta, config) {
  const dropTxt = config.drop === null ? "None" : config.drop;
  let html = `<div class="meta-config">
      <span>drop = <code>${dropTxt}</code></span>
      <span>handle_unknown = <code>${meta.handle_unknown}</code></span>
      <span>features = <code>${meta.n_features}</code></span>
      <span>encoded columns = <code>${meta.total_encoded_columns}</code></span>
    </div>`;

  html += meta.features.map((f) => {
    const flag = f.can_produce_all_zeros;
    return `<div class="meta-feature">
        <div class="meta-feature-name">${escapeHtml(f.feature_name)}
          ${f.is_binary ? '<span class="cell-muted">(binary)</span>' : ""}</div>
        <div class="meta-feature-detail">
          categories: ${f.categories.map(escapeHtml).join(", ")}<br/>
          dropped: <code>${f.dropped_category == null ? "none" : escapeHtml(f.dropped_category)}</code>
          &middot; encoded cols: <code>${f.n_encoded_columns}</code>
          &middot; ambiguity possible:
          <span class="meta-flag ${flag ? "on" : "off"}">${flag ? "YES ⚠" : "no ✓"}</span>
        </div>
      </div>`;
  }).join("");

  $("metadataBox").innerHTML = html;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// -------------------------------------------------------------------------
// Init
// -------------------------------------------------------------------------
async function init() {
  initTheme();
  checkHealth();
  await loadPresets();

  $("presetSelect").addEventListener("change", (e) => applyPreset(e.target.value));
  $("analyzeBtn").addEventListener("click", analyze);

  // Load the canonical preset by default so the page shows the core problem.
  if (PRESET_CACHE["gender_binary"]) {
    $("presetSelect").value = "gender_binary";
    await applyPreset("gender_binary");
  }
}

document.addEventListener("DOMContentLoaded", init);
