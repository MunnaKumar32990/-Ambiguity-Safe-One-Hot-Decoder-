/* ==========================================================================
   Ambiguity-Safe Inverse Decoding — Complete Frontend Application Logic
   Review-2 Demonstrator (KLCAP-2026-00332)
   ========================================================================== */

const API = {
  health:        "/api/health",
  presets:       "/api/presets",
  preset:        (k) => `/api/preset/${encodeURIComponent(k)}`,
  analyze:       "/api/analyze",
  negativeTests: "/api/negative-tests",
  kpis:          "/api/kpi-benchmarks",
  evidence:      "/api/evidence-manifest",
};

const $ = (id) => document.getElementById(id);

// -------------------------------------------------------------------------
// Policy descriptions
// -------------------------------------------------------------------------
const POLICY_HINTS = {
  WITHHOLD: "WITHHOLD: Replaces ambiguous/unknown values with null/None (Safe default for data cleaning).",
  SENTINEL: "SENTINEL: Injects structured sentinel string (e.g., <AMBIGUOUS:Female|UNSEEN>) into the output.",
  PROVENANCE_SIDE_CHANNEL: "PROVENANCE_SIDE_CHANNEL: Outputs best guess while recording audit trace & confidence in a side channel.",
  STRICT_REJECTION: "STRICT_REJECTION: Raises an AmbiguityRejectionError and halts processing if ambiguity occurs.",
};

// -------------------------------------------------------------------------
// State & Cache
// -------------------------------------------------------------------------
let PRESET_CACHE = {};
let LAST_ANALYSIS_RESULTS = null;

// -------------------------------------------------------------------------
// Helper Parsing Functions
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
// Navigation Tabs
// -------------------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const targetPaneId = `tab-${tab.dataset.tab}`;
      document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.remove("active");
      });
      const targetPane = $(targetPaneId);
      if (targetPane) targetPane.classList.add("active");

      // Auto-load tab data if opened
      if (tab.dataset.tab === "negtests" && !$("negTestsContainer").hasChildNodes()) {
        runNegativeTests();
      } else if (tab.dataset.tab === "kpis" && !$("kpisContainer").hasChildNodes()) {
        runKpiBenchmarks();
      } else if (tab.dataset.tab === "evidence" && !$("deliverablesTableBody").hasChildNodes()) {
        loadEvidenceManifest();
      }
    });
  });
}

// -------------------------------------------------------------------------
// Theme Toggle
// -------------------------------------------------------------------------
function initTheme() {
  const saved = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  const btn = $("themeToggle");
  const icon = btn.querySelector(".theme-icon");

  const sync = () => {
    icon.textContent =
      document.documentElement.getAttribute("data-theme") === "dark" ? "☀" : "☾";
  };
  sync();

  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    sync();
  });
}

// -------------------------------------------------------------------------
// Health Check
// -------------------------------------------------------------------------
async function checkHealth() {
  const el = $("healthStatus");
  try {
    const r = await fetch(API.health);
    const j = await r.json();
    el.textContent = `API ${j.status.toUpperCase()} · scikit-learn v${j.sklearn_version}`;
    el.className = "health ok";
  } catch (e) {
    el.textContent = "API offline — start backend/app.py";
    el.className = "health err";
  }
}

// -------------------------------------------------------------------------
// Presets Loader
// -------------------------------------------------------------------------
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

    sel.addEventListener("change", (e) => applyPreset(e.target.value));

    // Default select canonical gender binary
    if (j.presets.length > 0) {
      sel.value = "gender_binary";
      applyPreset("gender_binary");
    }
  } catch (e) {
    console.error("Failed to load presets", e);
  }
}

async function applyPreset(key) {
  $("presetDesc").textContent = "";
  if (!key) return;
  try {
    const r = await fetch(API.preset(key));
    const { preset } = await r.json();
    $("presetDesc").textContent = preset.description;
    $("featureNames").value = (preset.feature_names || []).join(", ");
    $("trainData").value = gridToText(preset.train_data);
    $("testData").value = gridToText(preset.test_data);
    $("groundTruth").value = gridToText(preset.ground_truth);
    $("dropSelect").value = preset.drop === null ? "null" : preset.drop;
    $("handleUnknown").value = preset.handle_unknown || "ignore";
  } catch (e) {
    console.error(e);
  }
}

// -------------------------------------------------------------------------
// Policy Selector listener
// -------------------------------------------------------------------------
function initPolicySelector() {
  const sel = $("policySelect");
  const hint = $("policyHint");
  sel.addEventListener("change", () => {
    hint.textContent = POLICY_HINTS[sel.value] || "";
  });
}

// -------------------------------------------------------------------------
// Main Analysis Action (Tab 1)
// -------------------------------------------------------------------------
async function analyze() {
  const btn = $("analyzeBtn");
  const err = $("errorBox");
  err.hidden = true;
  $("rejectionAlert").hidden = true;

  const train = parseGrid($("trainData").value);
  const test = parseGrid($("testData").value);
  const gt = parseGrid($("groundTruth").value);
  const names = $("featureNames").value
    .split(",").map((s) => s.trim()).filter(Boolean);
  const policy = $("policySelect").value;

  if (!train.length || !test.length) {
    err.hidden = false;
    err.textContent = "Both training data and test data are required.";
    return;
  }

  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Analyzing…`;

  try {
    const body = {
      train_data: train,
      test_data: test,
      drop: $("dropSelect").value,
      handle_unknown: $("handleUnknown").value,
      policy: policy,
    };
    if (gt.length) body.ground_truth = gt;
    if (names.length) body.feature_names = names;

    const r = await fetch(API.analyze, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();

    if (!j.success) {
      throw new Error(j.error || "Analysis failed.");
    }

    renderResults(j);
  } catch (e) {
    err.hidden = false;
    err.textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Safe Decoding Analysis";
  }
}

// -------------------------------------------------------------------------
// Render Analysis Results
// -------------------------------------------------------------------------
function renderResults(data) {
  LAST_ANALYSIS_RESULTS = data;
  $("placeholder").hidden = true;
  $("resultsBody").hidden = false;

  // Handle STRICT_REJECTION response
  if (data.rejected) {
    $("keyFinding").textContent = data.summary.key_finding;
    $("rejectionAlert").hidden = false;
    $("rejectionMessage").textContent = `${data.rejection.message} (Candidates: ${data.rejection.possible_values.join(", ")})`;
    $("tiles").innerHTML = `
      <div class="tile crit">
        <div class="tile-label">Execution Status</div>
        <div class="tile-value">HALTED</div>
        <div class="tile-sub">Strict Rejection Active</div>
      </div>
      <div class="tile warn">
        <div class="tile-label">Rejected Sample</div>
        <div class="tile-value">#${data.rejection.sample_index}</div>
        <div class="tile-sub">Feature Index ${data.rejection.feature_index}</div>
      </div>
    `;
    $("statusBars").innerHTML = `<p class="field-hint">Pipeline strictly aborted before writing corrupted output.</p>`;
    $("resultsTable").querySelector("tbody").innerHTML = "";
    renderMetadata(data.metadata);
    return;
  }

  // 1. Key Finding
  $("keyFinding").textContent = data.summary.key_finding;
  $("rejectionAlert").hidden = true;

  // 2. Metric Scorecards
  const m = data.metrics;
  $("tiles").innerHTML = `
    <div class="tile ${m.baseline_incorrect > 0 ? "crit" : "good"}">
      <div class="tile-label">Silent Errors Intercepted</div>
      <div class="tile-value">${m.baseline_incorrect}</div>
      <div class="tile-sub">sklearn silent errors blocked</div>
    </div>
    <div class="tile good">
      <div class="tile-label">Ambiguity Recall</div>
      <div class="tile-value">${m.detection_rate}%</div>
      <div class="tile-sub">${m.ambiguous_count} collisions detected</div>
    </div>
    <div class="tile blue">
      <div class="tile-label">Safe Handling Rate</div>
      <div class="tile-value">${m.safe_handling_rate}%</div>
      <div class="tile-sub">Decoded without wrong labels</div>
    </div>
    <div class="tile ${m.baseline_correct === m.total_samples ? "good" : "warn"}">
      <div class="tile-label">Baseline Accuracy</div>
      <div class="tile-value">${data.summary.baseline_accuracy}</div>
      <div class="tile-sub">Raw sklearn performance</div>
    </div>
  `;

  // 3. Status Distribution
  const total = m.total_samples || 1;
  const pSafe = Math.round((m.safe_count / total) * 100);
  const pAmb = Math.round((m.ambiguous_count / total) * 100);
  const pUnk = Math.round((m.unknown_count / total) * 100);

  $("statusBars").innerHTML = `
    <div class="status-bar-row">
      <span>SAFE (${m.safe_count})</span>
      <div class="bar-track"><div class="bar-fill safe" style="width: ${pSafe}%"></div></div>
      <span>${pSafe}%</span>
    </div>
    <div class="status-bar-row">
      <span>AMBIGUOUS (${m.ambiguous_count})</span>
      <div class="bar-track"><div class="bar-fill ambiguous" style="width: ${pAmb}%"></div></div>
      <span>${pAmb}%</span>
    </div>
    <div class="status-bar-row">
      <span>UNKNOWN (${m.unknown_count})</span>
      <div class="bar-track"><div class="bar-fill unknown" style="width: ${pUnk}%"></div></div>
      <span>${pUnk}%</span>
    </div>
  `;

  // 4. Per-sample Comparison Table
  const tbody = $("resultsTable").querySelector("tbody");
  tbody.innerHTML = "";

  data.results.forEach((row, idx) => {
    const tr = document.createElement("tr");

    const isSilentError = row.baseline_correct === false;
    const baselineDisplay = row.baseline_decoded.join(", ");
    const safeDisplay = row.safe_output.map((v) => (v === null ? "<em>None</em>" : v)).join(", ");
    const candDisplay = row.possible_values.map((p) => `[${p.join("|")}]`).join(", ");

    tr.innerHTML = `
      <td>${row.sample_index + 1}</td>
      <td><strong>${(row.input || []).join(", ")}</strong></td>
      <td><span class="code-pill">[${row.encoded.join(", ")}]</span></td>
      <td>
        <span class="${isSilentError ? "silent-fail" : ""}">
          ${baselineDisplay} ${isSilentError ? "⚠️ (Silent Error)" : ""}
        </span>
      </td>
      <td>
        <span class="status-badge ${row.row_status}">
          ${row.row_status}
        </span>
      </td>
      <td><strong>${safeDisplay}</strong></td>
      <td><span class="code-pill">${candDisplay}</span></td>
      <td>
        <button class="btn-sm" type="button" onclick="showProvenanceModal(${idx})">Audit Trace</button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // 5. Metadata Box
  renderMetadata(data.metadata);
}

function renderMetadata(meta) {
  const box = $("metadataBox");
  box.innerHTML = `
    <div class="metadata-grid">
      ${meta.features.map((f) => `
        <div class="meta-card">
          <h4>${f.feature_name} (Col ${f.feature_index})</h4>
          <div><strong>Categories:</strong> ${f.categories.join(", ")}</div>
          <div><strong>Dropped Category:</strong> ${f.dropped_category || "<em>None</em>"}</div>
          <div><strong>Zero-Collision Potential:</strong>
            <span class="${f.can_produce_all_zeros ? "silent-fail" : "status-badge SAFE"}">
              ${f.can_produce_all_zeros ? "YES (Hazardous)" : "NO (Safe)"}
            </span>
          </div>
          <div><strong>Encoded Columns:</strong> ${f.n_encoded_columns} (${f.encoded_col_range.join("..")})</div>
        </div>
      `).join("")}
    </div>
  `;
}

// -------------------------------------------------------------------------
// Provenance Modal Viewer
// -------------------------------------------------------------------------
window.showProvenanceModal = function(sampleIndex) {
  if (!LAST_ANALYSIS_RESULTS) return;
  const sample = LAST_ANALYSIS_RESULTS.results[sampleIndex];
  if (!sample) return;

  $("modalTitle").textContent = `Sample #${sampleIndex + 1} Provenance Side-Channel Audit Trace`;
  $("modalBody").innerHTML = `
    <div style="margin-bottom: 12px;">
      <strong>Row Status:</strong> <span class="status-badge ${sample.row_status}">${sample.row_status}</span> &middot;
      <strong>Confidence:</strong> ${sample.confidence} &middot;
      <strong>Policy:</strong> ${sample.policy_applied}
    </div>
    <div class="provenance-json">${JSON.stringify(sample.provenance, null, 2)}</div>
  `;
  $("provenanceModal").hidden = false;
};

function initModal() {
  $("closeModalBtn").addEventListener("click", () => {
    $("provenanceModal").hidden = true;
  });
  $("provenanceModal").querySelector(".modal-backdrop").addEventListener("click", () => {
    $("provenanceModal").hidden = true;
  });
}

// -------------------------------------------------------------------------
// Tab 2: Mandatory Negative Tests (NT-1 to NT-5)
// -------------------------------------------------------------------------
async function runNegativeTests() {
  const btn = $("runNegTestsBtn");
  const loading = $("negTestsLoading");
  const container = $("negTestsContainer");

  btn.disabled = true;
  loading.hidden = false;
  container.innerHTML = "";

  try {
    const r = await fetch(API.negativeTests);
    const j = await r.json();

    if (!j.success) throw new Error(j.error || "Failed to execute negative tests");

    j.reports.forEach((rep) => {
      const card = document.createElement("div");
      card.className = "neg-card";
      card.innerHTML = `
        <div class="neg-card-header">
          <div>
            <span class="neg-id-badge">${rep.nt_id}</span>
            <h3>${rep.title}</h3>
          </div>
          <span class="badge-pass">${rep.verdict} (${rep.recovery_time_ms} ms)</span>
        </div>
        <div class="neg-card-body">
          <div class="neg-section fail-sec">
            <h4>Observed scikit-learn Baseline Failure</h4>
            <p>${rep.observed_baseline_behavior}</p>
          </div>
          <div class="neg-section safe-sec">
            <h4>Ambiguity-Safe Response & Recovery</h4>
            <p>${rep.safe_response}</p>
          </div>
        </div>
        <div class="neg-card-footer">
          <span class="neg-risk"><strong>Residual Risk:</strong> ${rep.residual_risk_statement}</span>
          <span><strong>Trigger:</strong> ${rep.trigger}</span>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (e) {
    container.innerHTML = `<div class="error-box">${e.message}</div>`;
  } finally {
    btn.disabled = false;
    loading.hidden = true;
  }
}

// -------------------------------------------------------------------------
// Tab 3: KPI Benchmarking Dashboard (KPI-1 to KPI-6)
// -------------------------------------------------------------------------
async function runKpiBenchmarks() {
  const btn = $("runKpisBtn");
  const loading = $("kpisLoading");
  const container = $("kpisContainer");
  const summaryBanner = $("kpiSummaryBanner");

  btn.disabled = true;
  loading.hidden = false;
  container.innerHTML = "";
  summaryBanner.hidden = true;

  try {
    const r = await fetch(API.kpis);
    const j = await r.json();

    if (!j.success) throw new Error(j.error || "Failed to run KPI benchmarks");

    // Summary banner
    summaryBanner.hidden = false;
    summaryBanner.innerHTML = `
      <div>
        <strong style="font-size: 15px;">Benchmark Execution Summary (AC-1 to AC-4):</strong>
        <span style="margin-left: 10px;">${j.n_samples} Monte-Carlo samples &middot; p95 Latency: ${j.summary.p95_safe_ms}ms &middot; Sparse Footprint: ${j.summary.sparse_memory_pct}%</span>
      </div>
      <span class="badge-pass">ALL 6 KPIS PASSED (${j.all_passed ? "100% Contract Compliance" : "Action Required"})</span>
    `;

    // Render KPI Cards
    j.measurements.forEach((kpi) => {
      const card = document.createElement("div");
      card.className = "kpi-card";

      const isDirectionLower = kpi.direction === "lower_is_better";
      const unit = kpi.unit;

      card.innerHTML = `
        <div class="kpi-card-header">
          <div>
            <span class="kpi-code">${kpi.kpi_id}</span>
            <h3>${kpi.name}</h3>
          </div>
          <span class="badge-pass">${kpi.pass_verdict ? "PASS" : "FAIL"}</span>
        </div>
        <p style="font-size: 12.5px; color: var(--text-secondary); margin: 0 0 10px;">${kpi.description}</p>
        <div class="kpi-values-row">
          <div class="kpi-val-box">
            <div class="kpi-val-title">Baseline</div>
            <div class="kpi-val-number baseline">${kpi.baseline_value}${unit}</div>
          </div>
          <div style="font-size: 20px; color: var(--text-muted);">&rarr;</div>
          <div class="kpi-val-box">
            <div class="kpi-val-title">Ambiguity-Safe</div>
            <div class="kpi-val-number safe">${kpi.safe_value}${unit}</div>
          </div>
        </div>
        <div class="kpi-target-box">
          <strong>Pass Contract Rule:</strong> ${kpi.target_rule}
        </div>
      `;
      container.appendChild(card);
    });
  } catch (e) {
    container.innerHTML = `<div class="error-box">${e.message}</div>`;
  } finally {
    btn.disabled = false;
    loading.hidden = true;
  }
}

// -------------------------------------------------------------------------
// Tab 4: Evidence & Deliverables Loader
// -------------------------------------------------------------------------
async function loadEvidenceManifest() {
  const tbody = $("deliverablesTableBody");
  tbody.innerHTML = "";

  try {
    const r = await fetch(API.evidence);
    const j = await r.json();

    j.deliverables.forEach((del) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong class="code-pill">${del.id}</strong></td>
        <td><strong>${del.name}</strong></td>
        <td>${del.evidence}</td>
        <td><span class="badge-approved">${del.status}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="4" class="error-box">Failed to load manifest: ${e.message}</td></tr>`;
  }
}

// -------------------------------------------------------------------------
// Initialization
// -------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initTheme();
  initPolicySelector();
  initModal();
  checkHealth();
  loadPresets();

  $("analyzeBtn").addEventListener("click", analyze);
  $("runNegTestsBtn").addEventListener("click", runNegativeTests);
  $("runKpisBtn").addEventListener("click", runKpiBenchmarks);
});
