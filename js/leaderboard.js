// leaderboard.js — client-side filtering and sorting for pu-rs.org
// Loads leaderboard.json (generated at build time from SQLite), populates
// drop-down filters, renders a sortable table. Zero server dependencies.

(function () {
  "use strict";

  const DATA_URL = "data/leaderboard.json";
  let allRows = [];
  let sortCol = "throughput_gops";
  let sortAsc = false;

  // ── Bootstrap ───────────────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", async () => {
    const body = document.getElementById("leaderboard-body");
    if (!body) return; // not on the leaderboard page

    try {
      const resp = await fetch(DATA_URL);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      allRows = await resp.json();
    } catch (e) {
      body.innerHTML = `<tr><td colspan="12">Loading benchmark data... (${e.message})</td></tr>`;
      return;
    }

    populateFilters();
    bindEvents();
    render();
  });

  // ── Populate filter dropdowns from data ─────────────────────────────────
  function populateFilters() {
    populate("filter-kernel", unique("kernel_name"));
    populate("filter-dtype", unique("dtype"));
    populate("filter-vendor", unique("vendor"));
    populate("filter-xpu-type", unique("xpu_type"));
    populate("filter-batch", unique("batch_size").map(String));
  }

  function unique(key) {
    return [...new Set(allRows.map((r) => r[key]))].filter(Boolean).sort();
  }

  function populate(id, values) {
    const sel = document.getElementById(id);
    if (!sel) return;
    const first = sel.options[0].text;
    sel.innerHTML = `<option value="">${first}</option>`;
    values.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      sel.appendChild(opt);
    });
  }

  // ── Event binding ───────────────────────────────────────────────────────
  function bindEvents() {
    ["filter-kernel", "filter-dtype", "filter-vendor", "filter-xpu-type", "filter-batch"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("change", render);
    });

    const resetBtn = document.getElementById("btn-reset");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        document.querySelectorAll("#filters select").forEach((s) => (s.value = ""));
        render();
      });
    }

    document.querySelectorAll("#leaderboard-table th[data-sort]").forEach((th) => {
      th.style.cursor = "pointer";
      th.addEventListener("click", () => {
        const col = th.dataset.sort;
        if (col === sortCol) {
          sortAsc = !sortAsc;
        } else {
          sortCol = col;
          // lower is better for latency; higher is better for throughput metrics
          sortAsc = col === "latency_us";
        }
        render();
      });
    });
  }

  // ── Render ──────────────────────────────────────────────────────────────
  function render() {
    const filters = {
      kernel_name: val("filter-kernel"),
      dtype: val("filter-dtype"),
      vendor: val("filter-vendor"),
      xpu_type: val("filter-xpu-type"),
      batch_size: val("filter-batch"),
    };

    let rows = allRows.filter((r) => {
      for (const [key, fv] of Object.entries(filters)) {
        if (fv && String(r[key]) !== fv) return false;
      }
      return true;
    });

    rows.sort((a, b) => {
      let va = a[sortCol],
        vb = b[sortCol];
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "string") {
        return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return sortAsc ? va - vb : vb - va;
    });

    const body = document.getElementById("leaderboard-body");
    const noData = document.getElementById("no-data");
    const summary = document.getElementById("summary-bar");

    if (rows.length === 0) {
      body.innerHTML = "";
      if (noData) noData.style.display = "block";
      if (summary) summary.textContent = "";
      return;
    }
    if (noData) noData.style.display = "none";

    // Find best (highest throughput) per kernel+shape for highlighting
    const bestByKernelShape = {};
    rows.forEach((r) => {
      const k = r.kernel_name + "|" + r.input_shape;
      if (r.throughput_gops != null && (!bestByKernelShape[k] || r.throughput_gops > bestByKernelShape[k])) {
        bestByKernelShape[k] = r.throughput_gops;
      }
    });

    body.innerHTML = rows
      .map((r, i) => {
        const k = r.kernel_name + "|" + r.input_shape;
        const isBest = r.throughput_gops != null && r.throughput_gops === bestByKernelShape[k];
        const cls = isBest ? ' class="best-row"' : "";
        return `<tr${cls}>
        <td>${i + 1}</td>
        <td>${esc(r.vendor)}</td>
        <td>${esc(r.model)}</td>
        <td>${esc(r.xpu_type)}</td>
        <td>${esc(r.kernel_name)}</td>
        <td>${esc(r.dtype)}</td>
        <td>${esc(r.input_shape)}</td>
        <td>${fmt(r.latency_us)}</td>
        <td>${fmt(r.throughput_gops)}</td>
        <td>${fmt(r.gops_per_dollar)}</td>
        <td>${fmt(r.gops_per_watt)}</td>
        <td>${r.verified ? "Y" : ""}</td>
      </tr>`;
      })
      .join("");

    if (summary) {
      const devices = new Set(rows.map((r) => r.model)).size;
      const kernels = new Set(rows.map((r) => r.kernel_name)).size;
      summary.textContent = `${rows.length} results | ${devices} devices | ${kernels} kernels | sorted by ${sortCol} ${sortAsc ? "asc" : "desc"}`;
    }
  }

  function val(id) {
    const el = document.getElementById(id);
    return el ? el.value : "";
  }
  function esc(s) {
    return s == null ? "" : String(s).replace(/</g, "&lt;");
  }
  function fmt(v) {
    if (v == null) return "-";
    if (typeof v === "number") return v < 1 ? v.toFixed(4) : v < 100 ? v.toFixed(1) : Math.round(v).toLocaleString();
    return String(v);
  }
})();
