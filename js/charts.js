// charts.js — Chart.js helpers for kernel detail pages and cost-effectiveness
// Loads leaderboard.json and renders bar/scatter charts.
// Requires Chart.js to be loaded via CDN in book.toml additional-js.

(function () {
  "use strict";

  // Placeholder: charts will be rendered when Chart.js is available
  // and leaderboard.json is loaded. For now, kernel detail pages
  // show the leaderboard link as the primary data view.

  document.addEventListener("DOMContentLoaded", () => {
    // ── Per-kernel results table on kernel detail pages ───────────────────
    const kernelDiv = document.getElementById("kernel-results");
    if (kernelDiv) {
      // Kernel pages live at /kernels/X.html so data is at ../data/...
      fetch("../data/leaderboard.json")
        .then((r) => r.json())
        .then((rows) => {
          const kid = kernelDiv.dataset.kernel;
          const matches = rows.filter((r) => r.kernel_id === kid);
          if (matches.length === 0) {
            kernelDiv.innerHTML = "<p><em>No benchmark data submitted yet.</em></p>";
            return;
          }
          matches.sort((a, b) => a.latency_us - b.latency_us);
          const bestPerShape = {};
          matches.forEach((r) => {
            const key = r.input_shape + "|" + r.dtype;
            if (!bestPerShape[key] || r.latency_us < bestPerShape[key]) {
              bestPerShape[key] = r.latency_us;
            }
          });
          let html = '<table class="xpu-table"><thead><tr>'
            + '<th>#</th><th>Vendor</th><th>Device</th><th>Type</th>'
            + '<th>Shape</th><th>Dtype</th><th>Impl</th><th>Latency (us)</th>'
            + '</tr></thead><tbody>';
          matches.forEach((r, i) => {
            const key = r.input_shape + "|" + r.dtype;
            const isBest = r.latency_us === bestPerShape[key];
            const cls = isBest ? ' class="best-row"' : "";
            html += `<tr${cls}><td>${i + 1}</td><td>${r.vendor}</td>`
              + `<td>${r.model}</td><td>${r.xpu_type}</td>`
              + `<td>${r.input_shape}</td><td>${r.dtype}</td>`
              + `<td>${r.impl_lang}</td><td>${r.latency_us.toFixed(2)}</td></tr>`;
          });
          html += '</tbody></table>';
          html += `<p style="font-size:0.85em;color:#666;">${matches.length} results across ${new Set(matches.map(r=>r.model)).size} devices. Best per (shape, dtype) highlighted.</p>`;
          kernelDiv.innerHTML = html;
        })
        .catch((e) => {
          kernelDiv.innerHTML = `<p><em>Failed to load data: ${e.message}</em></p>`;
        });
    }

    // Cost-effectiveness scatter plot (cost.md)
    const costBody = document.getElementById("cost-body");
    if (!costBody) return;

    fetch("data/leaderboard.json")
      .then((r) => r.json())
      .then((rows) => {
        // Filter to rows with price data
        const withPrice = rows.filter((r) => r.msrp_usd > 0 && r.latency_us > 0);
        withPrice.sort((a, b) => (b.gops_per_dollar || 0) - (a.gops_per_dollar || 0));

        costBody.innerHTML = withPrice
          .map((r, i) => `<tr>
            <td>${i + 1}</td>
            <td>${r.model}</td>
            <td>${r.kernel_name}</td>
            <td>${r.latency_us.toFixed(1)}</td>
            <td>${r.msrp_usd.toLocaleString()}</td>
            <td>${r.tdp_watts || "-"}</td>
            <td>${r.gops_per_dollar ? r.gops_per_dollar.toFixed(4) : "-"}</td>
            <td>${r.gops_per_watt ? r.gops_per_watt.toFixed(4) : "-"}</td>
          </tr>`)
          .join("");
      })
      .catch(() => {});
  });
})();
