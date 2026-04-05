// charts.js — Chart.js helpers for kernel detail pages and cost-effectiveness
// Loads leaderboard.json and renders bar/scatter charts.
// Requires Chart.js to be loaded via CDN in book.toml additional-js.

(function () {
  "use strict";

  // Placeholder: charts will be rendered when Chart.js is available
  // and leaderboard.json is loaded. For now, kernel detail pages
  // show the leaderboard link as the primary data view.

  document.addEventListener("DOMContentLoaded", () => {
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
