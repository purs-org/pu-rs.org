# Leaderboard

<div id="leaderboard-app">
<div id="filters" style="margin-bottom: 1em; display: flex; gap: 0.5em; flex-wrap: wrap;">
  <select id="filter-kernel"><option value="">All Kernels</option></select>
  <select id="filter-dtype"><option value="">All Dtypes</option></select>
  <select id="filter-vendor"><option value="">All Vendors</option></select>
  <select id="filter-xpu-type"><option value="">All xPU Types</option></select>
  <select id="filter-batch"><option value="">All Batch Sizes</option></select>
  <button id="btn-reset" style="cursor:pointer;">Reset Filters</button>
</div>

<div id="summary-bar" style="margin-bottom: 1em; font-size: 0.9em; color: #666;"></div>

<table id="leaderboard-table" class="xpu-table">
  <thead>
    <tr>
      <th data-sort="rank">#</th>
      <th data-sort="vendor">Vendor</th>
      <th data-sort="model">Device</th>
      <th data-sort="xpu_type">Type</th>
      <th data-sort="kernel_name">Kernel</th>
      <th data-sort="dtype">Dtype</th>
      <th data-sort="input_shape">Shape</th>
      <th data-sort="latency_us">Latency (us)</th>
      <th data-sort="throughput_gops">GOPS</th>
      <th data-sort="gops_per_dollar">GOPS/$</th>
      <th data-sort="gops_per_watt">GOPS/W</th>
      <th data-sort="verified">Verified</th>
    </tr>
  </thead>
  <tbody id="leaderboard-body"></tbody>
</table>

<p id="no-data" style="display:none; color:#999;">No results match the selected filters.</p>
</div>
