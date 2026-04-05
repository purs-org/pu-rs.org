# Cost Effectiveness

The most important metric for deployment decisions: **how much real performance do you get per dollar and per watt?**

<div id="cost-chart-container">
  <canvas id="cost-scatter" width="800" height="400"></canvas>
</div>

<div id="cost-filters" style="margin: 1em 0;">
  <select id="cost-kernel"><option value="">All Kernels</option></select>
</div>

<table id="cost-table" class="xpu-table">
  <thead>
    <tr>
      <th>#</th>
      <th>Device</th>
      <th>Kernel</th>
      <th>Latency (us)</th>
      <th>MSRP ($)</th>
      <th>TDP (W)</th>
      <th>GOPS/$</th>
      <th>GOPS/W</th>
    </tr>
  </thead>
  <tbody id="cost-body"></tbody>
</table>
