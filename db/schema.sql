-- pu-rs.org xPU Kernel Benchmark Schema
-- Analogous to SPECfp2000 but for AI accelerators.

CREATE TABLE IF NOT EXISTS xpu_devices (
    device_id       TEXT PRIMARY KEY,
    vendor          TEXT NOT NULL,
    model           TEXT NOT NULL,
    architecture    TEXT NOT NULL,
    xpu_type        TEXT NOT NULL CHECK(xpu_type IN ('GPU','TPU','NPU','FPGA','CPU')),
    tdp_watts       REAL,
    memory_gb       REAL,
    memory_bw_gbps  REAL,
    compute_units   INTEGER,
    launch_date     TEXT,
    msrp_usd        REAL,
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kernels (
    kernel_id       TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    category        TEXT NOT NULL,
    description     TEXT,
    reference_impl  TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bench_configs (
    config_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    kernel_id       TEXT NOT NULL REFERENCES kernels(kernel_id),
    dtype           TEXT NOT NULL,
    input_shape     TEXT NOT NULL,
    batch_size      INTEGER DEFAULT 1,
    extra_params    TEXT,
    UNIQUE(kernel_id, dtype, input_shape, batch_size, extra_params)
);

CREATE TABLE IF NOT EXISTS bench_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id       TEXT NOT NULL REFERENCES xpu_devices(device_id),
    config_id       INTEGER NOT NULL REFERENCES bench_configs(config_id),
    impl_lang       TEXT NOT NULL DEFAULT 'rust',
    latency_us      REAL NOT NULL,
    latency_min_us  REAL,
    latency_max_us  REAL,
    latency_p99_us  REAL,
    throughput_gops REAL,
    num_runs        INTEGER NOT NULL DEFAULT 20,
    driver_version  TEXT,
    toolchain       TEXT,
    git_sha         TEXT,
    run_date        TEXT DEFAULT (datetime('now')),
    submitter       TEXT,
    verified        INTEGER DEFAULT 0,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS price_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    category        TEXT NOT NULL CHECK(category IN ('stock','device','commodity','fx')),
    price_usd       REAL NOT NULL,
    recorded_at     TEXT NOT NULL,
    source          TEXT
);

CREATE TABLE IF NOT EXISTS device_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id       TEXT NOT NULL REFERENCES xpu_devices(device_id),
    price_usd       REAL NOT NULL,
    source          TEXT NOT NULL,
    recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_device    ON bench_results(device_id);
CREATE INDEX IF NOT EXISTS idx_results_config    ON bench_results(config_id);
CREATE INDEX IF NOT EXISTS idx_results_combo     ON bench_results(device_id, config_id);
CREATE INDEX IF NOT EXISTS idx_configs_kernel    ON bench_configs(kernel_id);
CREATE INDEX IF NOT EXISTS idx_prices_symbol     ON price_history(symbol, recorded_at);
CREATE INDEX IF NOT EXISTS idx_device_prices_dev ON device_prices(device_id, recorded_at);

CREATE VIEW IF NOT EXISTS leaderboard AS
SELECT
    d.device_id,
    d.vendor,
    d.model,
    d.xpu_type,
    d.tdp_watts,
    d.msrp_usd,
    d.memory_gb,
    d.memory_bw_gbps,
    k.kernel_id,
    k.display_name AS kernel_name,
    k.category     AS kernel_category,
    c.dtype,
    c.input_shape,
    c.batch_size,
    r.impl_lang,
    r.latency_us,
    r.throughput_gops,
    r.num_runs,
    r.driver_version,
    r.toolchain,
    r.run_date,
    r.verified,
    CASE WHEN d.msrp_usd > 0 THEN r.throughput_gops / d.msrp_usd END AS gops_per_dollar,
    CASE WHEN d.tdp_watts > 0 THEN r.throughput_gops / d.tdp_watts END AS gops_per_watt
FROM bench_results r
JOIN xpu_devices d   ON d.device_id = r.device_id
JOIN bench_configs c ON c.config_id = r.config_id
JOIN kernels k       ON k.kernel_id = c.kernel_id
ORDER BY r.throughput_gops DESC;
