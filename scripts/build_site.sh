#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="$ROOT/db/xpu_bench.db"

echo "=== Step 1: Build xpu_bench.db ==="
rm -f "$DB"
sqlite3 "$DB" < "$ROOT/db/schema.sql"
sqlite3 "$DB" < "$ROOT/db/seed_devices.sql"
sqlite3 "$DB" < "$ROOT/db/seed_kernels.sql"

echo "=== Step 2: Ingest submission CSVs ==="
for csv in "$ROOT"/submissions/*.csv; do
    [ -f "$csv" ] || continue
    echo "  Ingesting $csv"
    python3 "$ROOT/scripts/ingest_bench_csv.py" "$DB" "$csv"
done

echo "=== Step 3: Export JSON for static site ==="
mkdir -p "$ROOT/src/data"
python3 "$ROOT/scripts/export_leaderboard_json.py" "$DB" "$ROOT/src/data/"

echo "=== Step 4: Build mdbook ==="
cd "$ROOT"
mdbook build

echo "=== Step 5: Copy CNAME for GitHub Pages ==="
cp "$ROOT/src/CNAME" "$ROOT/book/CNAME" 2>/dev/null || true

echo "=== Done: book/ ready for deployment ==="
echo "  Open: file://$ROOT/book/index.html"
