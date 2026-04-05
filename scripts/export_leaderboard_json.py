#!/usr/bin/env python3
"""Export SQLite leaderboard view to JSON files for the static site."""
import json
import sqlite3
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <db_path> <output_dir>")
        sys.exit(1)

    db_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Leaderboard (full denormalized table)
    rows = conn.execute("SELECT * FROM leaderboard").fetchall()
    leaderboard = [dict(r) for r in rows]
    write_json(out_dir / "leaderboard.json", leaderboard)
    print(f"  leaderboard.json: {len(leaderboard)} rows")

    # Devices
    rows = conn.execute("SELECT * FROM xpu_devices ORDER BY vendor, model").fetchall()
    write_json(out_dir / "devices.json", [dict(r) for r in rows])
    print(f"  devices.json: {len(rows)} devices")

    # Kernels
    rows = conn.execute("SELECT * FROM kernels ORDER BY category, kernel_id").fetchall()
    write_json(out_dir / "kernels.json", [dict(r) for r in rows])
    print(f"  kernels.json: {len(rows)} kernels")

    # Prices (latest per symbol)
    rows = conn.execute("""
        SELECT symbol, category, price_usd, recorded_at, source
        FROM price_history
        WHERE id IN (SELECT MAX(id) FROM price_history GROUP BY symbol)
        ORDER BY symbol
    """).fetchall()
    write_json(out_dir / "prices.json", [dict(r) for r in rows])
    print(f"  prices.json: {len(rows)} price entries")

    conn.close()


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    main()
