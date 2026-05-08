#!/usr/bin/env python3
"""Calibrate alpha/beta from the leaderboard CSVs and save results.

Produces three artefacts in results/:
  raw_coefficients.csv          — one row per leaderboard observation
  calibrated_coefficients.csv   — aggregated by (hw, backend, precision)
  calibrated_fine.csv           — aggregated by (hw, backend, precision, attention)
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.calibrate import calibrate, aggregate, filter_outliers

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HW_FILES = {
    "1xA10":      "leaderboard-1xA10.csv",
    "1xA100":     "leaderboard-1xA100.csv",
    "1xT4":       "leaderboard-1xT4.csv",
    "32vCPU-C7i": "leaderboard-32vCPU-C7i.csv",
    "RTX-3090":      "leaderboard-RTX-3090-llamacpp.csv",
    "2xRTX-3090":    "leaderboard-2xRTX-3090-llamacpp-32b.csv",
}

HW_CAPACITY_MB = {"1xA10": 24576, "1xA100": 81920, "1xT4": 16384, "32vCPU-C7i": 65536, "RTX-3090": 24576, "2xRTX-3090": 49152}


def main():
    per_hw = {}
    for hw, fn in HW_FILES.items():
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            df = pd.read_csv(path)
            cap = HW_CAPACITY_MB[hw]
            before = len(df)
            df = df[df["Memory (MB)"] <= 0.95 * cap]
            print(f"loaded {hw}: {before} rows, kept {len(df)} that fit in {cap/1024:.0f} GB")
            per_hw[hw] = df

    raw = calibrate(per_hw)
    raw.to_csv(os.path.join(RESULTS_DIR, "raw_coefficients.csv"), index=False)
    print(f"\nraw_coefficients.csv: {len(raw)} rows")

    clean = filter_outliers(raw)
    print(f"after outlier filter (0.005<alpha,beta<1.0): {len(clean)} rows "
          f"({len(raw)-len(clean)} dropped)")

    coarse = aggregate(clean, by=["hw", "backend", "precision_label"])
    coarse.to_csv(os.path.join(RESULTS_DIR, "calibrated_coefficients.csv"), index=False)
    print(f"\ncalibrated_coefficients.csv (coarse):\n{coarse.to_string(index=False)}")

    fine = aggregate(clean, by=["hw", "backend", "precision_label", "attention"])
    fine.to_csv(os.path.join(RESULTS_DIR, "calibrated_fine.csv"), index=False)
    print(f"\ncalibrated_fine.csv (with attention split): {len(fine)} buckets")


if __name__ == "__main__":
    main()
