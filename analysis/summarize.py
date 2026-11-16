"""Print manuscript table values from the archived per-seed rows."""

import csv
from pathlib import Path
from statistics import mean, stdev

DATA = Path(__file__).resolve().parents[1] / "data" / "results.csv"
FIXED = "score>=0.25"
MATCHED = "matched FP against B in same seed"
VALIDATION = (
    ("official", "AP"), ("official", "AP75"), ("official", "APvt"),
    (FIXED, "FP_per_image"), (FIXED, "dup_per_TP"),
    (MATCHED, "E1vt_R50"), (MATCHED, "E1vt_R75"),
)
TEST = (
    ("official", "AP"), ("official", "APvt"), (FIXED, "FP_per_image"),
    (FIXED, "E1vt_R50"), (FIXED, "E1vt_R75"),
    (FIXED, "E1_outside_R50"), (FIXED, "E1_outside_R75"),
)


def format_stat(values, metric):
    if metric == "dup_per_TP":
        return f"{mean(values):.3f}"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{mean(values):.2f} +/- {stdev(values):.2f}"


def print_table(rows, study, columns, arm_seeds):
    print("model | " + " | ".join(metric for _, metric in columns))
    for arm, seeds in arm_seeds.items():
        entries = []
        for setting, metric in columns:
            values = [rows[study, arm, setting, metric, str(seed)] for seed in seeds]
            entries.append(format_stat(values, metric))
        print(f"{arm} | " + " | ".join(entries))


def main():
    with DATA.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
        rows = {
            (r["study"], r["arm"], r["setting"], r["metric"], r["seed"]): float(r["value"])
            for r in records
            if r["study"] in {"baseline50", "test50"}
        }

    paired = dict.fromkeys(("B", "STAL", "STAL_SD", "P2"), (0, 1, 2))
    print("Table 1: Validation, mix of official, fixed-score, and matched-FP metrics")
    print_table(rows, "baseline50", VALIDATION, {**paired, "TWIN": (0, 1)})
    print("\nTable 2: Held-out test, fixed-score recall and false positives")
    print_table(rows, "test50", TEST, {**paired, "B_trainval": (7,)})

    readout = {
        (r["seed"], r["setting"], r["metric"]): float(r["value"])
        for r in records
        if r["study"] == "readout"
    }
    counts = ("R", "G", "P", "K")
    settings = ("native decode score>=0.25", "restricted decode score>=0.25")
    totals = [0] * len(counts)
    print("\nTable 3: Same-weight readout intervention on validation")
    print("seed | R | G | P | K | E1vt R75 native -> clipped | FP/img native -> clipped")
    for seed in ("0", "1", "2"):
        values = [int(readout[seed, "E1vt score>=0.25 IoU>=0.75", metric]) for metric in counts]
        totals = [total + value for total, value in zip(totals, values)]
        pairs = [
            " -> ".join(f"{readout[seed, setting, metric]:.2f}" for setting in settings)
            for metric in ("E1vt_R75", "FP_per_image")
        ]
        print(" | ".join([seed, *map(str, values), *pairs]))
    print(" | ".join(["Total", *map(str, totals)]))
    r, g, _, _ = totals
    print(f"pooled G/R = {g}/{r} = {g / r:.2f}")


if __name__ == "__main__":
    main()
