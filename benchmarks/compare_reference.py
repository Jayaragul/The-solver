#!/usr/bin/env python3
"""Head-to-head comparison against an independent reference LP solver.

This script is BENCHMARKING ORCHESTRATION ONLY, which is the sole role
prompt.md permits Python in this project. The reference solver (HiGHS, via
scipy.optimize.milp) is invoked here, in a separate process, purely to
produce comparison numbers. It is never linked into, wrapped by, called
from, or depended upon by the C++ solver -- prompt.md's Benchmark Strategy
explicitly asks for exactly this: benchmark against publicly available
reference implementations while the core solver stays independent.

Both solvers receive the IDENTICAL problem: the .lp.txt files were written
by benchmarks/export_lp.cpp straight from the parsed LpProblem the C++
solver itself solved, so a parser discrepancy cannot masquerade as a
solver discrepancy.

Usage:
    python3 benchmarks/compare_reference.py [export_dir]
"""

import csv
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def parse_floats(line):
    return np.array(
        [
            math.inf if t == "inf" else (-math.inf if t == "-inf" else float(t))
            for t in line.split()
        ],
        dtype=float,
    )


def load_lp(path):
    with open(path) as f:
        n_rows, n_cols, nnz = (int(t) for t in f.readline().split())
        row_ptr = np.array([int(t) for t in f.readline().split()], dtype=np.int64)
        col_idx = (
            np.array([int(t) for t in f.readline().split()], dtype=np.int64)
            if nnz
            else np.zeros(0, dtype=np.int64)
        )
        values = parse_floats(f.readline()) if nnz else np.zeros(0)
        obj = parse_floats(f.readline())
        col_lower = parse_floats(f.readline())
        col_upper = parse_floats(f.readline())
        row_lb = parse_floats(f.readline())
        row_ub = parse_floats(f.readline())

    A = csr_matrix((values, col_idx, row_ptr), shape=(n_rows, n_cols))
    return A, obj, col_lower, col_upper, row_lb, row_ub


def solve_reference(A, obj, col_lower, col_upper, row_lb, row_ub):
    """Solve with HiGHS via scipy, using two-sided row bounds directly."""
    from scipy.optimize import LinearConstraint

    con = LinearConstraint(A, row_lb, row_ub)
    bounds = list(zip(col_lower, col_upper))
    t0 = time.perf_counter()
    res = linprog(
        c=obj,
        constraints=[con],
        bounds=bounds,
        method="highs",
    )
    secs = time.perf_counter() - t0
    return res, secs


def main():
    export_dir = sys.argv[1] if len(sys.argv) > 1 else "build/lp_export"
    ours_csv = os.path.join(export_dir, "ours.csv")
    if not os.path.exists(ours_csv):
        print(f"missing {ours_csv} -- run build/benchmarks/export_lp first")
        return 2

    with open(ours_csv) as f:
        ours = list(csv.DictReader(f))

    print(
        f"{'instance':>14} {'rows':>6} {'cols':>6} | "
        f"{'ours obj':>20} {'ref obj':>20} {'rel.diff':>10} | "
        f"{'ours s':>8} {'ref s':>8} {'ratio':>8}  verdict"
    )
    print("-" * 132)

    agree = disagree = skipped = 0
    ours_total = ref_total = 0.0
    ours_wins = ref_wins = 0

    for row in ours:
        name = row["instance"]
        lp_path = os.path.join(export_dir, f"{name}.lp.txt")
        if not os.path.exists(lp_path):
            skipped += 1
            continue

        A, obj, cl, cu, rl, ru = load_lp(lp_path)
        try:
            res, ref_secs = solve_reference(A, obj, cl, cu, rl, ru)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:>14} reference solver error: {exc}")
            skipped += 1
            continue

        if not res.success:
            print(f"{name:>14} reference did not solve: {res.message[:60]}")
            skipped += 1
            continue

        ours_obj = float(row["objective"])
        ours_secs = float(row["seconds"])
        ref_obj = float(res.fun)
        rel = abs(ours_obj - ref_obj) / (1.0 + abs(ref_obj))
        ok = row["status"] == "OPTIMAL" and rel < 1e-6

        if ok:
            agree += 1
            ours_total += ours_secs
            ref_total += ref_secs
            if ours_secs < ref_secs:
                ours_wins += 1
            else:
                ref_wins += 1
        else:
            disagree += 1

        ratio = ref_secs / ours_secs if ours_secs > 1e-9 else float("nan")
        print(
            f"{name:>14} {row['rows']:>6} {row['cols']:>6} | "
            f"{ours_obj:>20.8f} {ref_obj:>20.8f} {rel:>10.2e} | "
            f"{ours_secs:>8.3f} {ref_secs:>8.3f} {ratio:>7.2f}x  "
            f"{'AGREE' if ok else 'DISAGREE'}"
        )

    print("=" * 132)
    print(f"Objective agreement: {agree} agree, {disagree} disagree, {skipped} skipped")
    if agree:
        print(
            f"Total wall-clock on agreeing instances: "
            f"ours {ours_total:.3f}s vs reference(HiGHS) {ref_total:.3f}s"
            f"  -> reference is {ours_total / ref_total:.2f}x "
            f"{'faster' if ref_total < ours_total else 'slower'}"
        )
        print(
            f"Per-instance wins: ours faster on {ours_wins}, "
            f"reference faster on {ref_wins}"
        )
    return 0 if disagree == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
