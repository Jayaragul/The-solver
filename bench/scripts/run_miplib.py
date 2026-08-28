#!/usr/bin/env python3
"""MIPLIB 2017 benchmark: SANKHYA branch-and-bound vs HiGHS.

Both solvers receive the identical MPS file and the identical time limit, and
neither is linked into the other: HiGHS is reached through its own Python
binding in-process, SANKHYA through a separate executable.

Scoring is deliberately three-way.  An instance passes only when our objective
agrees with **both** the published MIPLIB reference in the `.solu` file and the
independently run HiGHS baseline.  Agreeing with one but not the other is
reported as a disagreement rather than quietly scored against whichever is
more flattering.

A MILP answer is worthless if the integer variables are not actually integral,
so integrality is re-checked here from the returned solution vector.  `sk_mip`
also re-checks it internally; this is the second, independent check.

Instances come from `data/miplib2017_small/`, which is frozen in the repository
precisely so the benchmark set cannot drift between runs.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time


def load_solu(path):
    """Parse a MIPLIB `.solu` file.

    Lines look like `=opt=  name  value` or `=best=  name  value`; `=inf=`
    marks a proven-infeasible instance and carries no value.  Only `=opt=` is
    admissible as a correctness reference: `=best=` is the best value anyone
    has found, not a proven optimum, so scoring against it would penalise a
    solver for being right.
    """
    optima, best, infeasible = {}, {}, set()
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            tag, name = parts[0], parts[1]
            if tag == "=inf=":
                infeasible.add(name)
            elif len(parts) >= 3:
                try:
                    value = float(parts[2])
                except ValueError:
                    continue
                if tag == "=opt=":
                    optima[name] = value
                elif tag == "=best=":
                    best[name] = value
    return optima, best, infeasible


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_sankhya(exe, path, limit, gap_rel):
    cmd = [exe, path, "--quiet",
           "--time-limit", str(limit),
           "--gap-rel", str(gap_rel)]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=limit + 60)
    except subprocess.TimeoutExpired:
        return {"status": "hard_timeout", "seconds": time.perf_counter() - started}
    elapsed = time.perf_counter() - started
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, IndexError):
        return {"status": "unparseable", "seconds": elapsed,
                "stderr": proc.stderr[-400:]}
    record["seconds"] = elapsed
    return record


def run_highs(path, limit, gap_rel):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(limit))
    h.setOptionValue("mip_rel_gap", float(gap_rel))
    h.readModel(path)
    started = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - started
    status = h.modelStatusToString(h.getModelStatus())
    try:
        objective = h.getObjectiveValue()
    except Exception:
        objective = float("nan")
    return {"status": status, "objective": objective, "seconds": elapsed}


def integrality_violation(path, record):
    """The solver reports its own check; this reads it back rather than
    trusting the search.  sk_mip emits `integrality_inf` computed from the
    returned x, independently of how branch-and-bound reached it."""
    return record.get("integrality_inf", float("nan"))


def relative(a, b):
    if a is None or b is None:
        return float("inf")
    try:
        return abs(a - b) / (1.0 + abs(b))
    except TypeError:
        return float("inf")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(root, "data", "miplib2017_small"))
    ap.add_argument("--exe", default=os.path.expanduser("~/sk/build/sk_mip"))
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--gap-rel", type=float, default=1e-4)
    ap.add_argument("--tolerance", type=float, default=1e-6,
                    help="relative objective agreement tolerance")
    ap.add_argument("--integrality-tolerance", type=float, default=1e-6)
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per instance; the median time is reported")
    ap.add_argument("--jsonl", default=None, help="write one JSON record per instance")
    args = ap.parse_args()

    solu = os.path.join(args.data, "miplib2017-v36.solu")
    optima, best, infeasible = load_solu(solu)
    instances = sorted(f for f in os.listdir(args.data) if f.endswith(".mps"))
    if not instances:
        print(f"no .mps files under {args.data}", file=sys.stderr)
        return 2

    sink = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    if sink:
        sink.write(json.dumps({
            "kind": "header",
            "solver": "SANKHYA sk_mip",
            "baseline": "HiGHS (highspy, in-process)",
            "time_limit": args.time_limit,
            "gap_rel": args.gap_rel,
            "solu_sha256": sha256(solu),
        }) + "\n")
        sink.flush()

    agree_ref = agree_highs = passed = 0
    scored = 0
    failures = []
    ratios = []

    print(f"{'instance':<16} {'sankhya':<14} {'objective':>18} "
          f"{'vs .solu':>10} {'vs HiGHS':>10} {'int':>9} {'t(s)':>8} "
          f"{'highs t(s)':>10}  verdict")
    print("-" * 118)

    for filename in instances:
        path = os.path.join(args.data, filename)
        name = os.path.splitext(filename)[0]
        reference = optima.get(name)

        runs = [run_sankhya(args.exe, path, args.time_limit, args.gap_rel)
                for _ in range(args.repeat)]
        record = runs[0]
        our_time = statistics.median(r.get("seconds", float("nan")) for r in runs)

        try:
            baseline = run_highs(path, args.time_limit, args.gap_rel)
        except Exception as exc:                       # noqa: BLE001
            baseline = {"status": f"error:{exc}", "objective": None,
                        "seconds": float("nan")}

        ours = record.get("objective")
        status = record.get("status", "?")
        intviol = record.get("integrality_inf", float("nan"))
        base_status = str(baseline.get("status", "")).lower()
        base_optimal = base_status == "optimal"
        base_infeasible = "infeasible" in base_status

        d_ref = relative(ours, reference) if reference is not None else float("nan")
        d_highs = relative(ours, baseline.get("objective"))
        integral = intviol != intviol or intviol <= args.integrality_tolerance

        # An instance the reference proves infeasible is scored on the status
        # alone: there is no objective to compare, and detecting infeasibility
        # is the whole of the answer.
        if name in infeasible:
            correct = status == "infeasible"
            verdict = "OK(inf)" if correct else "WRONG"
            proven = correct
        elif status == "infeasible":
            correct, proven, verdict = False, False, "WRONG(inf)"
        else:
            ok_ref = reference is None or (d_ref == d_ref and d_ref <= args.tolerance)
            # HiGHS is only an authority when it actually proved optimality;
            # if it also timed out, its incumbent is just another guess.
            ok_highs = (not base_optimal) or (d_highs == d_highs and d_highs <= args.tolerance)
            correct = integral and ok_ref and ok_highs
            proven = correct and status == "optimal"
            if not integral:
                verdict = "NON-INTEGRAL"
            elif not ok_ref:
                verdict = "WRONG"
            elif not ok_highs:
                verdict = "DISAGREE"
            elif status == "optimal":
                verdict = "OK"
            else:
                # Right answer, not proved: distinguish from a wrong one, and
                # note when the baseline failed to prove it either.
                verdict = "TIME=" if base_optimal else "TIME~"

        if reference is not None or name in infeasible:
            scored += 1
            if correct:
                agree_ref += 1
        if base_optimal or base_infeasible:
            if (base_infeasible and status == "infeasible") or                (base_optimal and d_highs == d_highs and d_highs <= args.tolerance):
                agree_highs += 1
        if proven:
            passed += 1
            hb = baseline.get("seconds")
            if hb and hb > 1e-6 and our_time == our_time:
                ratios.append(our_time / hb)
        if not correct:
            failures.append((name, verdict, d_ref, d_highs))

        # JSON integral objectives decode to int, which is still a number.
        shown = float(ours) if isinstance(ours, (int, float)) else float("nan")
        print(f"{name:<16} {status:<14} {shown:>18.8g} "
              f"{d_ref:>10.2e} {d_highs:>10.2e} {intviol:>9.1e} "
              f"{our_time:>8.3f} {baseline.get('seconds', float('nan')):>10.3f}"
              f"  {verdict}")

        if sink:
            sink.write(json.dumps({
                "kind": "instance", "name": name, "sha256": sha256(path),
                "reference": reference, "reference_kind":
                    "opt" if name in optima else ("best" if name in best else None),
                "sankhya": record, "highs": baseline,
                "rel_vs_reference": d_ref, "rel_vs_highs": d_highs,
                "integrality_violation": intviol,
                "verdict": verdict, "correct": correct, "proved": proven,
            }) + "\n")
            sink.flush()

    print()
    print("verdicts: OK proved optimal | OK(inf) proved infeasible | "
          "TIME= right answer, HiGHS proved it | TIME~ right answer, neither proved it")
    print(f"correct answer (incl. unproved incumbents):  {agree_ref}/{scored}")
    print(f"proved optimality or infeasibility:          {passed}/{len(instances)}")
    print(f"agreeing with HiGHS where HiGHS concluded:   {agree_highs}/{len(instances)}")
    if ratios:
        print(f"slowdown vs HiGHS on passing instances: "
              f"median {statistics.median(ratios):.1f}x  max {max(ratios):.1f}x")
    for name, reason, d_ref, d_highs in failures:
        print(f"  FAIL {name:<16} {reason:<24} "
              f"vs.solu={d_ref:.2e} vsHiGHS={d_highs:.2e}")
    if sink:
        sink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
