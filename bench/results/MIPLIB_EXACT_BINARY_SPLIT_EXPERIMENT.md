# Exact binary split experiment

Date: 2026-09-07

The comparison repository contains an opt-in meet-in-the-middle enumerator for
binary equality systems with nonnegative integral coefficients and one unit
slack per row. We prototyped that approach against `markshare2`, the same
MIPLIB instance that currently exposes our weakest MILP bound.

## Protocol

```text
bench_miplib.exe data/miplib2017_small \
  data/miplib2017_small/miplib2017-v36.solu markshare2 \
  60 pseudocost on serial 1 off 50000 off off off on 0.0
```

The prototype used a 2.5 GB table budget and two enumeration workers. It did
not complete within 90 seconds on this laptop; no incumbent or certificate was
accepted from the experiment. The general branch-and-bound path was restored
unchanged and remains the production path.

## Decision

The method is structurally sound, but its complete enumeration cost is not
acceptable for this hardware and instance size without a substantially better
state-space reduction. It is therefore recorded as a rejected experiment,
not exposed as an unbounded solver option and not counted as benchmark
performance. A future implementation must add a proven time/memory budget and
an instance-family reduction before reconsideration.
