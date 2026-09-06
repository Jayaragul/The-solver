# MIPLIB branching-rule comparison

Date: 2026-09-06

This controlled comparison uses the same native CUDA-release executable, frozen
19-instance MIPLIB 2017 small set, 10-second per-instance limit, warm starts,
serial LP execution, and two independent repetitions. Only the MILP branching
rule changes.

| branching rule | exact/certified | time/node-limit records | notable result |
|---|---:|---:|---|
| reliability | 6 / 19 | 13 / 19 | baseline absolute-tolerance sweep |
| pseudocost | **9 / 19** | 10 / 19 | promoted production default |

Pseudocost also improved the difficult general-integer cases in focused runs:

| instance | reliability incumbent | pseudocost incumbent | pseudocost nodes |
|---|---:|---:|---:|
| `gen-ip002` | -4762.7873726 | **-4769.7405888** | 31,422 median-scale run |
| `gen-ip054` | 6881.9381413 | **6858.2629061** | 40,018 median-scale run |
| `markshare2` | 231 | 231 | 112,923 |

The production default is now `MilpBranchingRule::PSEUDOCOST`; reliability
branching remains available for controlled ablations. The result is a
time-limited benchmark improvement, not a claim that all 19 instances are
solved or that pseudocost dominates on every model family.

As a 60-second spot check after promotion, pseudocost reached the same exact
incumbent on `gen-ip002`, improved `gen-ip054` from 6857.6491 to 6852.3680,
and matched the reliability incumbent (231) on `markshare2`. The existing
19-instance 60-second table remains explicitly labeled as the earlier
reliability baseline until a complete replacement sweep is run.

Raw repeated-run output: [`MIPLIB_BRANCH_PSEUDO_SWEEP_10S_R2.txt`](MIPLIB_BRANCH_PSEUDO_SWEEP_10S_R2.txt).
