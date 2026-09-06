# MILP LP-relaxation time-budget fix

Date: 2026-09-06

The MILP loop previously checked its wall-time limit only between nodes. A
single cold or warm LP relaxation could therefore continue after the MILP
budget expired, producing misleading limit telemetry (especially with the
automatic parallel policy). The solver now propagates the remaining MILP
budget into every simplex relaxation, including heuristic and strong-branching
probes, and maps an interrupted relaxation to `TIME_LIMIT`.

## Regression evidence

Using the native Release build, pseudocost branching, warm starts, automatic
parallel policy, and a 10-second limit:

| instance | result after fix | wall time | nodes |
|---|---|---:|---:|
| `p0201` | OPTIMAL, certified | 1.055 s | 3,277 |
| `noswot` | TIME_LIMIT | 10.036 s | 41,649 |

The focused MILP suite passes **32/32**, including
`milp_time_limit_is_not_reported_as_numerical_failure`.

After the fix, the full 19-instance production-style sweep (pseudocost,
warm-started, automatic parallel mode, 10-second limit) completed with sane
telemetry and **9/19 certified** results. Raw output is retained in
[`MIPLIB_PSEUDO_AUTO_SWEEP_10S.txt`](MIPLIB_PSEUDO_AUTO_SWEEP_10S.txt).
