# Validated cover-cut ablation

This is a controlled native comparison on the bundled binary `tiny_milp.mps`
instance. Both runs used the same Release executable and a five-second limit;
only the validated cover-cut switch changed.

| Mode | Status | Objective | Nodes | LP iterations | Solve seconds |
|---|---|---:|---:|---:|---:|
| cuts enabled | optimal | -2 | 0 | 2 | 0.0000954 |
| `--no-cuts` | optimal | -2 | 3 | 4 | 0.0003793 |

The cut-enabled relaxation is integral at the root, so branch-and-bound does
not open a search node. This is a correctness/performance smoke measurement,
not a claim about large MIPLIB performance; the classic MIPLIB record remains
the authoritative industrial-scale comparison.

Commands:

```text
sk_bench.exe --time-limit 5 bench\smoke\tiny_milp.mps
sk_bench.exe --no-cuts --time-limit 5 bench\smoke\tiny_milp.mps
```
