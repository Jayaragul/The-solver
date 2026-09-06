# Root RENS ablation

RENS restricts each integer variable at the fractional root relaxation to its
adjacent integer neighborhood, fixes root-integral variables, then runs one
LP solve. It can only submit an original-space verified incumbent; it cannot
change B&B bounds or certification.

## Frozen 10-second gate

Pseudocost branching, warm starts, automatic LP parallelism, 50,000 retained
warm bases, one repetition:

| instance | RENS off incumbent | RENS on incumbent | reference |
|---|---:|---:|---:|
| `gen-ip002` | −4769.7405888 | −4769.7405888 | −4783.733392 |
| `gen-ip054` | 6858.2629061 | 6890.6143139 | 6840.9656418 |
| `pk1` | 44 | 44 | 11 |

The raw RENS-on logs are `MIPLIB_RENS_GENIP002_10S.txt`,
`MIPLIB_RENS_GENIP054_10S.txt`, and `MIPLIB_RENS_PK1_10S.txt`.

## Decision

RENS is **implemented but disabled by default**. It has no positive result in
this gate, so enabling it would violate the project’s benchmark-adoption rule.
