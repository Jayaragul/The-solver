# MIPLIB 2017 small subset — combined heuristic ablation

Date: 2026-09-11

This native Release sweep enables coefficient-floor cuts, RENS, the bounded
feasibility pump, and ordinary rounding together. It uses pseudocost branching,
warm node bases, serial LP relaxations, and a 10-second limit per instance.

| outcome | count |
|---|---:|
| exact certified reference matches | 9/19 |
| optimal-objective incumbents | 7/19 |
| certified solver results | 9/19 |
| time/node-limit results | 10/19 |

The configuration improves the `pk1` incumbent from 44 to **21** (reference
11), but does not improve the production exact-match count and degrades some
other time-limited incumbents. It therefore remains an opt-in research
configuration rather than a default. The full machine output is retained in
the run log used for this ablation.

The benchmark driver accepts `-` as a shell-safe selector for all instances;
this avoids Windows wrappers dropping an empty quoted argument.
