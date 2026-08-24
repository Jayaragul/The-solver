# Raw measurement outputs

Unedited stdout from the benchmark and validation runs that the claims in
`docs/architecture/` and `docs/research/SOTA.md` rest on. They are checked in so
that every quoted number can be traced to the run that produced it.

| file | command | what it establishes |
|---|---|---|
| `netlib-validation-20000rows.txt` | `validate_netlib data/netlib_lp 20000` | 92/93 pass at the 20,000-row cap (`NUMERICS.md` §3.2) |
| `pdlp-vs-cpu-under-2500rows.txt` | `bench_pdlp data/netlib_lp/feasible 2500 1e-6 256 40 700 2500` | GPU PDLP is 4.4× slower than the CPU simplex below 2,500 rows (`PDLP.md` §5) |
| `pdlp-vs-cpu-2500-20000rows.txt` | `bench_pdlp data/netlib_lp/feasible 20000 1e-6 256 60 2500 20000` | GPU PDLP is 1.56× faster from 2,500–20,000 rows, and solves `dfl001` where the simplex fails (`PDLP.md` §5) |

## The one rule these files exist to enforce

**Every timing here was taken in a single process with nothing else running.**

That rule is not a stylistic preference. An earlier round of PDLP measurements
was taken with builds and test suites running concurrently, each spawning 16
OpenMP threads on a 16-core machine. Because pricing loops above
`kParallelNnzThreshold` genuinely parallelize, the contamination landed almost
entirely on the CPU column and produced a reported "39× GPU speedup on `degen3`"
that clean re-measurement showed to be a **tie** — and an "8× speedup on
`d2q06c`" that was really a **2.8× loss**. The conclusion drawn from those
numbers (that the GPU path wins on degenerate models) was wrong in mechanism as
well as magnitude.

`PDLP.md` §5 records the full account. The short version: on this machine
concurrent benchmark processes are not merely noisy, they are *actively
misleading*, because oversubscription hits size-gated parallel paths far harder
than serial ones — so the error has a consistent sign and looks like a result.

Anything added here must state its command line and must have been run alone.
