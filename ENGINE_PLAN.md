# ENGINE PLAN — one benchmark, one engine

Zoom-in on the critical path of `PLAN.md`. Nothing here contradicts that document; this is
`H+0 → H+24` of it, executed properly, with everything else deleted from view until it works.

---

## 1. The benchmark: **Netlib LP** (the canonical 98-instance suite)

**Chosen. The others are not chosen — here is why.**

| Candidate | Verdict |
|---|---|
| **Netlib LP (98)** | **CHOSEN.** See below. |
| MIPLIB 2017 subset | Needs a working LP core *plus* B&B. It is Netlib + 3 more layers. Not a first benchmark, a fourth one. |
| QPLIB convex subset | Needs a second numerical stack (PDHG proximal or IPM). Same problem. |
| Mittelmann large LP | Same algorithm as Netlib, but instances large enough that a bug costs an hour per debug cycle instead of a second. Wrong tool for finding bugs. |
| Our industrial families | We write both the generator and the solver, so a shared misunderstanding of MPS semantics passes silently. No independent ground truth. Useless as a first correctness target. |

### Why Netlib is the right single target

1. **Every instance has a published optimal objective.** A scoreboard exists from hour one that
   we did not author. `N/98` is a number nobody has to take our word for.
2. **Pure LP.** One algorithm. No integrality, no quadratic term, no tree search. The entire
   rest of `PLAN.md` — B&B, PDHG, QP, GPU — sits *on top of* this core. Solving Netlib is not a
   detour from the roadmap, it is the load-bearing member of it.
3. **Small enough to debug at speed.** AFIRO is ~27×32. The largest, DFL001, is ~6k rows ×
   ~12k cols; MAROS-R7 is the densest at ~144k nonzeros. The whole suite is tens of MB. Every
   debug cycle is seconds, not minutes. This is the property that matters most right now.
4. **Adversarial by construction.** DEGEN2/DEGEN3 are massively degenerate. PILOT87, GREENBEA,
   GREENBEB, D2Q06C, PEROLD are numerically vicious — they are *why* Harris ratio tests and
   Markowitz thresholds exist. Getting 85+/98 is a real robustness result, not a toy one.
5. **The primal-dual certificate works here and only here.** LP has an exact optimality
   certificate `(x, y, z)`. MILP does not — the bound requires verifying the whole search tree.
   Our single strongest differentiator is only available on an LP benchmark.

### The guardrail — non-negotiable

`PLAN.md §6.1` designates Netlib the *immutable correctness baseline, never a tuning target*.
Making it the sole benchmark puts that at risk, so it is restated here as a rule:

> **Netlib is a PASS/FAIL correctness target. It is never a performance-tuning target.**
> We tune nothing on it. No parameter is chosen by looking at a Netlib timing. No instance is
> special-cased. The AutoResearch loop never sees these files.

If we ever find ourselves tuning against Netlib timings, we have converted our credibility
asset into an overfitting liability.

---

## 2. The KPI

One number drives the whole build:

```
N / 98   solved to   |obj - obj*| / (1 + |obj*|) <= 1e-6
         AND passing an independent primal-dual certificate check
```

Not "solved". Not "terminated". **Certificate-verified.** A run that reports OPTIMAL and fails
the verifier counts as a failure, and is louder than a timeout.

Secondary, reported but never optimized: wall time, iterations, refactorizations, peak RSS.

### KPI-2 — the baseline comparison (requirement, not optional)

> *"...with solution quality and computational performance compared against at least one
> established commercial or open-source solver."*

Every reported run is a **pair**: `sankhya` and **HiGHS** on the same instance, same machine,
same timeout, same MPS file. Two axes, reported separately because they are not the same claim:

| Axis | Metric | Honest expectation |
|---|---|---|
| **Solution quality** | objective agreement to 1e-6; primal/dual infeasibility; certificate status | **We can win here.** HiGHS reports a status; we report a machine-checked certificate, exact over ℚ where the budget allows. |
| **Computational performance** | wall time, iterations, refactorizations, peak RSS | **We will lose, and by how much depends on §7.** HiGHS is 15 years of tuned C++. Reported as measured, never spun. |

Wall time is reported **alongside implementation-independent counters** — simplex iterations,
refactorizations, nonzeros touched. Those separate *algorithmic* quality from language overhead,
which is a legitimate comparison and the only one a Python prototype can win. It is legitimate
only if raw wall time is printed in the same table, so it always is.

**A quality win we do not get for free:** where our exact rational certificate passes and HiGHS's
floating-point answer disagrees in the last digits, that is *our* result to report — but only if
we also report every instance where HiGHS is right and we are not.

---

## 3. Engine architecture — build order is dependency order

Each layer has a **gate**. The gate is a test that must pass before the next layer is written.
This ordering is the entire defence against the failure mode that kills solver projects: a
subtle bug in the factorization that only surfaces as "the simplex stalls on instance 43".

```
L0   data + manifest         -> 98 files, checksums, reference optima
L1   MPS reader + model      -> min c'x,  L <= Ax <= U,  l <= x <= u,  CSC
L2   scaling                 -> geometric + equilibration, power-of-2
L3   sparse LU               -> Markowitz + threshold partial pivoting   <-- HIGHEST RISK
L4   basis maintenance       -> PFI eta updates + refactorization policy
L5   primal revised simplex  -> bounded variables, phase 1 + phase 2
L5b  robustness pass         -> Harris 2-pass + bound flip, Devex, perturbation, Bland
L6   dual simplex            -> second attack path; warm starts later
L7   presolve + postsolve    -> with round-trip test
L8   certificate + verifier  -> float pass, then exact rational pass
L9   baseline comparison     -> HiGHS as a separate process; quality + performance table
```

### L0 — data and manifest · ~1.5 h

Netlib distributes most instances in a compressed form; `emps.c` from the same directory is the
decompressor. We reimplement it in Python — it is short, and it keeps the pipeline ours.

Reference optima come from the netlib readme table, **cross-checked against a second published
source**. Any disagreement is recorded as a disagreement rather than silently resolved.

Produces `bench/NETLIB_MANIFEST.txt`: filename · SHA-256 · rows · cols · nnz · reference
optimum · source of that optimum · exclusion + reason if excluded.

> `N/98` is meaningless without this file. The literature quotes 89-, 93- and 98-instance
> variants with undocumented exclusions. Ours are documented.

**Gate:** 98 files present, 98 checksums, 98 reference optima, zero unexplained exclusions.

### L1 — MPS reader and model · ~3 h

Fixed-format MPS (Netlib is fixed-format; free-format support comes almost free afterwards).
Sections: `NAME` `ROWS` `COLUMNS` `RHS` `RANGES` `BOUNDS` `ENDATA`.

The traps, listed because each one silently corrupts a subset of the suite:

- `RANGES` on an `E` row — the **sign** of R decides which side moves. Get it wrong and you
  quietly solve a different problem on a handful of instances.
- `UP` bound with a **negative** value — conventionally implies the lower bound drops to `-inf`.
  Genuinely ambiguous across readers. We pick a convention, **write it down in the reader**, and
  record which instances it touches.
- Missing `BOUNDS` section → default `[0, +inf)`.
- An `RHS` entry on the objective row is the **negative** of a constant term in the objective.
  Forgetting this fails the objective comparison by a constant on a few instances.
- `FR` / `MI` / `PL` / `FX` / `BV` all appear. `MI` with no matching `UP` means `(-inf, +inf)`
  under some conventions and `(-inf, 0]` under others. Same treatment: pick, document, record.

Internal form is the computational one, not the file one:

```
[A  -I] [x; s] = 0     with   L <= s <= U,   l <= x <= u
```

Every row gets a logical variable, so the basis is always a square `m × m` matrix and there is
exactly one code path.

**Gate:** read → write → re-read is identical for all 98; dimensions match the manifest for
all 98.

### L2 — scaling · ~1 h

Iterated geometric-mean row/column scaling, then equilibration, with all scale factors **rounded
to powers of 2** so scaling introduces exactly zero floating-point error.

**Gate:** the max/min `|a_ij|` ratio drops by orders of magnitude on the known-nasty instances;
the objective is unchanged on the easy ones with scaling on vs. off.

### L3 — sparse LU · ~5 h · **the highest-risk component in the project**

Markowitz ordering with threshold partial pivoting (`u ≈ 0.1`), CSC, our own data structures.

Written and fully tested **before a single line of simplex code exists.** This ordering is not
negotiable. A wrong LU does not announce itself — it shows up ten hours later as "the simplex
stalls", and costs a day.

**Gate:** on random sparse matrices *and* deliberately pathological ones (near-singular,
wildly-scaled, structurally rank-deficient):
`||PAQ - LU||_inf / ||A||_inf < 1e-12`, and the `Bx = b` residual within tolerance — both
checked against an independently written dense reference LU.

### L4 — basis maintenance · ~2.5 h

Product-form (PFI) eta updates. Refactorize on pivot count (~50–100), eta-file growth, or
measured accuracy drift. Forrest–Tomlin stays on the roadmap — PFI is simpler and correct, and
correct is the KPI.

**Gate:** after k updates the residual `||Bx - b||` stays bounded; drift triggers a forced
refactorization; refactorization reproduces the same basis solution.

### L5 — primal revised simplex, bounded variables · ~5 h

All-logical starting basis. Phase 1 minimizes the sum of infeasibilities (piecewise-linear
composite objective) → Phase 2. **Dantzig pricing and a textbook ratio test first** — the goal
of this layer is a solver that is *correct on the easy half*, not a fast one.

**Gate:** AFIRO, then SC50A / SC50B / SC105 / ADLITTLE / BLEND / SHARE2B / KB2 / RECIPE —
objective matches the manifest to 1e-9.

### L5b — the robustness pass · ~4 h · *this is where 20/98 becomes 60/98*

Harris two-pass ratio test with bound flipping · Devex pricing · cost perturbation · Bland's
rule fallback after N stalls · numerical-trouble detection with recovery-by-refactorization.

**Gate:** 60/98. DEGEN2 and DEGEN3 must terminate without cycling.

### L6 — dual simplex · ~4 h

A second attack path — some Netlib instances that resist the primal fall immediately to the
dual. It is also what MILP warm starts will need later, so it is not a detour.

**Gate:** 70/98, with the driver allowed to try primal then dual.

### L7 — presolve + postsolve · ~4 h

Empty and singleton rows/columns, forcing and redundant constraints, fixed variables, dominated
and implied-free columns, duplicate rows.

**Postsolve is written in the same sitting as presolve, never later.** A presolve without a
correct postsolve produces a solution to the wrong problem — the worst possible failure, because
it looks exactly like success.

**Gate:** round-trip — for every instance, presolve → solve → postsolve reconstructs a solution
to the *original* model that passes the L8 certificate. Not "close to". Passes.

### L8 — certificate and verifier · ~2.5 h + ~2.5 h

The solver emits `(x, y, z)`. A **separate process**, which never imports the solver, reads only
the original untouched MPS and checks:

1. **primal feasibility** — `L <= Ax <= U`, `l <= x <= u`
2. **dual feasibility** — `c - A'y - z = 0`, with sign conditions per row type and bound status
3. **complementary slackness / zero duality gap**

Then the **exact pass**: the same three checks in `fractions.Fraction` from the reported basis,
for every instance inside a size budget. Where it completes, "we believe this is optimal" becomes
a machine-checked proof over ℚ — no tolerance, no trusted reference value, and no trust in our
own solver.

**Gate:** every instance we claim solved passes the float certificate; ≥40 pass the exact one.

### Harness · ~2 h

`bench/run.py` → SQLite → one table, one command, `N/98` at the top. Built at L5, not at the end,
because the scoreboard is what drives every subsequent decision.

### L9 — baseline comparison · ~3 h

**The baseline is HiGHS.** Open-source, the reference LP/MIP solver in the Mittelmann
benchmarks, and the strongest thing we could be measured against — which is the point. CBC or
GLPK may be added as a third column; neither replaces HiGHS.

**The sovereignty firewall — the rule that makes this safe.** `PLAN.md §4` permits HiGHS as an
external reference *process*: "never linked, never in our address space." So:

- The baseline runs in a **subprocess**, never imported by the engine or the harness parent.
- `bench/baselines/run_highs.py` is the only file in the tree allowed to import it. It reads an
  MPS path, writes a JSON result, exits.
- A test asserts the engine's import closure contains no baseline package. It is a test, not a
  convention, because the whole sovereignty claim dies quietly the first time someone adds a
  convenience import.

**Install.** `highspy` from PyPI — a pip wheel, no C++ toolchain, unaffected by §6. Version and
wheel hash get pinned into `bench/BASELINE_MANIFEST.txt` alongside the Netlib checksums, because
"compared against HiGHS" means nothing without saying which HiGHS. **Vendor the wheel offline**
(`PLAN.md §11`, no-internet risk).

*Not* `scipy.optimize.linprog` — it is HiGHS underneath, so it is the same data point wearing a
different name. If it is ever used it gets labelled as HiGHS.

**The timing trap on Netlib.** Netlib instances are tiny: HiGHS solves AFIRO in well under a
millisecond, and a subprocess timing comparison at that scale measures Python startup and MPS
parsing, not either solver. So the performance axis needs bigger inputs, which is where the
requirement's *"or Mittelmann"* earns its place:

- **Netlib (98)** — the *quality* axis. Correctness, certificates, objective agreement. This is
  the pass/fail benchmark and it does not change.
- **Netlib Kennington + the smaller Mittelmann LPs** — the *performance* axis. Added once the
  core is green, sized so a solve is seconds not microseconds and the comparison is real.

Protocol per `PLAN.md §5.4`: fresh process, hard timeout, memory cap, median of 3, randomized
order. Both solvers get the same untouched MPS — never our presolved model.

**Gate:** every instance in the scoreboard has a HiGHS result beside it — objective, status, wall
time, iterations — and every disagreement is listed by instance with which side the certificate
supports.

---

## 4. Milestone ladder

| # | Milestone | Unlocked by |
|---|---|---|
| M1 | AFIRO solves, objective matches to 1e-9 | L0–L5 |
| M2 | **20 / 98** — the easy set | L5 |
| M3 | **60 / 98** | L5b — Harris + Devex + perturbation |
| M4 | **70 / 98** | L6 — dual simplex |
| M5 | **85 / 98** — the commit number | L7 — presolve |
| M6 | **90+ / 98** — the hard tail | grinding |

The hard tail is roughly PILOT87, GREENBEA, GREENBEB, DFL001, D2Q06C, CYCLE, PEROLD, PILOT-JA,
MAROS-R7, FIT2P. These are instances real solvers spent *years* on. We commit to 85 and report
honestly whatever the tail gives us.

---

## 5. Effort

| Layer | Hours |
|---|---|
| L0 data + manifest | 1.5 |
| L1 MPS + model | 3.0 |
| L2 scaling | 1.0 |
| L3 sparse LU | 5.0 |
| L4 basis maintenance | 2.5 |
| L5 primal simplex | 5.0 |
| L5b robustness pass | 4.0 |
| L6 dual simplex | 4.0 |
| L7 presolve / postsolve | 4.0 |
| L8 certificate + exact verifier | 5.0 |
| harness | 2.0 |
| L9 baseline comparison | 3.0 |
| **Total focused hours** | **~40** |

That is *the LP core alone*, and it is why choosing one benchmark is the correct call. It also
lines up with `PLAN.md`'s own allocation (H+2 → H+24), which is a good sign the original estimate
was sound.

---

## 6. Machine reality — verified, not assumed

```
cl / g++ / clang++   NOT FOUND
cmake / ninja        NOT FOUND
nvcc                 NOT FOUND
python               3.11.9
numpy                2.4.6
scipy                not installed
git                  present
highspy / scipy      NOT INSTALLED   <-- baseline solver; pip wheel, needs one download
highs/cbc/glpsol/scip  NOT FOUND     <-- no baseline binary on this machine
C: free              29.5 GB       R: free   33.6 GB
```

No C++ toolchain exists yet. `PLAN.md` budgeted ~90 min to install one; that estimate stands, but
it is now a *decision* rather than a given — see §7.

---

## 7. The one open decision — implementation track

The failure mode of this project is **a numerical bug**, not slowness. Netlib is small; nothing
here is compute-bound at this stage. That reframes the language question.

**Track A — Python/NumPy engine now, C++ port later. (recommended)**
Our own CSC arrays, our own LU, NumPy for dense vector arithmetic only, no SciPy.
*For:* zero install; seconds-long debug cycles on exactly the code most likely to be wrong
(Markowitz LU, Harris ratio test, degeneracy); Netlib is small enough that Python genuinely
reaches 85/98; and the result is **permanently reusable as the differential-testing oracle for
the C++ port** — every pivot sequence becomes checkable.
*Against:* written twice; the hard tail (DFL001, MAROS-R7, PILOT87) may exceed a sane timeout;
NumPy links BLAS, so *the prototype* is not sovereign — only the C++ core will be. Fine, as long
as we never claim otherwise.
*Condition:* written in **port-ready style** — flat NumPy arrays, explicit index arithmetic, no
dicts or objects in hot paths. That discipline is what makes the port a translation rather than a
rewrite.

**Track B — C++17 immediately.**
*For:* one implementation; the shipped artifact; sovereignty and performance intact from day one.
*Against:* toolchain install first (VS Build Tools, several GB onto a 29.5 GB C: drive), and
debugging a Markowitz LU in C++ with no debugger configured is the slowest possible way to find
the bug that L3 exists to catch.

**Track C — hybrid.**
C++ core, but L3 (LU) and the ratio test are written in Python **first**, as executable
specifications and test oracles, then implemented in C++ against them. Costs ~4 h more than B and
removes most of B's risk.

**What KPI-2 does to this decision.** The requirement asks for *computational performance*
compared against HiGHS, and that is the one axis a Python engine cannot argue its way out of —
expect 1-3 orders of magnitude on the larger instances. So the choice is no longer "Python now,
C++ maybe":

- Track A still wins the **quality** axis outright, and wins it *sooner*, which is what
  determines whether there is a comparison to publish at all.
- But the C++ port stops being optional. It moves from "later" to **the committed path to a
  defensible performance number**, and the schedule has to name a point at which it starts.
- Until it lands, the performance column is reported with the implementation-independent counters
  beside it and a plain sentence saying this is a Python prototype. That is a weak result stated
  honestly, which survives scrutiny; a wall-time number quietly omitted does not.

**Recommendation: Track A, with the C++ port promoted to committed scope.** Get to a verified
`N/98` first — a fast solver with a numerical bug scores zero on both axes. Port with the oracle
in hand and a green scoreboard to port *against*. Track C is the right answer instead if the
performance comparison is weighted as heavily as the correctness one, or if the C++ artifact is a
hard requirement of a clock that is already running.

---

## 8. First session, if approved

1. **L0** — fetch the suite, decompress, build `NETLIB_MANIFEST.txt` with checksums and reference
   optima. Ends with 98 verified files on disk.
2. **L0b** — install and pin the baseline: `highspy` wheel + version + hash into
   `bench/BASELINE_MANIFEST.txt`, wheel vendored for offline use, and `bench/baselines/run_highs.py`
   working as a subprocess over one instance. **Do this now, not at L9** — it gives an independent
   second opinion on every objective from L1 onward, and it is the cheapest hour in the plan.
   *Needs a download, so it needs your go-ahead.*
3. **L1** — MPS reader + model, with the round-trip test over all 98 as the gate. HiGHS reads the
   same 98 files: any instance it parses and we do not is a reader bug found on day one.
4. **Stop and report** — how many of the 98 parse cleanly, every convention ambiguity found listed
   by instance, and the HiGHS objective for all 98 as a second reference column beside the
   published optima.

No simplex code is written until L3's gate is green.
