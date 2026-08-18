# Revisiting exact-penalty linear programming: a projection-path walker for the regime where Newton slows down

**Release state: DRAFT.** Claims and artifacts in this repository may still
change. There is no stable tag, no permanent archive, and no DOI yet, so there
is nothing here that should be cited as a fixed version.

A linear program can be solved exactly by minimizing a quadratic penalty
function, provided the penalty parameter is large enough but finite. That is
Mangasarian's result, and it turns an LP into a sequence of smooth problems
that a Newton method can attack. Mangasarian aimed it at very tall systems,
where it does well. It slows down sharply when constraints outnumber variables
by less than about 2 or 3.

Converted to the `Bx ≥ b` form used throughout this repository, all 27 Netlib
models sit in that band, between 1.5 and 2.9. In canonical MPS notation the
models have fewer explicit rows than columns, but MPS also imposes default
variable bounds. The conversion makes those bounds explicit and represents
equalities as paired inequalities. The ratio therefore describes the explicit
inequality system solved here, not merely the number of MPS row records.

This repository contains a second method for that regime, called the
**t-walker**, together with the code, fixtures, and frozen measurements needed
to check what it does.

> **What is claimed.** On generated low-ratio instances, the t-walker beats
> Newton in eleven of twelve certified cells. Taking the fastest certified
> result of the two t-walker seeds and staged Newton is a median 2.82 times
> faster than Newton at ratios from 1.1 to 2. Choosing the fastest of those
> three methods per problem uses hindsight, not a dispatcher, and remains about
> 17 times slower than HiGHS in geometric mean.

A *face* of the dual feasible set is the subset where a given set of the
constraints `y ≥ 0` holds with equality; the path moves from one to the next.
*Certified* means the returned pair passed the accuracy check described under
[Evidence and its limits](#evidence-and-its-limits). *Netlib* is a long-standing
public collection of benchmark linear programs.

The technical note is [`paper/main.pdf`](paper/main.pdf);
its source is [`paper/main.tex`](paper/main.tex).

## The methods

Write the primal and dual linear programs as

```text
min  dᵀx   subject to  Bx ≥ b
max  bᵀy   subject to  Bᵀy = d,  y ≥ 0
```

Let `D = {y ≥ 0 : Bᵀy = d}` be the dual feasible set. Mangasarian's penalty
program returns exactly the projection of `tb` onto `D`:

```text
y(t) = P_D(tb) = argmin { ½‖y − tb‖²  :  y ∈ D }
```

As `t` grows this traces a path that is continuous and piecewise affine. The
two methods differ in how they travel it:

- **Staged Newton** solves a semismooth system at a fixed `t`, raises `t` in
  stages, and polishes at the end. It can cross many support changes in a few
  iterations, which is why it does well on tall systems.
- **The t-walker** holds one affine segment `y_W(t) = tg + h` on the current
  face and computes the next breakpoint by a minimum-ratio test. It pays for
  every breakpoint, so long paths are expensive, but near the square end the
  progress is cheap and explicit.
- **The triangular-seeded t-walker** runs that same walk, but starts from a
  QR/triangular fixed-`t = 0` projection seed instead of the default Newton
  seed. It presently requires full column rank and fails closed on some
  problems — on 3 of the 27 synthetic cells, for instance.

Those three are the methods the measured frontier selects among.

Write `φ(t) = max { t·bᵀy − ½‖y‖² : y ∈ D }`. Within a face `φ″(t) = gᵀg`, so
`gᵀg ≤ tol` detects a stationary face. That is **necessary but not sufficient**
for termination — `g` vanishes identically on any face whose active rows are
independent, including every vertex the path passes through — so the t-walker
accepts an endpoint only when the ratio scan also shows no forward event and
the certificate passes.

## What is new — and what is not

**The path is not new.** Mangasarian and Meyer established finite exactness in
1979; Madsen, Nielsen, and Pinar developed finite continuation methods on
piecewise-linear paths; and Pinar's 1997 algorithm already follows a
quadratic-penalty path with predictor steps, modified-Newton correction,
retained factor updates, and iterative refinement. After dualization and the
change of parameter `t = 1/ε`, Pinar's perturbed optimizer *is* `P_D(tb)` —
the same path used here. That equivalence is an exact derivation carried out
in this work, not a quotation from the cited paper. The distinction is
algorithmic: Pinar evaluates selected penalty values with predictor and
corrector steps, while the t-walker follows `y(t)` face by face and seeks the
next breakpoint. General parametric-QP active-set methods supply the larger
setting for piecewise-affine optimizer maps, ratio events, dependent constraint
exchanges, and factor updates. **No new path or homotopy principle is claimed.**

The narrower contributions are:

1. **A sparse direct breakpoint implementation** of that classical path, with
   the t-walker state maintained like a numerical simplex state — face
   coefficients, sparse products, support statuses, and a rank-revealing
   square core carried across pivots rather than re-solved.
2. **Original-data certification.** Every accepted result must pass one KKT
   certificate on the unscaled input, independent of `t`, using componentwise
   scaling on the primal and dual residuals. Solver status strings are not
   accepted as evidence of accuracy.
3. **Rank-deficient-face repairs**, including a weak-subspace SVD/COD repair
   and deterministic statuses for dependent zero rows, so that fixed-`t`
   exchanges cannot cycle by revisiting an equivalent support label.
4. **An empirical demonstration** that face following complements a
   Mangasarian-style Newton solver in the measured low-ratio regime.

## Evidence and its limits

Four kinds of support are kept separate:

- **Measurement.** Timings and accuracy on a 24-cell synthetic panel and the
  27-model Netlib panel, under one accuracy standard, from the frozen records
  in [`records/`](records/).
- **Implementation.** The C++ t-walker in [`cpp/twalker/`](cpp/twalker/) and the
  Python harness in [`experiments/`](experiments/).
- **Literature.** The projection path, finite exactness, and the parametric-QP
  setting are credited to earlier work and are not claimed here.
- **Exploration.** The provisional reconstruction of Pinar's 1997 algorithm in
  [`experiments/pinar1997/`](experiments/pinar1997/) is a correctness-first
  reference implementation. It is not a reproduction of Pinar's Fortran costs
  and should not be read as a measurement of his method.

Limitations that matter:

- **This is not a competitive LP solver.** On Netlib-27 the measured frontier
  is 16.6× slower than the faster HiGHS engine per model, geometric mean. That
  is an improvement on Newton alone (26.5×), not parity.
- **The frontier uses hindsight.** Choosing the fastest certified result of
  three methods per problem is a measured lower envelope, not a dispatcher. No
  rule is supplied that picks the right method without running them.
- **Coverage is incomplete.** The t-walker does not certify every Netlib model.
- **The prototype is hybrid.** By default an in-process Newton method builds the
  t-walker's `t = 0` seed, and on difficult faces the t-walker can invoke
  HiGHS-backed endpoint selection or terminal face repair. That time is counted
  against the t-walker, but the benchmark does not isolate purely native path
  algebra.
- **The generator confounds two variables.** It changes aspect ratio and
  planted support geometry together, so the synthetic panel establishes a
  regime, not a theorem that aspect ratio alone is causal.
- **The literature review was bounded, not exhaustive.** Novelty language
  should be read as provisional until an optimization specialist has checked
  the equation map and bibliography.

## Reproduce

Python 3.9 with the pinned environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Build the C++ t-walker. It needs SuiteSparse (SPQR, CHOLMOD), OSQP, and the
HiGHS shared library; point `PYTHON` at the virtualenv so the Makefile can
locate `highspy`:

```bash
cd cpp/twalker
make build/verify_walker PYTHON=../../.venv/bin/python
```

Then, from the repository root:

```bash
# Timing figures, from the frozen records
.venv/bin/python experiments/render_solver_timing_charts.py \
    --output-dir figures/solver_timings

# Common-accuracy figure, from the frozen accuracy table
.venv/bin/python experiments/render_common_accuracy.py \
    --output figures/common_accuracy/common_accuracy

# Provisional Pinar reference on the 27-model panel
.venv/bin/python experiments/pinar1997/run_netlib_panel.py \
    --models adlittle afiro bandm beaconfd blend boeing2 brandy capri \
             degen2 e226 fit1d grow7 israel kb2 lotfi recipe sc105 sc205 \
             sc50a sc50b scagr7 scorpion sctap1 share1b share2b ship04s \
             stocfor1 \
    --timeout 10 --output records/pinar1997/netlib_panel.json

# Netlib t-walker panel, from scratch (needs the built verify_walker)
.venv/bin/python experiments/bench_twalker_netlib_panel.py \
    --output records/twalker_cpp/netlib27_rerun.json

# The PDF (deterministic; two clean builds are byte-identical)
sh paper/build_paper.sh
```

Re-running the t-walker panel and recomputing the frontier lands within about
1% of the published 9.60 s. The t-walker's path is deterministic on a fixed
build, but the current build does not reproduce the frozen record's internal
counters on every model, and `beaconfd` has regressed. See
[`VERIFICATION.md`](VERIFICATION.md) §5 for the measurements.

Regenerating the accuracy table from scratch, rather than rendering the frozen
one, also runs the solvers and requires the built `verify_walker`:

```bash
.venv/bin/python experiments/bench_common_accuracy.py
```

See [`VERIFICATION.md`](VERIFICATION.md) for exact outputs, tool versions, and
what each command does and does not re-derive.

## Repository map

- `paper/` — the technical note source, bibliography, figures, and the
  deterministic build script.
- `cpp/twalker/` — the C++ t-walker: face algebra (`src/`), maintained-state
  solvers (`revised/`), unit tests (`tests/`), fixture tools (`tools/`), and
  the compact Netlib panel fixtures (`fixtures_panel/`).
- `experiments/` — the Python harness: benchmark drivers, the frozen
  certificate, the staged Newton reference, and the figure renderers.
- `experiments/pinar1997/` — the provisional reconstruction of Pinar's 1997
  algorithm, which calls neither the t-walker nor an LP solver.
- `netlib/` — the canonical Netlib MPS fixtures.
- `records/` — frozen measurement records. The figures are rendered from these,
  not from a live run.
- `figures/` — rendered output.

## AI assistance and responsibility

Jeff Kline directed the mathematical and computational work. Large language
models substantially assisted with drafting code, running bounded experiments,
auditing claims, and editing this prototype. They are neither authors nor
referees, and their agreement with each other is evidence about the checking
process, not a certificate of correctness. Jeff Kline is responsible for the
claims released under his name and will record material corrections or
withdrawals in [`CORRECTIONS.md`](CORRECTIONS.md).

## Citation

This release is in DRAFT state and has no DOI. Do not cite it as a fixed
version yet. When a version is archived, the citation metadata will be in
`CITATION.cff` and recorded here.

## References

- O. L. Mangasarian and R. R. Meyer, "Nonlinear perturbation of linear
  programs," *SIAM J. Control Optim.* 17 (1979), 745–752.
- O. L. Mangasarian, "Normal solutions of linear programs," *Math. Programming
  Study* 22 (1984), 206–216.
- O. L. Mangasarian, "A Newton method for linear programming," *J. Optim.
  Theory Appl.* 121 (2004), 1–18.
- K. Madsen, H. B. Nielsen, and M. Ç. Pinar, "A new finite continuation
  algorithm for linear programming," *SIAM J. Optim.* 6 (1996), 600–616.
- M. Ç. Pinar, "Piecewise-linear pathways to the optimal solution set in linear
  programming," *J. Optim. Theory Appl.* 93 (1997), 619–634.
- Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex
  method," *Math. Prog. Comp.* 10 (2018), 119–142.

The full bibliography is in
[`paper/main.bib`](paper/main.bib).

## License

Original source, text, and figures are copyright 2026 Jeffery Kline and
licensed under the GNU General Public License version 3 only
([`LICENSE`](LICENSE)).

The GPL permits commercial use; what it requires is that anyone distributing a
derivative also release that derivative's source under the GPL. This work is
additionally available under separate commercial terms for use where those
obligations are unwanted — contact Jeff Kline. Note that building the t-walker
links against SuiteSparse, whose own copyleft terms are independent of anything
granted here; see [`NOTICE`](NOTICE) for the full third-party record.

The Netlib MPS fixtures in `netlib/` are the canonical public benchmark files
and retain their own provenance; they are not original to this repository.
