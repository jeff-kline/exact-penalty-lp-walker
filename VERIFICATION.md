# Verification record

Exact commands, environment, and observed outputs. Everything below was run in
this repository, in the state it is in now. Where a command was **not** run, or
where an output disagrees with a claim, that is stated rather than omitted.

## Environment

| Component | Version |
|---|---|
| Platform | macOS (Darwin 25.5.0), x86_64 |
| Python | 3.9.6 |
| numpy | 2.0.2 |
| scipy | 1.13.1 |
| matplotlib | 3.9.4 |
| pillow | 11.3.0 |
| highspy (HiGHS) | 1.15.1 |
| clarabel | 0.11.1 |
| sparseqr (PySPQR, optional) | 1.6.0 |
| SuiteSparse | 7.13.0 |
| OSQP | 1.0.0 |
| Compiler | Apple clang, `-O3 -march=native -std=c++17` |
| TeX | pdfTeX (TeX Live), `pdflatex` + `bibtex` |

`-march=native` means the compiled t-walker is tuned to the build host. Timings
are not expected to reproduce exactly on different hardware; statuses and
certificates are.

## What each command does and does not re-derive

The figures render from **frozen records** in `records/`. Rendering them
re-derives the *aggregates and the figures* from those records. It does not
re-run the solvers. The distinction matters and is called out per command.

---

### 1. Timing figures and aggregate table

```sh
.venv/bin/python experiments/render_solver_timing_charts.py \
    --output-dir figures/solver_timings
```

Re-derives: the synthetic and Netlib timing figures, and
`solver_timing_chart_data.json`, from the frozen records.
Does **not** re-run any solver.

Observed: exits 0, writes `synthetic_aspect_timings.{png,svg}`,
`netlib27_timings.{png,svg}`, and `solver_timing_chart_data.json` (27 Netlib
rows).

Aggregates computed from that output, against the technical note:

| Quantity | Note | Observed |
|---|---|---|
| Measured frontier total, Netlib-27 | 9.60 s | 9.596 s |
| Shipped Newton total, Netlib-27 | 11.60 s | 11.599 s |
| Faster HiGHS engine per model, total | 0.370 s | 0.370 s |
| Geometric-mean frontier/HiGHS ratio | 16.6× | 16.60× over 27 models |

**The Netlib aggregates agree.** The *synthetic* figure does not come from this
command. Run as documented, the renderer reads its default record
(`ratio_1_to_512_20260816/summary.json`: 50/100/200 variables, 2 repeats, 33
cells) and produces a figure that differs materially from
`paper/figs/synthetic_aspect_timings_public.png`, which was rendered from
`postinit_m25_50_200_r32_20260816/runs.jsonl` (25/50/200 variables, 5 repeats,
24 cells, with a post-initialization row and a triangular-seeded series).

An earlier version of this section reported this check simply as "Agrees",
having verified only the Netlib aggregates. That was wrong.

A shipped command now bridges the two. `summarize_twalker_synth_nm.py` used to
require every arm on every cell in a repeat, and the triangular-seeded arm
fails closed on 3 of the 27 cells, so it rejected the published record outright
with "fewer than required complete repeats". It now treats an arm that is
absent from every repeat of a cell as a refusal, and `--require-every-arm`
restores the old behaviour. The renderer used to crash on such a cell while
placing its DNF marker; it now drops refused cells instead. Both panels of the
synthetic figure render:

```sh
.venv/bin/python experiments/summarize_twalker_synth_nm.py \
    records/twalker_synth_nm/postinit_m25_50_200_r32_20260816/runs.jsonl \
    --output <tmp>/postinit_summary.json \
    --attempted-repeats 5 --minimum-complete-repeats 2

.venv/bin/python experiments/render_solver_timing_charts.py \
    --synthetic-summary <tmp>/postinit_summary.json \
    --synthetic-m-list 25 50 200 --output-dir <tmp>/figs
```

Observed: the summarizer reports 1185 runs, 5 complete repeats, 0 excluded, and
6 refused cells (3 each for `twalker_triangular` and `twalker_triangular_init`).
The renderer exits 0 and writes both figures.

**The regenerated figure is not pixel-identical to the shipped
`paper/figs/synthetic_aspect_timings_public.png`.** 3.17% of pixels differ.
Four of the six panels match to 0.08–0.49%, which is antialiasing. The other
two — 200 variables complete, and 50 variables post-initialization — differ by
a vertical rescaling of the axis: matching them requires a y-scale factor of
1.07 and 1.14 respectively, after which they agree to 2.7% and 3.0%. The
plotted series coincide; those two panels were drawn with different y-limits.
The frozen PNG predates these repairs, and the renderer as shipped at
migration crashed on this record, so it cannot have produced that PNG.

The numbers behind the figure do reproduce exactly:

| Quantity | Note | Recomputed |
|---|---|---|
| Panel size | 24 cells | 24 (ratio > 1.0) |
| Median speedup, ratio <= 2 | 2.82x | 2.8194x |
| Max cell speedup, ratio <= 2 | 11.76x | 11.7603x |
| Min cell speedup, ratio <= 2 | 1.69x | **1.6950x** via the shipped renderer — agrees. A hand-rolled min-of-medians gives 1.6292 and is the wrong estimator; see `CORRECTIONS.md` |
| Min cell speedup, all 24 cells | not stated | 1.0000x, on five cells at ratio >= 8 where Newton wins outright. The note's range is scoped to ratios at most 2 and is correct as written; these two rows previously carried no scope label |
| Envelope picks | 11 / 6 / 7 | 11 / 6 / 7 |

---

### 2. Common-accuracy figure

```sh
.venv/bin/python experiments/render_common_accuracy.py \
    --output figures/common_accuracy/common_accuracy
```

Re-derives: the accuracy figure from
`records/common_accuracy_20260817/accuracy_rows.json`.
Does **not** re-run solvers or crossover.

Observed: exits 0, writes `common_accuracy.{png,svg}`.

---

### 3. Provisional Pinar panel

```sh
.venv/bin/python experiments/pinar1997/run_netlib_panel.py \
    --models adlittle afiro bandm beaconfd blend boeing2 brandy capri \
             degen2 e226 fit1d grow7 israel kb2 lotfi recipe sc105 sc205 \
             sc50a sc50b scagr7 scorpion sctap1 share1b share2b ship04s \
             stocfor1 \
    --timeout 10 --output records/pinar1997/netlib_panel.json
```

Re-derives: the full panel, from scratch. This **does** run the solver. It
calls neither the t-walker nor an LP solver.

Note that `--models` is required. The script's built-in default is three
models (`afiro`, `sc50b`, `sc50a`), not the panel, so invoking it bare does
not reproduce §5.

Observed, and **run twice with identical statuses on every model**:

| Status | Count | Models |
|---|---|---|
| CERTIFIED | 5 | `afiro` 15.7 ms, `sc50b` 43.5 ms, `sc50a` 61.7 ms, `sc105` 131.5 ms, `stocfor1` 306.9 ms |
| NUMERICAL_FAILURE | 18 | |
| TIME_LIMIT | 3 | `degen2`, `sctap1`, `ship04s` |
| NOT_MEASURED | 1 | `grow7` — no compact panel fixture; export failed with `initialization did not produce an accepted face`. The canonical MPS is shipped. |

This **disagreed** with the technical note as originally written, which
reported 4 certified and 19 numerical failures over 19.9–117.9 ms. The note was
corrected to match the code; see `CORRECTIONS.md`.

---

### 4. C++ t-walker build and panel

```sh
cd cpp/twalker
make build/verify_walker PYTHON=../../.venv/bin/python
./build/verify_walker fixtures_panel/*.twfx
```

Re-derives: the t-walker's status and original-data certificate on each panel
fixture, from scratch. This **does** run the solver.

Observed: builds with warnings only (deprecated Accelerate CBLAS prototypes),
no errors.

| Result | Count |
|---|---|
| Fixtures present | 26. `grow7` has no compact panel fixture: the shipped `cpp/twalker/fixtures_panel/manifest.json` records its export failing with `initialization did not produce an accepted face`. The canonical `netlib/grow7.mps` *is* present |
| CERTIFIED | 24 |
| Rejected — "settle support cycle" | 2 (`brandy`, `lotfi`) |
| Worst certificate residual among certified | 6.99e-09 |
| Total wall time | 81.3 s |

Two qualifications on this run. First, it uses the t-walker's **default**
budgets; the note records that `fit1d` and `lotfi` needed authorized extended
budgets, so `lotfi` failing here is consistent with the note rather than
contrary to it. Second, the frozen record reports `capri` as a cycling model,
whereas `capri` certifies on the current build and `lotfi` does not. This is
**not** run-to-run instability — §5 below shows the t-walker is deterministic
across independent runs — but a difference against the older code state that
produced the record. Claims about *which* models fail are specific to a build.

The certified count of 24 out of 27 models matches the note. The worst
certificate residual, 6.99e-09, is inside the stated 1e-7 acceptance
tolerance.

---

### 5. Netlib t-walker panel, and how far it reproduces

```sh
.venv/bin/python experiments/bench_twalker_netlib_panel.py \
    --output records/twalker_cpp/netlib27_rerun.json
```

Re-derives: the t-walker's status, certificate, and timing on all 27 panel
models, from scratch, one `verify_walker` subprocess per model. This **does**
run the solver and needs the built binary.

Observed: 24 of 27 certified. Substituting these times into the frontier where
a t-walker variant won gives **9.712 s against the published 9.596 s, +1.2%.**

Determinism was measured, not assumed:

| Comparison | Structurally identical |
|---|---|
| Two independent per-model runs | **26 / 26** |
| Batched single invocation vs per-model | **26 / 26** |
| Current build vs the frozen 2026-08-16 record | **11 / 24** |

So the t-walker's path is deterministic on a fixed build. The divergence is
against the older code state that produced the frozen record: pivot counts,
seed iterations, and accepted supports differ on 13 models, and `capri` now
certifies where the record shows a settle-support cycle.

Per-model wall-time ratios against the frozen record have **median 0.96x**,
with 21 of 24 models inside 2x. The exceptions are `sc50a` and `sc50b`, which
finish in single-digit milliseconds where a 3 ms jitter is a large ratio, and
`beaconfd`, which is **11x slower** than the record with its seed iteration
count risen from 11 to 300. That last one is a genuine regression in the
current build. It does not move the headline, because Newton wins `beaconfd`
on the frontier, but it is not measurement noise and is recorded as such.

---

### 7. C++ component verifiers

```sh
cd cpp/twalker
make build/verify_face_solver build/verify_gram_solver \
     build/verify_bound_core_solver PYTHON=../../.venv/bin/python
./build/verify_face_solver                    fixtures_panel/*.twfx
./build/verify_gram_solver       --min-served=12 fixtures_panel/*.twfx
./build/verify_bound_core_solver --min-served=1  fixtures_panel/*.twfx
```

| Verifier | Exit | Coverage | Result |
|---|---|---|---|
| `verify_face_solver` | 0 | 26 faces, one per fixture, 0 declines | worst oracle disagreement 2.97e-09 on `share1b`; worst self-reported residual 6.85e-08, also `share1b` |
| `verify_gram_solver` | 0 | serves 12 of 26, declines 14 | no inaccuracy among those served; floor of 12 asserted |
| `verify_bound_core_solver` | 0 | serves **1** of 26, all 26 eligible | no inaccuracy among the one served; floor of 1 asserted |

Four things must be said plainly about this table.

**`verify_face_solver` used to fail, and its gate was miscalibrated and
incomplete.** Agreement with the answer recorded in the fixture is a useful
regression property and is still checked. But it was the *only* property
checked, at a blanket 1e-10 across 26 faces of very different conditioning,
and the solver's own residual was never asserted at all. On `share1b` the
disagreement is 2.97e-09.

The relevant number is next to it. `FaceSolution::piece_residual` is what the
solver reports about its own answer on the original face data: orthogonality of
`B'g` relative to `‖g‖`, slope error, and constant error, each scaled. On
`share1b` it is 6.85e-08 — the worst on the panel by a factor of 13 over the
next, `israel` at 5.08e-09 — and the ranking of models by that residual tracks
the ranking by oracle disagreement. A forward disagreement of 3e-09 between two
solves whose backward residual is 7e-08 is not evidence of a defect.

The test now asserts both properties separately: `dres` and `piece_residual`
below a ceiling of 1e-6, which nothing checked before and which is the property
that says the returned face solves anything; and oracle agreement below
`max(1e-10, piece_residual)`. Exactly one of the 26 faces is relaxed by the
second term — `share1b` — and the other 25 are still held to 1e-10.

**The relaxation term is empirical, not a proven bound.** A forward error is
bounded by the backward residual times a condition number the test does not
compute. `core_diagonal_ratio` is available as a proxy and does *not* explain
the ranking: `lotfi` has a 50x worse ratio than `share1b` and a 1000x smaller
disagreement. So the scale is chosen because it is the quantity the solver
itself reports about the difficulty of the solve, not because it bounds the
error. Read it as "we do not require agreement tighter than the solve's own
residual", not as a guarantee.

**Coverage is now asserted, and it is thin.** `--min-served=N` fails the run if
the panel serves fewer than `N` faces in total; without the flag there is no
floor, so single-fixture debugging runs still work. The floors are the observed
values, 12 and 1. Both were verified to bite: `--min-served=13` and
`--min-served=2` exit 1 on this panel.

**Each fixture carries exactly one face.** All three verifiers exercise 26
faces in total, one per model, not the thousands the t-walker visits on a full
solve. That is the largest weakness in this section and no flag fixes it. Read
these as smoke tests on one face per model.

---

### 6. Deterministic document build

```sh
sh paper/build_paper.sh
```

Observed: 8 pages, exits 0, no overfull lines, all citations resolved, writes `paper/main.pdf`.

Determinism was checked by removing `tmp/texbuild/` entirely and rebuilding:

```text
6f861b143c76b5bedce010a274ee7d348c667c72508dc019ffb98142b04b9306  build 1
6f861b143c76b5bedce010a274ee7d348c667c72508dc019ffb98142b04b9306  build 2
```

**Byte-identical.** This required a fix: two clean builds previously differed
in 60 of 513,258 bytes, entirely within the PDF trailer `/ID`, which pdfTeX
derives from the output path. `\pdftrailerid{}` is now set in the preamble, and
`SOURCE_DATE_EPOCH` was already pinned in the build script.

---

## Not run

These are documented for completeness. Each is a real gap, not an oversight.

- **`experiments/bench_common_accuracy.py`** — regenerates the accuracy table
  by re-running every solver and crossover. Not executed here; the accuracy
  figure was rendered from the frozen table instead.
- **`experiments/bench_twalker_netlib_panel.py`** now exists and was run; see
  §5 below. An earlier version of this file said no script in the repository
  produced the Netlib t-walker record and that the timings were "not a
  measurement this repository can reproduce." That was too strong: the
  measuring apparatus (`verify_walker`) was always present, and only the loop
  around it was missing. It has been reconstructed and the frontier reproduces
  to 1.2%.
- **`experiments/bench_twalker_synth_nm.py`** — the synthetic timing producer
  is shipped but was not re-executed; the synthetic figures were rendered from
  the frozen summary.
- ~~The C++ unit tests other than `verify_walker` were not built or run.~~
  They have now been built and run over all 26 panel fixtures; see §7 below.
  **One fails.**
- **Audit lanes** — five process-separated AI review lanes ran: four against
  commit `40a4f58` and one confirming pass against `ffa5cc8`. Their dispositions
  are in `CORRECTIONS.md`. Separated agents share training data and blind spots,
  so this is process evidence, not independent expert review. No expert review
  has been obtained.

## Changes made to the migrated code

This repository is a curated extract of a larger private working repository.
Beyond file selection, the following edits were made, all mechanical:

1. Record paths repointed from the private tree's `agent_reports/raw/` and
   `agent_reports/figures/` to this repository's `records/` and `figures/`.
   Provenance strings *inside* the record files were deliberately left
   unchanged, so they still name the private paths the data came from.
2. `requirements-benchmark.txt` replaced by `requirements.txt`, adding the
   missing `clarabel` and `pillow` pins, and the one reference to it updated.
3. `cpp/twalker/Makefile`: the default HiGHS library location no longer
   hardcodes an absolute path inside the author's virtualenv.
4. `paper/main.tex`: `\pdftrailerid{}` added; §5 counts and
   the Netlib figure caption corrected.
5. `experiments/pinar1997/render_provisional_netlib.py`: the figure footnote
   now computes the certified count from its input instead of printing a
   hardcoded literal.
8. The paper was renamed to `paper/main.tex` / `paper/main.bib` /
   `paper/main.pdf` and its layout normalized to match the other releases:
   11pt, 1in margins, `\maketitle`, a centered repository/archive block, and an
   abstract. `microtype` was removed because this TeX installation has no
   scalable T1 fonts for it to expand, and none of the reference papers use it.
   `cmap` was added, which the reference papers do not use; it makes the PDF
   text layer extract ligatures correctly. Greek letters in math mode still do
   not survive text extraction, so the epsilon in the change-of-parameter
   derivation is legible on the page but absent from copied text.

6. `cpp/twalker/README.md`: six reproduction commands that hardcoded an
   absolute path inside the author's virtualenv now use `.venv/bin/python`.
7. `experiments/pinar1997/run_netlib_panel.py`: a missing-fixture message now
   reports a repository-relative path rather than an absolute one, and the one
   absolute path already frozen into `records/pinar1997/netlib_panel.json` was
   rewritten to match. No measurement was affected.

9. `experiments/render_solver_timing_charts.py`: a cell that an arm refused
   outright was still routed to the DNF-marker code, which crashed on its
   absent timing. Refused cells are now dropped from the plot. No plotted
   value changed.
10. `cpp/twalker/README.md` and `cpp/twalker/revised/README.md`: nine pointers
   to internal report files, and one to the excluded `cpp/walk.cpp`, were
   removed. Each was a bare "see `agent_reports/NNN_....md`" appended to prose
   that already states the finding; no claim was carried only by the pointer.

Code comments under `experiments/` still cite internal report numbers
(`agent_reports/11`, `agent_reports/28`, and similar) that are not part of this
release. These are annotations on promoted defaults inside source files, not
reader-facing prose, and they have been left alone. An earlier version of this
section described the problem as limited to code comments while it also
affected the two C++ READMEs; that is now true as written.
