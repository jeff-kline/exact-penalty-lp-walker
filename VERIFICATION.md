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

`-march=native` means the compiled walker is tuned to the build host. Timings
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

**Agrees.**

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
| NOT_MEASURED | 1 | `grow7` — no canonical fixture present |

This **disagreed** with the technical note as originally written, which
reported 4 certified and 19 numerical failures over 19.9–117.9 ms. The note was
corrected to match the code; see `CORRECTIONS.md`.

---

### 4. C++ walker build and panel

```sh
cd cpp/twalker
make build/verify_walker PYTHON=../../.venv/bin/python
./build/verify_walker fixtures_panel/*.twfx
```

Re-derives: the walker's status and original-data certificate on each panel
fixture, from scratch. This **does** run the solver.

Observed: builds with warnings only (deprecated Accelerate CBLAS prototypes),
no errors.

| Result | Count |
|---|---|
| Fixtures present | 26 (`grow7` absent, consistent with §5) |
| CERTIFIED | 24 |
| Rejected — "settle support cycle" | 2 (`brandy`, `lotfi`) |
| Worst certificate residual among certified | 6.99e-09 |
| Total wall time | 81.3 s |

Two qualifications on this run. First, it uses the walker's **default**
budgets; the note records that `fit1d` and `lotfi` needed authorized extended
budgets, so `lotfi` failing here is consistent with the note rather than
contrary to it. Second, the internal record behind the note reports `capri` as
a cycling model, whereas `capri` certified in this run and `lotfi` did not.
The identity of the cycling models is therefore **not** stable across
configurations, even though the count (24 certified) is. Claims about *which*
models fail should be read with that caveat.

The certified count of 24 out of 27 models matches the note. The worst
certificate residual, 6.99e-09, is inside the stated 1e-7 acceptance
tolerance.

---

### 6. Netlib walker panel, and how far it reproduces

```sh
.venv/bin/python experiments/bench_twalker_netlib_panel.py \
    --output records/twalker_cpp/netlib27_rerun.json
```

Re-derives: the walker's status, certificate, and timing on all 27 panel
models, from scratch, one `verify_walker` subprocess per model. This **does**
run the solver and needs the built binary.

Observed: 24 of 27 certified. Substituting these times into the frontier where
a walker variant won gives **9.712 s against the published 9.596 s, +1.2%.**

Determinism was measured, not assumed:

| Comparison | Structurally identical |
|---|---|
| Two independent per-model runs | **26 / 26** |
| Batched single invocation vs per-model | **26 / 26** |
| Current build vs the frozen 2026-08-16 record | **11 / 24** |

So the walker's path is deterministic on a fixed build. The divergence is
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

### 5. Deterministic document build

```sh
sh paper/build_twalker_progress_note.sh
```

Observed: 6 pages, exits 0, writes `output/pdf/twalker_progress_note.pdf`.

Determinism was checked by removing `tmp/texbuild/` entirely and rebuilding:

```text
19fa696ba28e08ecce68d50f61df50fcc4e4fdedb5819c5b15810ca95ad7659c  build 1
19fa696ba28e08ecce68d50f61df50fcc4e4fdedb5819c5b15810ca95ad7659c  build 2
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
  §6 below. An earlier version of this file said no script in the repository
  produced the Netlib walker record and that the timings were "not a
  measurement this repository can reproduce." That was too strong: the
  measuring apparatus (`verify_walker`) was always present, and only the loop
  around it was missing. It has been reconstructed and the frontier reproduces
  to 1.2%.
- **`experiments/bench_twalker_synth_nm.py`** — the synthetic timing producer
  is shipped but was not re-executed; the synthetic figures were rendered from
  the frozen summary.
- **The C++ unit tests** other than `verify_walker`
  (`verify_face_solver`, `verify_gram_solver`, `verify_bound_core_solver`) were
  not built or run.
- **Audit lanes** — no independent audit has been run against this tree.

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
4. `paper/twalker_progress_note.tex`: `\pdftrailerid{}` added; §5 counts and
   the Netlib figure caption corrected.
5. `experiments/pinar1997/render_provisional_netlib.py`: the figure footnote
   now computes the certified count from its input instead of printing a
   hardcoded literal.

6. `cpp/twalker/README.md`: six reproduction commands that hardcoded an
   absolute path inside the author's virtualenv now use `.venv/bin/python`.
7. `experiments/pinar1997/run_netlib_panel.py`: a missing-fixture message now
   reports a repository-relative path rather than an absolute one, and the one
   absolute path already frozen into `records/pinar1997/netlib_panel.json` was
   rewritten to match. No measurement was affected.

Several migrated files still cite internal report numbers (`agent_reports/11`,
`agent_reports/28`, and similar) that are not part of this release. These
dangling references are **not** confined to code comments: they also appear in
reader-facing prose in `cpp/twalker/README.md` and `cpp/twalker/revised/README.md`,
which additionally references `cpp/walk.cpp`, a file deliberately excluded from
this release. They have not been rewritten. An earlier version of this section
described the problem as limited to code comments; that was inaccurate.
