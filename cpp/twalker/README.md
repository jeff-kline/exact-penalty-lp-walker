# t-walker C++ rewrite

This tree is the correctness-first rewrite described in
`agent_reports/95_cpp_rewrite_handoff.md`.  It does not reuse the quarantined
normal-equations walker in `cpp/walk.cpp`.

The Python scripts under `tools/` only extract immutable oracle fixtures and
measure the sparse factorization structure.  They must be run with
`/Users/klinellc/.venvs/claude/bin/python`; `experiments/` remains unchanged.

Fixture export:

```sh
OMP_NUM_THREADS=1 /Users/klinellc/.venvs/claude/bin/python \
  cpp/twalker/tools/export_fixtures.py
```

Phase 1 direct-C++ fill/timing probe:

```sh
make -C cpp/twalker build/fill_probe
cpp/twalker/build/fill_probe \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion}.twfx \
  --out agent_reports/raw/twalker_phase1/fill_probe.csv
/Users/klinellc/.venvs/claude/bin/python \
  cpp/twalker/tools/summarize_fill.py \
  agent_reports/raw/twalker_phase1/fill_probe.csv \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion}.twfx \
  --out agent_reports/raw/twalker_phase1/summary.json
```

Build and run the correctness gates:

```sh
brew install osqp
make -C cpp/twalker all
cpp/twalker/build/verify_face_solver \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion,israel,boeing2,scagr7}.twfx
OMP_NUM_THREADS=1 cpp/twalker/build/verify_walker \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion,israel,boeing2,scagr7}.twfx
```

The default executable now constructs its own admissible seed at exactly
`t=0`.  It minimizes the fixed-`t` projection objective with an in-process
semismooth Newton method, reuses the full-rank Gram/core factor when that is
safe, and falls back to orthogonal range/null-space algebra on deficient
steps.  A rare maintained-QR NNLS support repair is available without calling
an external convex solver.  The resulting support is not trusted directly:
the ordinary fixed-`t` face settle and original-operator acceptance gates are
still authoritative.  The reported `wall_ms` starts before seed construction,
and the JSON separately reports `seed_ms`, `seed_route`, `seed_iterations`,
`seed_support`, and `seed_dres`.  For post-seed control experiments only,
`TWALKER_DISABLE_NATIVE_SEED=1` restores the support and `t0` stored in the
fixture.

Newton remains the default seed.  Native dense-kernel, implicit-triangular,
and HiGHS projection seeds are available as explicit opt-ins:

```sh
TWALKER_SEED=highs OMP_NUM_THREADS=1 cpp/twalker/build/verify_walker \
  cpp/twalker/fixtures/scorpion.twfx

TWALKER_SEED=kernel OMP_NUM_THREADS=1 cpp/twalker/build/verify_walker \
  cpp/twalker/fixtures_panel/afiro.twfx

TWALKER_SEED=triangular OMP_NUM_THREADS=1 cpp/twalker/build/verify_walker \
  cpp/twalker/fixtures_panel/capri.twfx
```

`TWALKER_SEED` accepts `newton` (the default), `kernel`, `triangular`, `highs`,
or `fixture`.
The `kernel` route is intended for low nullity `n-rank(B)`: it constructs an
explicit orthonormal basis of `ker(B.T)`, solves the identity-Hessian reduced
projection with a maintained active-row QR, and returns only a support
candidate.  The ordinary fixed-`t` settle and original-data gates remain
authoritative.  It does not support an experimental nonzero target nudge and
fails closed in that case.
The `triangular` route represents the same Euclidean projection without
forming the dense null basis.  A column-equilibrated pivoted QR of `B` retains
only its triangular factor and applies
`P=I-B(B.T*B)^-1*B.T` through sparse products, triangular solves, and
residual-gated iterative refinement.  Bound events update a triangular factor
of the active principal submatrix of `P`.  One final direct support-face solve
canonicalizes a degenerate `t=0` seed before the original-operator KKT gate.
The route currently requires full column rank and is an opt-in experiment;
rank-deficient inputs fail closed.
The HiGHS route solves the fixed-`t=0` projection QP
$\min \tfrac12\lVert y\rVert^2-s^\top y$ subject to
$B^\top y=d, y\geq0$, where $s$ is the optional target nudge.  It retains
positive dual variables and any structural basic-zero variables returned by
HiGHS as the initial working support.  The unchanged fixed-`t` settle and
original-operator acceptance gates must still admit the seed before walking;
HiGHS status is never trusted directly.  If HiGHS returns a finite but rough
QP point, its support may nominate a face, but that face must be repaired and
accepted at unchanged `t=0`; this is reported as
`seed_route="highs-qp-candidate+fixed-t-repair"`.  A directly accepted seed is
reported as `highs_seed=true` and `seed_route="highs-qp"`.

`verify_walker` links the installed HiGHS library for maintained repair
systems: the endpoint selector, terminal primal-face recovery, and a rare
fixed-support terminal dual rebalance.  The last route is reached only after a
complete forward event scan and both native and primal-only certificates fail;
it keeps `t` and the current support fixed, retains the returned simplex basis,
and maximizes the dual objective on that face.  HiGHS candidates are never
trusted directly: every accepted pair is checked componentwise on the original
`B,b,d`.

The production `verify_walker` target also links the maintained revised-column
solver.  The promoted cost-aware factored seed, bounded 16-epoch rebase, SVD
row-space representation, below-heuristic residual audit, decision-unguarded
cheap lane, and fail-closed restart perimeter are enabled by default.  No
positive environment bundle is required.  The relevant diagnostic opt-outs
are:

```text
TWALKER_DISABLE_REVISED_COLUMN=1
TWALKER_DISABLE_REVISED_FACTORED_SEED=1
TWALKER_DISABLE_REVISED_MULTI_EPOCH=1
TWALKER_REVISED_GUARDED=1
TWALKER_DISABLE_REVISED_COST_AWARE_SEED=1
TWALKER_DISABLE_REVISED_SVD_ROWSPACE=1
TWALKER_DISABLE_REVISED_SVD_AUDIT_BELOW_GATE=1
TWALKER_DISABLE_SPECULATIVE_RESTART=1
```

`TWALKER_REVISED_MAX_SEEDS` can override the default cap of 16.  A failed walk
is retried conservatively only if at least one revised face answer influenced
the path; otherwise a retry would reproduce the same direct decisions.  The
terminal original-data certificate remains authoritative.

The default solver uses the full-column-rank row-update lane when the program
has at least 500 equality columns.  It keeps a fixed-permutation triangular factor,
packed exact fill pattern, and `A' b`; rank-deficient or decision-ambiguous
faces fail closed to the ordinary SPQR/RZ/SVD route.  The 500-column boundary
keeps smaller tie-heavy walks from spending the savings on
conservative stability rechecks.  `TWALKER_DISABLE_QR_UPDATE_LIVE=1`
disables the default, while `TWALKER_QR_UPDATE_LIVE=1` explicitly enables the
lane below the default size gate.  Audit-only comparison against
the direct oracle uses `TWALKER_QR_UPDATE_AUDIT=1`; the two modes are mutually
exclusive.
Audit mode also reports whether each accurate updated face produces the same
settle support transition and complete event tie set as the authoritative
direct answer, plus time-weighted oracle coverage and numerical-rank changes.
It never returns the audited candidate. See
`agent_reports/125_qr_event_equivalence.md`.
For bounded structural routing experiments,
`TWALKER_QR_UPDATE_MIN_COLUMNS=<m>` overrides the default size boundary; it
does not affect audit mode or the direct solver.

`TWALKER_QR_UPDATE_LOCAL_REPIVOT=1` is a rejected, diagnostic-only experiment.
On the first 1,000 Fit1d pivots it recovered only 2 additional updated faces
after 137 solve-bound declines and cost more than the SPQR work it displaced.
The failure is not predominantly local ordering: the retained `R` and `A'b`
lose the orthogonal action needed to certify the minimum-norm affine tail after
many normal-equation updates.  The switch is off by default.

Degenerate basic rows are kept in that square factor even when their path
value is numerically zero.  If both the accepted current value and computed
derivative are at roundoff scale, the default event scan assigns the row a
sticky basic-zero status instead of repeatedly rebuilding the same face to
decide its sign.  This is basis bookkeeping only: the row is not pruned, the
event winner is unchanged, and all ordinary face and original-data
certificate gates remain in force.  Set
`TWALKER_DISABLE_STICKY_BASIC_ZERO=1` to restore the conservative refactor on
each such row.

The rare medium-rectangular projection repair links Homebrew's standalone
OSQP C library from `/usr/local/opt/osqp`.  `OSQP_PREFIX` can override that
build-time path.  This route is a repair-only fallback, not part of ordinary
face pivots.  OSQP is used only to propose a support which must pass the same
exact face settle and original-data certificate.  The native defaults therefore
target support recovery rather than full QP convergence: `rho=1`, 2,000
iterations, and at most one accepted QP repair per walk.  The audit overrides
`TWALKER_QP_RHO`, `TWALKER_QP_MAX_ITER`, and
`TWALKER_QP_ALLOW_MULTIPLE_REPAIRS` reproduce alternative settings.

The ordinary fast lane maintains a CHOLMOD factor of `B_W' B_W` with sparse
rank-one row updates, plus `B_W' b_W`, the factor permutation, condition
metadata, both affine solutions, their full sparse products, and the active
row list.  It is admitted only on full-rank faces with `rcond >= 1e-6` and
original-operator residual checks.  Faces below `5e-4` receive long-double
original-operator refinement and explicit coefficient-error bounds.  The
walker propagates those bounds through settle, terminal, and breakpoint
decisions; ambiguous decisions are recomputed by direct SPQR.  After the
extension band has actually been used, a multi-row tied event is recomputed
directly and retires the maintained factor for the remaining degenerate path.
Rank-deficient or more poorly conditioned faces fail closed to the direct
SPQR/RZ solver.  Set `TWALKER_GRAM_MIN_RCOND=5e-4` for the conservative
pre-extension A/B route.

The Fit1d bound/core reduction is a quarantined component experiment.  For an
active operator made of one-entry bound rows plus at most 48 general rows, it
writes `B_W'B_W = D + C'C`, eliminates the positive diagonal part of `D`, and
factors only a bordered dense system of order
`active_core_rows + zero_diagonal_entries`.  It is not enabled by default.
`TWALKER_BOUND_CORE_AUDIT=1` compares candidates with the authoritative Gram
answer without changing the walk; `TWALKER_BOUND_CORE_LIVE=1` enables the
fail-closed experimental route.  Numerical rank is gated by a reciprocal
condition estimate and every returned face is checked on the original
operator.  The bounded admission and rejected wider variant are recorded in
`agent_reports/144_fit1d_bound_core_hinge.md`.

The wider rank-revealing variant is also quarantined.  Set
`TWALKER_BOUND_CORE_WIDE_SHADOW=1` with the audit flag to use an equilibrated
SVD on borders with up to 32 missing diagonal coordinates while leaving the
authoritative walk unchanged.  A full Fit1d shadow run compared 805 answers
with zero `1e-10` audit violations, but live use skipped one numerically
delicate event at pivot 1,168.  `TWALKER_BOUND_CORE_WIDE_LIVE=1` and
`TWALKER_BOUND_CORE_UNGUARDED=1` exist only to reproduce that rejected A/B;
they are not production settings.  Details are in
`agent_reports/145_fit1d_terminal_tail.md`.

Terminal detection is intentionally split.  A small active slope first tries
only the cheap original-data certificate; if inactive rows still expose a
forward event, the walk continues without invoking recovery.  Expensive
certificate recovery is allowed only after the full ratio scan proves that no
event lies inside the forward horizon.  This removed all recovery calls from
the fresh 25-certifier default panel.  Approximate structured lanes are never
allowed to declare the path terminal without one direct rank-revealing polish.

The direct lane memoizes up to 2,048 complete face artifacts by the exact
sorted active-row vector.  The fixture data and affine right-hand sides are
immutable, so an identical support has an identical answer; the hash lookup
also compares the full vector.  Set `TWALKER_DISABLE_FACE_CACHE=1` when running
`verify_walker` to force a numerical refactor on every logical face for A/B
measurement.  Native cumulative phase counters in its JSON separate assembly,
SPQR, RZ, triangular/COD application, products, residual checks, dense SVD,
and cache work.

After a batch settle reaches its round cap, recovery tries the existing
one-row `1e-10` perturbation before replaying the five batch perturbations.
If the early attempt fails, the original batch/QP/serial chain still runs in
full; successful faces pass the same settle and certificate gates.  Set
`TWALKER_DISABLE_SERIAL_FIRST_AFTER_ROUND_CAP=1` for the former ordering.

Endpoint checks form affine combinations with fused multiply-adds and test
`t(g-b)+(h-s)` directly rather than subtracting `t*b+s` from an already formed
`y(t)`.  A combined endpoint equality residual may use a `2e-7` intermediate
envelope only when the underlying slope identity is at most `1e-8`; all other
face gates retain their `1e-7` limit and every returned LP pair still passes
the unchanged componentwise original-data certificate.  This repaired the
clean-build `share1b` false rejection; `TWALKER_REPAIR_TRACE=1` reports any
use of the narrow envelope.

Maintained-factor oracle gate:

```sh
cpp/twalker/build/verify_gram_solver \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion,israel,boeing2,scagr7}.twfx
cpp/twalker/build/verify_gram_solver \
  cpp/twalker/fixtures_degen2/degen2.twfx
```

The measured implementation and walker are entirely native C++.  The Python
fixture exporter is an offline SVD-oracle freezer and is not linked or invoked
by `verify_walker`.

The bounded exact-equality quotient probe is likewise offline preprocessing:

```sh
/Users/klinellc/.venvs/claude/bin/python \
  cpp/twalker/tools/export_quotient_fixture.py --model sctap1 --faces 60
/Users/klinellc/.venvs/claude/bin/python \
  cpp/twalker/tools/certify_quotient_walker.py \
  cpp/twalker/fixtures_quotient/sctap1_quotient.twfx \
  cpp/twalker/fixtures_quotient/sctap1_quotient_lift.npz
```

The timed solve in the second command is the C++ executable; Python performs
only the exact lift and frozen certificate on the untouched original model.
This probe is not a default router: its maintained lane loses conditioning
after an initial fast segment on `sctap1`, as recorded in
`agent_reports/99_maintained_cod_execution.md`.

Additional design probes:

```sh
cpp/twalker/build/selector_probe \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion}.twfx
cpp/twalker/build/symbolic_probe \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion}.twfx
cpp/twalker/build/simplex_probe \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion,israel,boeing2,scagr7}.twfx
```

The first measures retained-basis face feasibility, the second measures the
best symbolic reuse exposed by SuiteSparseQR, and the third provides a native
HiGHS simplex-iteration baseline on the same model data.

Cheap panel coverage (one accepted post-seed oracle face per model, continuing
past parser or seed failures):

```sh
OMP_NUM_THREADS=1 /Users/klinellc/.venvs/claude/bin/python \
  cpp/twalker/tools/export_fixtures.py \
  --initial-only --continue-on-error --faces 1 \
  --outdir cpp/twalker/fixtures_panel \
  --models adlittle afiro bandm beaconfd blend boeing2 brandy capri degen2 \
  e226 fit1d forplan grow7 israel kb2 lotfi recipe sc105 sc205 sc50a sc50b \
  scagr7 scorpion sctap1 share1b share2b ship04s stocfor1
```

This panel mode is a coverage screen, not a replacement for the multi-face SVD
oracle gate.  A C++ `CERTIFIED` result is still recomputed on original data;
seed-only models do not provide pivot-speed evidence.
