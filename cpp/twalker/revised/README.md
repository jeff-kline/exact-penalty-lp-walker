# Revised-basis probes and maintained revised-column t-walker

The row-basis probe in this directory is intentionally disconnected from
`Walker`.  The revised-column solver is different: its admitted
small-deficiency route is linked into the default production walker.

For an active face `A=B[W,:]`, the probe retains independent active rows `C`
and active-row coordinates `L` such that `A=L C`.  It factors `C C'` and
`L' L`.  Adding or removing a dependent support row changes only `L' L` by
rank one.  When a retained basis row leaves, a dependent active row is
promoted by a coordinate pivot.  Only a true rank change invokes the
rank-revealing rebuild.

The minimum-norm face artifacts follow from

```text
ua = -C' (C C')^-1 (L' L)^-1 L'b
g  =  b - L (L' L)^-1 L'b

h  = s + L (L'L)^-1 (C C')^-1 C(d-A's)
uc = C' (C C')^-1 (L'L)^-1 (C C')^-1 C(d-A's).
```

Build and replay frozen oracle faces explicitly:

```sh
make -C cpp/twalker revised-probe
cpp/twalker/build/revised_basis_probe \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion,israel,boeing2,scagr7}.twfx
```

This is not an answer path.  Oracle agreement, transition coverage, and cost
must be established before any guarded integration is considered.

## Column-basis alternative

`revised-column-probe` retains independent columns `C` and `T` with `A=C*T`.
Unlike the row-basis experiment, an ordinary support-row pivot does not change
`T`; it applies one rank-one update to `C'C`.  The existing sparse QR supplies
the initial column permutation, so a second factorization of `A'` is avoided.

```sh
make -C cpp/twalker revised-column-probe
cpp/twalker/build/revised_column_probe \
  cpp/twalker/fixtures/{sctap1,brandy,scorpion,israel,boeing2,scagr7}.twfx
```

The promoted factored-seed route is enabled by default; use
`TWALKER_DISABLE_REVISED_FACTORED_SEED=1` only for an audit comparison.  It
consumes the incumbent direct solver's
accepted SPQR `R` and column permutation, retains actual sparse independent
columns `C`, and maintains a CHOLMOD factor of `C'C`.  It does not materialize
`A+` and does not retain the dense active-by-rank coordinate table.

For `T=[I,D]`, applying the minimum-norm map uses Woodbury:

```text
(T T')^-1 z = z - D (I + D'D)^-1 D'z.
```

Thus reconstruction costs `O(rank * deficiency)` instead of `O(m * rank)`;
the full-rank case is only a permutation/copy.  A leaving row's coordinates
are recovered directly from the sparse fixture, so add/drop maintenance no
longer shifts an `active * rank` dense array.  Every returned face is checked
against the original operators.  The normal residual gate is `1e-11`; a face
that required and completed the marginal-factor correction is allowed
`1e-10`.  The frozen 26-model seed-face panel admits 17 models with no oracle
false accepts.

Build and run the current probe with:

```sh
make -C cpp/twalker revised-column-probe
cpp/twalker/build/revised_column_probe cpp/twalker/fixtures_panel/*.twfx
```

On the maintained 60-face sequences, three-run medians were `17.6 us`
(`scagr7`), `45.1 us` (`scorpion`), and `100.6 us` (`sctap1`), respectively
about `1.2x`, `2.9x`, and `4.7x` the measured HiGHS simplex pivot costs.
The production walker and `make all` link this route.  The decision-unguarded
cheap lane is the default inside the original-data certification and
fail-closed restart perimeter; `TWALKER_REVISED_GUARDED=1` restores the former
decision interval for an audit comparison.

An additional rejected experiment can be enabled with
`TWALKER_REVISED_PERSIST_RANK=1`.  It maintains rank increases and decreases
using an explicit row-space tableau and Cholesky deletion/update operations.
It is intentionally off by default: full evolving-`t` tests found that normal
equations lose update accuracy on the large deficient faces and trigger
BLAS-core refresh/rebuild churn.  See the report for the measured result.

The follow-up square-root version reuses the same flag.  Its cold `C` and `T`
artifacts are now constructed by dense QR/least squares, and rank expansion is
a direct Givens update.  `TWALKER_REVISED_PRIMARY=1` places it ahead of Gram for
an alternative-architecture panel.  A 19-program majority sweep rejected both
the persistent-rank fallback and primary routing, so those two flags remain
off by default; this is distinct from the promoted bounded epoch route.

`TWALKER_REVISED_RECURRENCE=1` enables the later coefficient-recurrence probe.
It maintains the dense Moore--Penrose inverse `P=A+` with Greville row
addition/deletion formulas and updates the affine face coefficients directly.
This removes all triangular solves from an ordinary pivot.  It is materially
faster than direct SPQR on admitted stored faces, but a fresh guarded
19-program sweep was 3.3% slower overall because cold initialization/rebase
and decision guards outweighed the 60 faces it served.  It therefore remains a
component result, not a live routing option.  `TWALKER_REVISED_SVD_INIT=1`
selects a rejected, slower SVD cold initializer for diagnostic comparison.

`TWALKER_REVISED_SHARE_DIRECT_SEED=1` enables the final rejected initializer
experiment.  It reuses the incumbent direct face's SPQR/COD state, allows one
bounded reseed after a wholesale support repair, and never runs a revised-side
cold factorization.  This proved that duplicate QR can be removed, but forming
the dense pseudoinverse through `Q' I` costs as much as the direct solves it
replaces.  Five interleaved `sctap1` runs were about 8% slower in full-walk
median despite identical pivots, terminal `t`, and 53 recurrence-served faces.
The flag remains off; a future attempt should retain factored `R`/RZ actions
instead of materializing `A+`.

## Small-deficiency multi-epoch route

Multi-epoch rebasing, together with the factored seed lane, allows
an already-required accepted direct face to begin another reduced-factor epoch.
The default cap is 16; `TWALKER_REVISED_MAX_SEEDS` overrides it and
`TWALKER_DISABLE_REVISED_MULTI_EPOCH=1` disables rebasing.  Additional epochs are admitted
only after at least one revised return and only while every observed direct
seed has rank deficiency at most four.  This prevents unproductive reseed
churn on deeply deficient walks.

On `e226`, the decision-unguarded research route served 267/311 faces and cut
dense SVD fallbacks from 69 to 32, producing a repeatable roughly 2x wall
improvement.  A paired 19-certifier sweep preserved every pivot count and
terminal `t` and reduced summed wall by 18.1%.  The guarded route remains
slower because its blanket coefficient interval triggers a direct stability
refactor for every revised face.  The later cost-aware/SVD work below supplied
the original-data certificate and fail-closed perimeter used to promote this
bounded route.

## Cost-aware singularity rebase and certified restart

The default cost-aware seed policy changes singularity handling from permanent
retirement to a bounded epoch transition.  Before the first expensive event,
the revised solver stays dormant.  When the incumbent direct solver already
needs a dense SVD on a face with rank deficiency at most four, that SVD remains
authoritative for the current face.  If the independently computed SPQR rank
agrees, its QR/RZ artifacts seed the next reduced-factor epoch.  Thus the
repair neither changes the current answer nor performs another rank-revealing
factorization merely to initialize the cheap lane.

The intended configuration is the factored-seed, multi-epoch route with a
finite `TWALKER_REVISED_MAX_SEEDS`.  It is cost-aware rather than model-aware:
no model name, static scale threshold, or objective perturbation enters the
router.  On a fresh 19-certifier sweep it activated only on `e226` and
`israel`, certified 19/19, reduced dense SVD fallbacks from 439 to 375, and
measured 3.925 s versus an adjacent 4.766 s conservative sweep.  `e226`
accounted for most of the gain, with 152 revised faces and 17 dense SVDs rather
than 69.

The default `verify_walker` executable includes the fail-closed perimeter:
after the complete evolving-`t` walk, the solution is
certified against the original fixture.  Any non-`CERTIFIED` result is rerun
from the same seed in a fresh `Walker` with the revised branch disabled if a
revised answer influenced the failed path.  The reported wall time includes
both attempts and `speculative_restarts` reports whether this occurred.
`TWALKER_DISABLE_SPECULATIVE_RESTART=1` is the diagnostic opt-out;
`TWALKER_FORCE_SPECULATIVE_RESTART=1` exists only to exercise recovery in tests.

This route is promoted because output certification plus restart protects the
returned LP solution.  Promotion does not claim that every intermediate event
decision matches the conservative path.

## SVD row-space rebase

The default SVD row-space route retains the accepted right-singular row space
when a dense fallback was already required.  It forms orthonormal row
coordinates, initializes their triangular factor by QR, and maintains it by
row update/downdate.  This avoids reconstructing deficient columns through
`R11^-1 R`, which direct audit identified as the dominant forward-error source
on `e226`.

The SVD representation is checked immediately against the original face
operators.  If it fails—as it does on `israel` with a `2.61e-10` piece
residual—the same direct face seeds the independent-column representation
instead.  Rank changes and residual failures become bounded rebase points.
The original-operator residual audit is allowed below the old diagonal
heuristic because the useful `e226` coordinate factor lies there.  Set
`TWALKER_DISABLE_REVISED_SVD_AUDIT_BELOW_GATE=1` to restore the heuristic-only
comparison.

In a fresh adjacent 19-certifier comparison, the hybrid certified 19/19 with
zero restarts, reduced dense SVDs from 439 to 362, and measured 4.330 s versus
5.516 s.  `e226` fell from 69 SVDs to 4 and about 2.04 s to 0.60 s; `israel`
served 12 independent-column faces.  `e226` retained 303 pivots but terminal
`t` differed from conservative by `2.36e-7`, so the promoted route remains
inside original-data certification and the fail-closed restart perimeter.
