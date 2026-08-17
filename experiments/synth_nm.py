"""synth_nm: deterministic synthetic LPs in the n >> m regime.

WHY THIS FILE EXISTS
--------------------
Every benchmark in this repo runs on the 27-model Netlib panel, whose aspect
ratio in the repo's own ``min d'x s.t. Bx >= b`` encoding is narrow::

    degen2   1199 x  534   n/m = 2.24
    ship04s  2214 x 1458   n/m = 1.52
    fit1d    2077 x 1026   n/m = 2.02

The structural argument for this repo's method is that it solves the LP dual
through FACES whose Gram matrix is ``m x m``, so the per-step linear algebra is
``O(m^3)`` plus one pass over the data (``O(nnz)``), and is essentially
INDEPENDENT of the number of constraints ``n``.  Simplex, by contrast, carries a
basis whose size grows with the number of constraints.  ``n >> m`` should
therefore be the regime where the method is structurally advantaged -- and the
panel above cannot see it, because it never leaves ``n ~ 2m``.  This module
builds instances that do.

PROBLEM FORM
------------
The repo's canonical form (``exp11_netlib_rank_certificate.to_Bxgeb``)::

    min d'x   s.t.  B x >= b        B in R^{n x m}, x free in R^m
                                    n = # constraints, m = # variables

Mangasarian's Paper 1 (``assets/01_paper1_digest.md``) uses the OPPOSITE letters:
his ``A in R^{m x n}`` has ``m`` = # constraints and ``n`` = # variables, and his
target regime ``m >= 10n`` is exactly this file's ``n >= 10m``.  The generator
below is written in HIS letters internally (build ``A x <= p``) and converted at
the end by ``B = -A``, ``b = -p``, ``d = c``.  Every public name uses the REPO's
letters.

WHAT THE DIGEST ACTUALLY SPECIFIES (and what it does not)
---------------------------------------------------------
``assets/01_paper1_digest.md`` line 86 is the whole specification of Code 4.1
``lpgen`` that survives in this repo:

    A = 100(sprand(m,n,d) - 0.5*spones);  planted primal x with entries in
    [-10,10], ~half zeros; planted dual u >= 0 with ~3n positives (motivated by
    SVM support-vector counts); c, b back-derived.

That fixes: the matrix distribution (uniform on [-50, 50] on a sparse pattern of
density ``d``), the planted primal's range and its ~50% zero fraction, the
planted dual's support SIZE, and the back-derivation of ``c`` and ``b``.

It does NOT fix, and the MATLAB source is not in this repo, so the following are
CHOICES MADE HERE and are NOT to be attributed to Mangasarian:

  1. **The sparsity pattern law.**  MATLAB ``sprand`` scatters nonzeros i.i.d.,
     which leaves empty rows and columns at small ``d``.  This generator instead
     draws EXACTLY ``k = max(2, round(density*m))`` distinct columns per general
     row (uniformly without replacement).  Reason: an empty row is a vacuous
     constraint and an empty column makes ``x_j`` unconstrained, both of which
     would silently change what the sweep measures; fixed per-row ``k`` also
     makes the achieved density an exact, reportable function of the inputs.  A
     repair pass guarantees every COLUMN is hit at least once.  The achieved
     density is measured and reported on every instance, never assumed.
  2. **The magnitudes of the positive dual entries.**  Chosen ``U(0, 1]``.
  3. **The slacks of the inactive constraints.**  Chosen ``U(0, 1] * slack_scale``
     with ``slack_scale`` defaulting to ``max(1, median_i |(A x*)_i|)`` so the
     slack is on the scale of the row activities rather than an arbitrary
     absolute constant.  A fixed absolute slack would make every row nearly
     active once ``density*m`` grows.
  4. **The active-set size when ``3m`` does not fit.**  The digest's ``~3n``
     positives (= ``3m`` here) exceeds ``n`` at aspect ratio 2.  This generator
     uses ``n_active = min(3m, max(m+1, n//3))``: it keeps Mangasarian's ``3m``
     wherever it fits (aspect ratio >= 9) and otherwise takes a third of the
     rows, never fewer than ``m+1`` so that ``A_S`` generically has full column
     rank ``m`` and Mangasarian's strong-uniqueness condition (18) holds.
  5. **Box constraints.**  The digest has none.  They are added here because the
     task calls for them and because Netlib's ``to_Bxgeb`` always emits them
     (every finite variable bound becomes a unit row).  See BOX below.

GUARANTEED FEASIBILITY AND BOUNDEDNESS -- BY CONSTRUCTION, NOT BY REJECTION
--------------------------------------------------------------------------
Nothing is sampled and retried.  A primal-dual OPTIMAL PAIR is planted first and
the data is back-derived from it, so KKT holds identically:

    pick   x* in R^m,  u* >= 0 in R^n  with support S = {i : u*_i > 0}
    pick   A in R^{n x m}   (sparse general block + unit box rows)
    set    c := -A' u*                                    (dual equality)
    set    p := A x* + s   with   s_i = 0 for i in S,  s_i > 0 for i not in S

Then, for the program ``min c'x s.t. A x <= p``:

  * PRIMAL FEASIBILITY.  ``A x* - p = -s <= 0``.  So ``x*`` is feasible and the
    feasible set is nonempty.  No rejection sampling can fail.
  * DUAL FEASIBILITY.  ``A' u* + c = A' u* - A' u* = 0`` exactly, and ``u* >= 0``.
  * COMPLEMENTARY SLACKNESS.  ``u*_i (A x* - p)_i = -u*_i s_i = 0`` for every
    ``i``: either ``u*_i = 0`` (i not in S) or ``s_i = 0`` (i in S).
  * OPTIMALITY.  The three above are the KKT conditions of the LP, so ``x*`` is
    primal optimal and ``u*`` is dual optimal.
  * BOUNDEDNESS.  For ANY feasible ``x``: ``c'x = -(A'u*)'x = -u*'(A x) >=
    -u*'p`` because ``u* >= 0`` and ``A x <= p``.  So ``-p'u*`` is a finite lower
    bound on the objective over the whole feasible set.  The LP cannot be
    unbounded, whatever the sample.

Converting with ``B = -A``, ``b = -p``, ``d = c`` and ``y* = u*`` carries all four
properties to the repo's form, and the planted pair satisfies the FROZEN gate
``exp23_path_primal_dual.certificate_pair(B, b, d, x*, y*)`` to machine
precision::

    B x* - b = -(A x* - p) = s >= 0                       primal
    B' y* - d = -A' u* - c = 0                            dual equality
    y* = u* >= 0                                          sign
    d'x* = c'x* = -u*'A x* = -u*'p = b'y*                  zero gap

(the middle step uses ``u*_i (A x*)_i = u*_i p_i`` for every ``i``, which is
complementary slackness again).  ``--check`` runs that gate and reports the four
componentwise errors; it is the generator's self-test.

BOX
---
``--box-vars`` variables (default ``max(2, m//10)``, i.e. "a few") get a
two-sided box.  They are emitted as ORDINARY rows of ``A``: ``+e_j' x <= p``
and ``-e_j' x <= p'``.  Because ``p`` is back-derived from ``x*`` like every
other row, the box on variable ``j`` is ``x*_j - s_lo <= x_j <= x*_j + s_hi``
with ``s_lo, s_hi >= 0``, so it always contains ``x*_j`` and can be tight (in
``S``, with a positive multiplier) or slack.  No separate handling, no separate
feasibility argument: the guarantees above cover them because they ARE rows.
Each box row contributes exactly one nonzero, which is included in the reported
density.

DETERMINISM
-----------
``numpy.random.SeedSequence([seed, m, n, density_key])`` with
``density_key = round(density * 1e9)``, then ``default_rng`` of it.  A third
party reproduces any instance in the sweep from the four integers printed on its
row (``m``, ``n``, ``density``, ``seed``) and nothing else -- no global state, no
time, no PID, no ordering dependence between instances.  ``--check`` prints a
SHA-256 of ``(B, b, d)`` so reproduction can be verified byte for byte.

CLI
---
    # one instance, self-test it, print its census
    .venv/bin/python experiments/synth_nm.py \
        --m 100 --ratio 10 --density 0.10 --seed 20260810 --check

    # explicit n instead of a ratio, and save it
    ... --m 100 --n 2000 --density 0.05 --seed 20260810 --npz /tmp/inst.npz

    # the census for a whole sweep (no solving, no timing)
    ... --sweep --m-list 25 50 100 200 --ratios 2 5 10 50 100 \
        --density 0.10 --seed 20260810
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

# Mangasarian's lpgen constants (digest line 86), kept as named constants so a
# reader can see exactly which numbers are his.
MANG_A_HALFWIDTH = 50.0      # A entries ~ U[-50, 50] on the sparsity pattern
MANG_X_HALFWIDTH = 10.0      # planted primal entries ~ U[-10, 10]
MANG_X_ZERO_FRACTION = 0.5   # "~half zeros"
MANG_ACTIVE_PER_VAR = 3      # "~3n positives" with n = his #variables = our m

DEFAULT_SEED = 20260810


def _seeded_rng(m, n, density, seed):
    """Deterministic RNG keyed by the four numbers that identify an instance."""
    density_key = int(round(float(density) * 1e9))
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(m), int(n), density_key]))


def generate(m, n=None, ratio=None, density=0.10, seed=DEFAULT_SEED,
             box_vars=None, slack_scale=None):
    """Build one synthetic LP ``min d'x s.t. Bx >= b`` with a planted optimum.

    Parameters
    ----------
    m : int
        Number of VARIABLES (columns of ``B``).
    n : int, optional
        Number of CONSTRAINTS (rows of ``B``).  Give this or ``ratio``.
    ratio : float, optional
        ``n / m``.  The task's regime of interest requires ``ratio > 2``; the
        generator does not refuse smaller values but flags them.
    density : float
        TARGET density of the general block.  Each general row gets exactly
        ``k = max(2, round(density * m))`` nonzeros, so the general block's
        density is exactly ``k / m``; the achieved density of the FULL ``B``
        (which includes the one-nonzero box rows) is returned as
        ``census['density_achieved']``.
    seed : int
        Base seed.  ``(seed, m, n, density)`` determines the instance.
    box_vars : int, optional
        How many variables get a two-sided box.  Default ``max(2, m // 10)``,
        capped at ``m``.  Emits ``2 * box_vars`` unit rows.
    slack_scale : float, optional
        Scale of the inactive constraints' slacks.  Default (``None``) is
        ``max(1.0, median |(A x*)_i|)``, computed on the general block.

    Returns
    -------
    dict with keys ``B``, ``b``, ``d`` (the program), ``x_star``, ``y_star``
    (the planted optimal pair), ``active`` (the planted active set), and
    ``census`` (shapes, nnz, achieved density, and the parameters).
    """
    m = int(m)
    if m < 1:
        raise ValueError("m must be >= 1")
    if (n is None) == (ratio is None):
        raise ValueError("give exactly one of n or ratio")
    if n is None:
        n = int(round(float(ratio) * m))
    n = int(n)
    if box_vars is None:
        box_vars = max(2, m // 10)
    box_vars = int(min(max(0, box_vars), m))
    n_box = 2 * box_vars
    if n <= n_box:
        raise ValueError("n=%d leaves no general rows after %d box rows"
                         % (n, n_box))
    n_gen = n - n_box

    rng = _seeded_rng(m, n, density, seed)

    # ---------------------------------------------------- the general block
    # Exactly k distinct columns per row (see CHOICE 1 in the module docstring).
    k = int(max(2, min(m, round(float(density) * m))))
    A_gen = np.zeros((n_gen, m), dtype=float)
    for i in range(n_gen):
        cols = rng.choice(m, size=k, replace=False)
        A_gen[i, cols] = MANG_A_HALFWIDTH * (2.0 * rng.random(k) - 1.0)
    # Repair pass: no empty columns.  With n >> m and k >= 2 this fires almost
    # never, but "almost never" is not "never" and the guarantee is cheap.
    empty_cols = np.flatnonzero(~np.any(A_gen != 0.0, axis=0))
    for j in empty_cols:
        i = int(rng.integers(n_gen))
        A_gen[i, j] = MANG_A_HALFWIDTH * (2.0 * rng.random() - 1.0)

    # ------------------------------------------------------ the box rows
    if box_vars:
        boxed = np.sort(rng.choice(m, size=box_vars, replace=False))
    else:
        boxed = np.zeros(0, dtype=int)
    A_box = np.zeros((n_box, m), dtype=float)
    for t, j in enumerate(boxed):
        A_box[2 * t, j] = 1.0          # x_j <= upper
        A_box[2 * t + 1, j] = -1.0     # -x_j <= -lower

    A = np.vstack([A_gen, A_box]) if n_box else A_gen

    # --------------------------------------------- planted primal and dual
    x_star = MANG_X_HALFWIDTH * (2.0 * rng.random(m) - 1.0)
    zero_mask = rng.random(m) < MANG_X_ZERO_FRACTION
    x_star[zero_mask] = 0.0

    n_active = int(min(MANG_ACTIVE_PER_VAR * m, max(m + 1, n // 3)))
    n_active = int(min(n_active, n))
    active = np.sort(rng.choice(n, size=n_active, replace=False))
    u_star = np.zeros(n)
    # Strictly positive on S: U(0, 1] (CHOICE 2).  ``1 - random()`` is in (0, 1].
    u_star[active] = 1.0 - rng.random(n_active)

    # ---------------------------------------------- back-derived c and p
    c = -(A.T @ u_star)
    Ax = A @ x_star
    if slack_scale is None:
        activity = np.abs(A_gen @ x_star)
        slack_scale = float(max(1.0, np.median(activity) if activity.size
                                else 1.0))
    slack_scale = float(slack_scale)
    slack = slack_scale * (1.0 - rng.random(n))      # in (0, slack_scale]
    slack[active] = 0.0
    p = Ax + slack

    # --------------------------------------------- repo form: Bx >= b
    B = -A
    b = -p
    d = c
    y_star = u_star

    nnz = int(np.count_nonzero(B))
    census = {
        "m": m, "n": n, "ratio": float(n) / m,
        "n_general_rows": int(n_gen), "n_box_rows": int(n_box),
        "box_vars": int(box_vars),
        "nnz": nnz,
        "nnz_general": int(np.count_nonzero(A_gen)),
        "nnz_box": int(n_box),
        "density_target": float(density),
        "density_general_block": float(k) / m,
        "density_achieved": nnz / float(n * m),
        "nnz_per_general_row": k,
        "n_active_planted": int(n_active),
        "active_fraction": float(n_active) / n,
        "x_star_nonzeros": int(np.count_nonzero(x_star)),
        "slack_scale": slack_scale,
        "seed": int(seed),
        "aspect_ok_gt_2": bool(float(n) / m > 2.0),
        "planted_objective": float(d @ x_star),
        "empty_columns_repaired": int(empty_cols.size),
    }
    return {"B": B, "b": b, "d": d, "x_star": x_star, "y_star": y_star,
            "active": active, "census": census}


def digest(inst):
    """SHA-256 over ``(B, b, d)`` -- the reproduction fingerprint."""
    h = hashlib.sha256()
    for key in ("B", "b", "d"):
        a = np.ascontiguousarray(np.asarray(inst[key], dtype=float))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def check(inst):
    """Run the FROZEN certificate gate on the planted pair.

    This is the generator's self-test: if the planted pair does not certify on
    the generated ``(B, b, d)``, the construction is wrong and no benchmark run
    on it means anything.
    """
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from exp23_path_primal_dual import certificate_pair, TOL_FEAS
    ok, detail = certificate_pair(inst["B"], inst["b"], inst["d"],
                                  inst["x_star"], inst["y_star"])
    components = {k: detail[k] for k in
                  ("primal", "dual", "nonnegative", "gap") if k in detail}
    return {
        "planted_pair_certified": bool(ok),
        "reason": detail.get("reason"),
        "components": components,
        "worst": (max(components.values()) if components else None),
        "tol_feas": TOL_FEAS,
        "authority": ("frozen exp23_path_primal_dual.certificate_pair on the "
                      "generated (B, b, d), componentwise"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="deterministic synthetic LPs with n >> m, in the repo's "
                    "min d'x s.t. Bx >= b form, with a planted optimal pair")
    parser.add_argument("--m", type=int, default=100,
                        help="number of VARIABLES (columns of B)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--n", type=int, default=None,
                       help="number of CONSTRAINTS (rows of B)")
    group.add_argument("--ratio", type=float, default=None,
                       help="n/m; the regime of interest needs > 2")
    parser.add_argument("--density", type=float, default=0.10,
                        help="target density of the general block")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--box-vars", type=int, default=None,
                        help="variables given a two-sided box "
                             "(default max(2, m//10)); 2 rows each")
    parser.add_argument("--slack-scale", type=float, default=None)
    parser.add_argument("--check", action="store_true",
                        help="run the frozen certificate gate on the planted "
                             "pair and print the four componentwise errors")
    parser.add_argument("--npz", default=None,
                        help="save B, b, d, x_star, y_star to this .npz")
    parser.add_argument("--sweep", action="store_true",
                        help="print the census for a grid instead of one "
                             "instance")
    parser.add_argument("--m-list", type=int, nargs="*",
                        default=[25, 50, 100, 200])
    parser.add_argument("--ratios", type=float, nargs="*",
                        default=[2, 5, 10, 50, 100])
    args = parser.parse_args()

    if args.sweep:
        rows = []
        for m in args.m_list:
            for ratio in args.ratios:
                inst = generate(m, ratio=ratio, density=args.density,
                                seed=args.seed, box_vars=args.box_vars,
                                slack_scale=args.slack_scale)
                row = dict(inst["census"])
                if args.check:
                    row["check"] = check(inst)
                row["digest"] = digest(inst)[:16]
                rows.append(row)
                print(json.dumps(row, sort_keys=True))
        return

    ratio = args.ratio
    if args.n is None and ratio is None:
        ratio = 10.0
    inst = generate(args.m, n=args.n, ratio=ratio, density=args.density,
                    seed=args.seed, box_vars=args.box_vars,
                    slack_scale=args.slack_scale)
    out = dict(inst["census"])
    out["digest"] = digest(inst)
    if args.check:
        out["check"] = check(inst)
    if args.npz:
        np.savez(args.npz, B=inst["B"], b=inst["b"], d=inst["d"],
                 x_star=inst["x_star"], y_star=inst["y_star"],
                 active=inst["active"])
        out["npz"] = args.npz
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
