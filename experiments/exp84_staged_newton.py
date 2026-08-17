"""exp84: the SHIPPED single-model entry point for the native lane.

One Netlib model per run; prints a single JSON record.  Shaped like exp81's so
the two panels are directly comparable, plus the staged lane's own counters:
per-stage screening fractions, terminal tests run vs skipped, and the wall
split between the Newton solves, the screening passes and the frozen terminal
tests.

No external LP solver is called on the measured path.  ``--reference`` (a
HiGHS objective cross-check) is opt-in, off by default, and runs OUTSIDE the
timed region -- the claim under test is self-containment.

============================================================================
THE PROMOTED DEFAULTS  (agent_reports/22, 24, 25, 26, 27, 28)

The configuration that measured 12.219 s on the 27-model panel (11.760-13.647,
5 repeats, 27/27 CERTIFIED, worst frozen componentwise error 7.6206e-09) is now
what this entry point RUNS.  Until this promotion it existed only as runtime
installs inside benchmark harnesses; shipped defaults measured 22.321 s.  Four
independently measured pieces, each restorable by an explicit flag:

  1. ``--route qr_cod``      terminal_qr_cod.qr_cod_certificate_detail is
     (default)               installed as exp23._certificate_detail for the
                             duration of the solve.  -13.9 % panel, 486 runs.
                             ``--route frozen`` restores the incumbent.
  2. c1_const_cache          woodbury_face_solver.TIK_CONST_CACHE and
     c5_pchol_screen         PCHOL_SCREEN = 8.  -3.233 s panel, 405 runs.
                             ``--no-candidates`` restores both.
  3. Ruiz equilibration      power-of-two row/column balancing of (B, b, d);
     (default on)            the pair is unscaled BEFORE certification.
                             -17.8 % panel.  ``--no-equilibrate`` restores.
  4. a1/a2/a3/b1             allocation patches in woodbury_face_solver and
     (default on)            newton_oracle.  -6.7 % (replicated -8.8/-7.7).
                             ``--no-alloc-patches`` restores.

WHAT COUNTS AS CERTIFIED IS UNCHANGED.  ``exp23_path_primal_dual.
certificate_pair`` on the ORIGINAL raw (B, b, d), componentwise, TOL_FEAS =
1e-7, is the only thing that may declare CERTIFIED.  Equilibration transforms
the program, so the pipeline is: equilibrate -> solve the scaled program ->
unscale the pair (x = c*x', y = r*y') -> certify on the ORIGINAL raw data.  The
certificate the solver reached in the scaled space is recorded as
``transformed_status`` and never sets the verdict.  Power-of-two factors make
the round trip bit-exact; the round-trip assertions are kept and recorded.

THE CERTIFICATE ROUTE IS AN EXPLICIT, REVERSIBLE INSTALL, NOT AN IMPORT SIDE
EFFECT.  ``install_certificate_route`` is a context manager: it rebinds
``exp23._certificate_detail``, records what it installed, and restores the
captured original in a ``finally`` on every exit path, including exceptions.
Importing this module installs nothing.  It also YIELDS to a route a caller has
already installed (``bench_certificate_routes`` and the historical exp1xx
panels install their own arm around ``run``), so no harness silently loses its
arm.  ``exp23_path_primal_dual.py`` itself is NOT edited by any of this.

Run:
  OMP_NUM_THREADS=1 ... python exp84_staged_newton.py lotfi --seconds 300
  ... exp84_staged_newton.py lotfi --route frozen --no-equilibrate \\
        --no-candidates --no-alloc-patches      # the pre-promotion defaults
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp11_netlib_rank_certificate import to_Bxgeb
import exp23_path_primal_dual as exp23
import newton_oracle as newt
import oracle_terminal as ot
import staged_newton as sn
import woodbury_face_solver as wfs
# Imported for the default certificate route.  ``terminal_qr_cod`` imports
# ``hybrid_certificate_factor``, which captures ``exp23._certificate_detail``
# into its own ``_ORIGINAL`` AT IMPORT TIME -- so it must be imported while the
# incumbent is still installed, which is the case here because NOTHING is
# installed at import time.
import terminal_qr_cod as qr_cod                     # noqa: E402
import hybrid_certificate_factor as hybrid           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The frozen incumbent, captured before any install can happen.
FROZEN_CERTIFICATE_DETAIL = exp23._certificate_detail
FROZEN_CERTIFICATE_PAIR = exp23.certificate_pair

# Default Newton-step kernel for this harness.  "fast" is the pre-S1/S2 lane;
# "fast+tik+pchol" adds the Tikhonov-Woodbury route (behind its unit-row
# fraction gate, so it installs on exactly the unit-heavy models) and the
# pivoted-Cholesky route (m <= 700, screen-and-retire with hysteresis).  Both
# are held to the same honest step residual as every other fast route.
DEFAULT_KERNEL = "fast+tik+pchol"

# ---------------------------------------------------------------- promoted
ROUTES = ("qr_cod", "frozen", "hybrid")
DEFAULT_ROUTE = "qr_cod"
DEFAULT_EQUILIBRATE = True
DEFAULT_RUIZ_ROUNDS = 10
# The kernel-side promotions, as (module, attribute, promoted, pre-promotion).
# ``promoted_kernel`` sets them; ``quarantined_speedups`` /
# ``quarantined_pivchol_alloc`` set the same attributes for their A/B arms.
CANDIDATE_FLAGS = (
    (wfs, "TIK_CONST_CACHE", True, False),                 # c1_const_cache
    (wfs, "PCHOL_SCREEN", 8, wfs.PCHOL_SCREEN_PRE_PROMOTION),   # c5
)
ALLOC_FLAGS = (
    (wfs, "ALLOC_TRIL_SLICE", True, False),                # a1_tril_slice
    (wfs, "ALLOC_U_GATHER", True, False),                  # a2_u_gather
    (wfs, "ALLOC_DIAG_SCALAR", True, False),               # a3_diag_scalar
    (newt, "CHURN_MASK", True, False),                     # b1_churn_mask
)

# Expected certified lower bounds, for the 1e-9 relative match gate (exp81's).
# Equilibration does not move them: the dual objective is preserved EXACTLY
# (b' y = (Rb)' (R^-1 y) term by term, and every factor is a power of two).
KNOWN_LB = {
    "grow7": -47787811.81471147,
    "lotfi": -25.264706061882478,
    "brandy": 1518.5098964881315,
}


# ==========================================================================
# THE CERTIFICATE ROUTE -- explicit install, restored in ``finally``
# ==========================================================================

def route_function(name):
    if name in (None, "frozen"):
        return FROZEN_CERTIFICATE_DETAIL
    if name == "qr_cod":
        return qr_cod.qr_cod_certificate_detail
    if name == "hybrid":
        return hybrid.single_factor_certificate_detail
    raise ValueError("unknown certificate route %r (known: %s)"
                     % (name, ", ".join(ROUTES)))


@contextlib.contextmanager
def install_certificate_route(name, record=None):
    """Rebind ``exp23._certificate_detail`` for the duration of the block.

    Explicit and reversible by construction: the current binding is captured,
    the replacement is assigned, and the capture is restored in ``finally`` on
    every exit path.  ``exp23_path_primal_dual.py`` is never edited.

    THREE THINGS THIS DELIBERATELY DOES NOT DO.

    * It does not install anything at import time.  Importing this module
      leaves ``exp23._certificate_detail`` exactly as it found it.
    * It YIELDS to a caller-installed route.  If ``exp23._certificate_detail``
      is no longer the frozen incumbent when the block is entered, some caller
      (``bench_certificate_routes._run_one``, the exp1xx panels) owns the
      binding and this is a no-op -- their arm is theirs.  ``name=None`` asks
      for the same no-op unconditionally.
    * It does not relax any gate.  ``_certificate_detail`` only PROPOSES a
      primal ``x`` for a dual ``y``.  Both replacement routes return ``True``
      only after ``exp23.certificate_pair`` has passed, and the entry point
      re-runs the frozen pair on the ORIGINAL raw data after unscaling.

    ``record``, when given, receives what actually happened.
    """
    current = exp23._certificate_detail
    yielded = (name is None
               or current is not FROZEN_CERTIFICATE_DETAIL
               or name == "frozen")
    target = current if yielded else route_function(name)
    if record is not None:
        record.update({
            "requested": name,
            "installed": getattr(target, "__qualname__", str(target)),
            "installed_module": getattr(target, "__module__", None),
            "yielded_to_caller": bool(
                current is not FROZEN_CERTIFICATE_DETAIL),
            "is_frozen_incumbent": target is FROZEN_CERTIFICATE_DETAIL,
        })
    if target is current:
        yield target
        return
    exp23._certificate_detail = target
    try:
        yield target
    finally:
        exp23._certificate_detail = current


@contextlib.contextmanager
def promoted_kernel(candidates=True, alloc=True):
    """Set the promoted kernel flags; restore the previous values in ``finally``.

    ``candidates=False`` restores c1_const_cache / c5_pchol_screen to their
    pre-promotion values; ``alloc=False`` does the same for a1/a2/a3/b1.
    ``None`` for either group LEAVES THOSE FLAGS ALONE, which is how a harness
    that installed its own arm through ``quarantined_speedups.active`` or
    ``quarantined_pivchol_alloc.active`` keeps it.
    """
    entries = []
    if candidates is not None:
        entries += [(mod, attr, (on if candidates else off))
                    for mod, attr, on, off in CANDIDATE_FLAGS]
    if alloc is not None:
        entries += [(mod, attr, (on if alloc else off))
                    for mod, attr, on, off in ALLOC_FLAGS]
    saved = [(mod, attr, getattr(mod, attr)) for mod, attr, _ in entries]
    try:
        for mod, attr, value in entries:
            setattr(mod, attr, value)
        yield
    finally:
        for mod, attr, value in saved:
            setattr(mod, attr, value)


def kernel_state():
    """What is live RIGHT NOW, read off the modules -- not off intent."""
    return {attr: getattr(mod, attr)
            for mod, attr, _on, _off in CANDIDATE_FLAGS + ALLOC_FLAGS}


# ==========================================================================
# EQUILIBRATION -- Ruiz row/column balancing with power-of-two factors
# ==========================================================================
#
# Moved here from ``bench_transforms``, which measured it (-17.8 % panel) and
# now imports it back so there is ONE implementation and the harness keeps
# meaning what it meant.  With ``R = diag(r) > 0`` (rows) and ``C = diag(c) > 0``
# (columns):
#
#     B' = R B C        b' = R b        d' = C d
#     recover  x = C x'          y = R y'
#
# The four identities that make this a legitimate reformulation, each verified
# by algebra AND asserted numerically at runtime (``equil_identities``):
#
#   1. FEASIBILITY.   B'x' - b' = R (Bx - b) componentwise, R > 0, so the
#      feasible sets correspond under x = C x'.
#   2. DUAL EQUALITY. B''y' = C (B'y) with y = R y', and d' = C d, so
#      B''y' = d' iff B'y = d  (C > 0 diagonal, hence invertible).
#   3. SIGN.          y' >= 0 iff R y' >= 0 iff y >= 0  (R > 0).
#   4. OBJECTIVE.     (Cd)'x' = d'(C x') = d'x, and (Rb)'y' = b'(R y') = b'y,
#      so the duality gap is preserved exactly.
#
# Only the MATRIX is equilibrated; ``b`` and ``d`` are then scaled by the
# factors the matrix balancing produced.  No separate cost or right-hand-side
# scaling is applied -- that would be a second, untested knob.

def _pow2(values, max_exp=500):
    """Round positive factors to the nearest power of two, exactly.

    ``np.exp2`` of an integer-valued float is an exact power of two, so every
    subsequent multiply/divide by such a factor perturbs only the exponent
    field: the scaling and its inverse are bit-exact (barring over/underflow,
    which the exponent clip and the finiteness assertions guard against).
    Non-finite or non-positive inputs -- an empty row or column -- get a factor
    of exactly 1.0.
    """
    v = np.asarray(values, dtype=float)
    out = np.ones(v.shape, dtype=float)
    good = np.isfinite(v) & (v > 0.0)
    if np.any(good):
        exponent = np.clip(np.rint(np.log2(v[good])), -max_exp, max_exp)
        out[good] = np.exp2(exponent)
    return out


def _inf_norms(M):
    if M.size == 0:
        return (np.ones(M.shape[0]), np.ones(M.shape[1]))
    A = np.abs(M)
    return A.max(axis=1), A.max(axis=0)


def _ratio(values):
    positive = values[values > 0.0]
    if positive.size == 0:
        return 1.0
    return float(positive.max() / positive.min())


def equilibrate(B, b, d, rounds=DEFAULT_RUIZ_ROUNDS):
    """Ruiz row/column infinity-norm balancing, power-of-two factors.

    Returns ``(B', b', d', r, c, info)`` with ``B' = R B C``, ``b' = R b``,
    ``d' = C d``, ``R = diag(r)``, ``C = diag(c)``.  Recover the original-space
    pair with ``x = c * x'`` and ``y = r * y'``.
    """
    B = np.asarray(B, dtype=float)
    n, m = B.shape
    r = np.ones(n)
    c = np.ones(m)
    M = B
    row0, col0 = _inf_norms(B)
    performed = 0
    for _ in range(int(rounds)):
        rn, cn = _inf_norms(M)
        dr = _pow2(1.0 / np.sqrt(np.where(rn > 0.0, rn, 1.0)))
        dc = _pow2(1.0 / np.sqrt(np.where(cn > 0.0, cn, 1.0)))
        if np.all(dr == 1.0) and np.all(dc == 1.0):
            break                       # fixed point of the power-of-two map
        M = (dr[:, None] * M) * dc[None, :]
        r = r * dr
        c = c * dc
        performed += 1
    # Recompute from the ORIGINAL data rather than trusting the accumulated
    # product, then assert the two agree BIT FOR BIT.  If any factor were not a
    # power of two this assertion is what would catch it.
    Bs = (r[:, None] * B) * c[None, :]
    bit_exact_accumulation = bool(np.array_equal(Bs, M))
    bs = r * np.asarray(b, dtype=float)
    ds = c * np.asarray(d, dtype=float)
    row1, col1 = _inf_norms(Bs)
    info = {
        "rounds_performed": performed,
        "rounds_max": int(rounds),
        "row_norm_ratio_before": _ratio(row0),
        "row_norm_ratio_after": _ratio(row1),
        "col_norm_ratio_before": _ratio(col0),
        "col_norm_ratio_after": _ratio(col1),
        "max_abs_before": float(np.abs(B).max()) if B.size else 0.0,
        "max_abs_after": float(np.abs(Bs).max()) if Bs.size else 0.0,
        "r_log2_min": float(np.log2(r).min()) if r.size else 0.0,
        "r_log2_max": float(np.log2(r).max()) if r.size else 0.0,
        "c_log2_min": float(np.log2(c).min()) if c.size else 0.0,
        "c_log2_max": float(np.log2(c).max()) if c.size else 0.0,
        "bit_exact_accumulation": bit_exact_accumulation,
        "all_finite": bool(np.all(np.isfinite(Bs)) and np.all(np.isfinite(bs))
                           and np.all(np.isfinite(ds))),
        # The invertibility claim, tested rather than asserted in prose: undo
        # the scaling and demand the ORIGINAL bytes back.
        "roundtrip_B_exact": bool(np.array_equal(
            B, (Bs / r[:, None]) / c[None, :])),
        "roundtrip_b_exact": bool(np.array_equal(np.asarray(b, float), bs / r)),
        "roundtrip_d_exact": bool(np.array_equal(np.asarray(d, float), ds / c)),
        "factors_are_powers_of_two": bool(
            np.array_equal(np.log2(r), np.rint(np.log2(r)))
            and np.array_equal(np.log2(c), np.rint(np.log2(c)))),
    }
    return Bs, bs, ds, r, c, info


def equil_identities(B, b, d, Bs, bs, ds, r, c, rng):
    """Numeric check of the four identities the block comment proves.

    Run OUTSIDE every timed region, on random probes, so the algebra is
    verified on the data actually being solved rather than only on paper.
    """
    n, m = B.shape
    if n == 0 or m == 0:
        return {"skipped": "empty program"}
    xs = rng.standard_normal(m)
    ys = np.abs(rng.standard_normal(n))
    x = c * xs
    y = r * ys
    scale_p = max(1.0, float(np.abs(B @ x).max()) if n else 1.0)
    scale_d = max(1.0, float(np.abs(B.T @ y).max()) if m else 1.0)
    return {
        "feasibility_max_abs": float(np.max(np.abs(
            (Bs @ xs - bs) - r * (B @ x - b)))) / scale_p,
        "feasibility_sign_agrees": bool(np.array_equal(
            (Bs @ xs - bs) >= 0.0, (B @ x - b) >= 0.0)),
        "dual_equality_max_abs": float(np.max(np.abs(
            (Bs.T @ ys - ds) - c * (B.T @ y - d)))) / scale_d,
        "sign_agrees": bool((ys.min() >= 0.0) == (y.min() >= 0.0)),
        "primal_objective_abs_diff": abs(float(ds @ xs) - float(d @ x)),
        "dual_objective_abs_diff": abs(float(bs @ ys) - float(b @ y)),
    }


_EQUILIBRATE = equilibrate      # alias: ``run``'s parameter shadows the name


class Alarm(Exception):
    pass


def _alarm(_signum, _frame):
    raise Alarm("ceiling reached")


def _clean(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape)}
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return str(value)


def run(name, seconds=300, staged_seconds=60.0, reference=False, t0=1.0,
        stage_factor=sn.STAGE_FACTOR, aim_factor=sn.AIM_FACTOR,
        viol_cap=sn.VIOL_CAP, max_stages=sn.MAX_STAGES, maxiter=newt.MAXITER,
        kernel=DEFAULT_KERNEL, screen_rows=False, screen_eps=sn.SCREEN_EPS,
        screen_after=sn.SCREEN_AFTER, face_cache=True,
        predictive_skip=False, g_skip_max=sn.G_SKIP_MAX, quiet=False,
        pchol_terminal=None, route=DEFAULT_ROUTE,
        equilibrate=DEFAULT_EQUILIBRATE, ruiz_rounds=DEFAULT_RUIZ_ROUNDS,
        candidates=True, alloc_patches=True):
    # The ORIGINAL raw program.  Everything the frozen gate ever sees is this.
    B0, b0, d0, _ = to_Bxgeb(ROOT / "netlib" / f"{name}.mps")
    n, m = B0.shape
    # S2 on the TERMINAL-TEST side.  ``None`` leaves oracle_terminal's own
    # default in place (ON); ``False`` restores the pre-S2 terminal solver for
    # an A/B arm.  Set per run, never left dangling.
    if pchol_terminal is not None:
        ot.PCHOL_TERMINAL = bool(pchol_terminal)

    fstar, reference_seconds = None, 0.0
    if reference:                       # OUTSIDE the timed region, opt-in only
        from scipy.optimize import linprog
        started = time.perf_counter()
        ref = linprog(d0, A_ub=-B0, b_ub=-b0, bounds=[(None, None)] * m,
                      method="highs")
        reference_seconds = time.perf_counter() - started
        fstar = float(ref.fun) if ref.success else None

    # ------------------------------------------------- forward transform
    # INSIDE the timed region, as its inverse is: a transform whose cost is
    # not charged to the lane is not a measurement of the lane.
    B, b, d = B0, b0, d0
    r_scale = c_scale = None
    equil_info = None
    transform_seconds = 0.0
    if equilibrate:
        transform_start = time.perf_counter()
        B, b, d, r_scale, c_scale, equil_info = _EQUILIBRATE(
            B0, b0, d0, rounds=ruiz_rounds)
        transform_seconds = time.perf_counter() - transform_start
        # FAIL CLOSED on the property the whole scheme rests on.  If the
        # factors were not powers of two, or the round trip did not return the
        # original bytes, this is not a reformulation and the run must not
        # proceed on it.
        for key in ("roundtrip_B_exact", "roundtrip_b_exact",
                    "roundtrip_d_exact", "factors_are_powers_of_two",
                    "bit_exact_accumulation", "all_finite"):
            if not equil_info[key]:
                raise AssertionError(
                    "equilibration is not exactly invertible on %s: %s is "
                    "False" % (name, key))

    route_record = {}
    live_flags = kernel_state()
    diag = {}
    started = time.perf_counter()
    status, result = None, None
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds, 5.0)
    try:
        with install_certificate_route(route, route_record), \
                promoted_kernel(candidates=candidates, alloc=alloc_patches):
            # Read off the modules INSIDE the block: the flags are restored on
            # exit, so a state recorded afterwards would describe the defaults
            # rather than the run.
            live_flags = kernel_state()
            result = sn.follow_staged(
                B, b, d, staged_seconds=staged_seconds,
                staged_diagnostics=diag,
                t0=t0, stage_factor=stage_factor, aim_factor=aim_factor,
                viol_cap=viol_cap, max_stages=max_stages, maxiter=maxiter,
                kernel=kernel, screen_rows=screen_rows,
                screen_eps=screen_eps,
                screen_after=screen_after, face_cache=face_cache,
                predictive_skip=predictive_skip, g_skip_max=g_skip_max)
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        status = result.get("status")
    except Alarm:
        status = "RESOURCE-LIMIT"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)
    wall = time.perf_counter() - started

    # ---------------------------- inverse transform, then THE ONLY VERDICT --
    # ORDER IS THE WHOLE POINT.  Unscale FIRST (x = c*x', y = r*y'), certify
    # SECOND, with the FROZEN ``exp23.certificate_pair`` on the ORIGINAL raw
    # (B0, b0, d0), componentwise, TOL_FEAS = 1e-7.  The status the solver
    # reached in the SCALED space is kept as ``transformed_status`` and is
    # never allowed to set the verdict.  With ``--no-equilibrate`` the solver
    # already ran on the original data and this block is inert, so the
    # pre-promotion record is reproduced field for field.
    transformed_status = status
    inverse_seconds = 0.0
    certify_seconds = 0.0
    certificate = None
    x_raw = y_raw = None
    if isinstance(result, dict):
        x_t, y_t = result.get("x"), result.get("y")
        if x_t is not None:
            x_raw = np.asarray(x_t, dtype=float)
        if y_t is not None:
            y_raw = np.asarray(y_t, dtype=float)
        if equilibrate:
            inverse_start = time.perf_counter()
            if x_raw is not None:
                x_raw = c_scale * x_raw
            if y_raw is not None:
                y_raw = r_scale * y_raw
            inverse_seconds = time.perf_counter() - inverse_start
    if equilibrate:
        if x_raw is None or y_raw is None:
            status = "SOLVE-RETURNED-NO-PAIR" if status != "RESOURCE-LIMIT" \
                else status
        elif x_raw.shape != (m,) or y_raw.shape != (n,):
            status = "LIFT-SHAPE-FAILED"
        else:
            digest_before = (x_raw.tobytes(), y_raw.tobytes())
            certify_start = time.perf_counter()
            ok, detail = FROZEN_CERTIFICATE_PAIR(B0, b0, d0, x_raw, y_raw)
            certify_seconds = time.perf_counter() - certify_start
            components = [detail.get(k) for k in
                          ("primal", "dual", "nonnegative", "gap")]
            certificate = {
                "primal": detail.get("primal"), "dual": detail.get("dual"),
                "nonnegative": detail.get("nonnegative"),
                "gap": detail.get("gap"), "reason": detail.get("reason"),
                "max": (max(components) if all(v is not None
                                               for v in components) else None),
                "tol_feas": exp23.TOL_FEAS,
                "authority": ("frozen exp23.certificate_pair on the ORIGINAL "
                              "raw (B, b, d), componentwise"),
                "pair_unmodified_by_gate": bool(
                    digest_before == (x_raw.tobytes(), y_raw.tobytes())),
            }
            status = "CERTIFIED" if ok else "GATE-FAILED"
            # FAIL CLOSED on the audited property: nothing may report
            # CERTIFIED unless the frozen componentwise gate passed on the
            # ORIGINAL data, on the very bytes the inverse transform produced.
            if status == "CERTIFIED" and not (
                    certificate["reason"] == "passed"
                    and certificate["max"] is not None
                    and certificate["max"] <= exp23.TOL_FEAS
                    and certificate["pair_unmodified_by_gate"]):
                raise AssertionError(
                    "CERTIFIED without a clean frozen gate on the original "
                    "data: %r" % (certificate,))

    stages = diag.get("stages", [])
    output = {
        "model": name,
        "shape": [int(n), int(m)],
        "status": status,
        "lane": "staged-newton",
        "kernel": kernel,
        "screen": bool(screen_rows),
        "predictive_skip": bool(predictive_skip),
        "g_skip_max": g_skip_max,
        "screen_eps": screen_eps,
        "stage_factor": stage_factor,
        "viol_cap": viol_cap,
        "t0": t0,
        "route": (result or {}).get("route", "n/a"),
        "backend": (result or {}).get("backend"),
        "staged_fallback": (result or {}).get("staged_fallback"),
        "staged_fallback_reason": (result or {}).get("staged_fallback_reason"),
        "ceiling_seconds": seconds,
        # HEADLINE.  With equilibration on, wall = transform + solve + inverse,
        # exactly as ``bench_pivchol_alloc._run_cell`` defines it, so the
        # transform is charged to the lane.  With it off this is the solve
        # alone, which is what every pre-promotion harness reads it as.
        "wall_seconds": round(
            wall + transform_seconds + inverse_seconds, 3),
        "solve_seconds": round(wall, 3),
        "transform_seconds": round(transform_seconds, 6),
        "inverse_seconds": round(inverse_seconds, 6),
        "certify_seconds": round(certify_seconds, 6),
        # ---- the promoted configuration, as READ OFF the modules ----
        "configuration": {
            "route": route,
            "certificate_route": route_record,
            "equilibrate": bool(equilibrate),
            "ruiz_rounds": int(ruiz_rounds) if equilibrate else None,
            "candidates": candidates,
            "alloc_patches": alloc_patches,
            "kernel_flags": _clean(live_flags),
        },
        "equilibration": equil_info,
        "transformed_status": transformed_status,
        "certificate": _clean(certificate),
        "certificate_max": (certificate or {}).get("max"),
        "reference_seconds": round(reference_seconds, 3),
        "wall_split": {
            "newton_solve_seconds": round(
                float(diag.get("newton_seconds", 0.0)), 3),
            "newton_step_seconds": round(
                float(diag.get("newton_step_seconds", 0.0)), 3),
            "screen_seconds": round(float(diag.get("screen_seconds", 0.0)), 3),
            "terminal_test_seconds": round(
                float(diag.get("terminal_test_seconds", 0.0)), 3),
            "fallback_walk_seconds": round(float(
                (result or {}).get("staged_fallback_walk_seconds", 0.0)), 3),
        },
        "newton_iters_total": diag.get("newton_iters_total"),
        "stages_run": diag.get("stages_run"),
        "terminal_tests_run": diag.get("terminal_tests_run"),
        "terminal_tests_skipped": diag.get("terminal_tests_skipped"),
        "t_accepted": _clean(diag.get("t_accepted")),
        "accepted_stage": _clean(diag.get("accepted_stage")),
        "accepted_face_size": _clean(diag.get("accepted_support")),
        "sweep_reason": diag.get("sweep_reason"),
        # S1/S2/F2 gate and route detail
        "kernel_counters": _clean(diag.get("kernel_counters")),
        "tik_terminal": diag.get("tik_terminal"),
        "tik_unit_fraction": diag.get("tik_unit_fraction"),
        "tik_stats": _clean(diag.get("tik_stats")),
        # TERMINAL-TEST face solver counters, recorded on EVERY model (the
        # staged lane's own ``tik_stats`` only appears when the S1 classifier
        # was installed, which is 2 of 27; the S2 piv-chol route serves the
        # m<=700 cohort, where it is not).
        "pchol_terminal": bool(ot.PCHOL_TERMINAL),
        "terminal_face_stats": _clean(ot.tik_stats()),
    }

    output["stage_table"] = [{
        "stage": s.get("stage"), "t": s.get("t"),
        "seconds": s.get("newton_seconds"), "iterations": s.get("iterations"),
        "rows_solved": s.get("rows_solved"),
        "screen": _clean(s.get("screen")),
        "g_predicted": s.get("g_predicted"),
        "screen_reverted": s.get("screen_reverted"),
        "converged": s.get("converged"),
        "newton_reason": s.get("newton_reason"),
        "test_reason": s.get("reason"),
        "support": s.get("support"), "grad_relative": s.get("grad_relative"),
        "max_step_residual": s.get("max_step_residual"),
        "step_solver": s.get("step_solver"),
        "g_relative": s.get("g_relative"),
        "face_terminal": s.get("face_terminal"),
        "sigma_min_relative": s.get("sigma_min_relative"),
        "t_bar": s.get("t_bar"), "t_bar_candidate": s.get("t_bar_candidate"),
        "gate_primal": (s.get("gate") or {}).get("primal"),
        "gate_reason": (s.get("gate") or {}).get("reason"),
        "repair_rounds": len(s.get("repair") or []),
        "outcome": s.get("outcome"),
    } for s in stages]

    # screening effect, aggregated
    fractions = [1.0 - float(s["rows_solved"]) / n for s in stages
                 if s.get("rows_solved")]
    if fractions:
        output["screen_effect"] = {
            "stages": len(fractions),
            "median_row_reduction": float(np.median(fractions)),
            "max_row_reduction": float(np.max(fractions)),
            "stages_screened": int(sum(1 for f in fractions if f > 0.0)),
            "reverted": int(sum(1 for s in stages
                                if s.get("screen_reverted"))),
        }

    if result is not None:
        # ``lb`` is the CERTIFIED lower bound and must be read off the pair the
        # frozen gate saw, i.e. on the ORIGINAL data.  Under equilibration the
        # two agree exactly anyway -- b'y = (Rb)'(R^-1 y) term by term with
        # power-of-two factors -- but the record must quote the original.
        lb = result.get("lb")
        if equilibrate and y_raw is not None:
            lb = float(b0 @ y_raw)
        output["lb"] = _clean(lb)
        output["pivots"] = _clean(result.get("pivots"))
        expected = KNOWN_LB.get(name)
        if expected is not None and lb is not None:
            output["lb_expected"] = expected
            output["lb_relative_error"] = abs(float(lb) - expected) / max(
                1.0, abs(expected))
            output["lb_match_1e9"] = bool(output["lb_relative_error"] <= 1e-9)
        if fstar is not None and x_raw is not None:
            objective_error = abs(float(d0 @ x_raw) - fstar) / max(
                1.0, abs(fstar))
            output["f_star"] = fstar
            output["objective_relative_error"] = objective_error
            output["objective_match"] = bool(objective_error <= 1e-9)
        detail = result.get("certificate_detail")
        if detail is None:
            detail = result.get("fallback_certificate")
        output["certificate_detail"] = _clean(detail)
        # Diagnostics on the ORIGINAL data with the UNSCALED pair.  Note this
        # block is differently scaled from the frozen componentwise gate above
        # and is NOT the accuracy number (agent_reports/24).
        x, y = x_raw, y_raw
        if y is not None:
            scale = max(1.0, float(np.abs(y).max()) if y.size else 1.0)
            output["final_face_size"] = int((y > exp23.TOL_Y * scale).sum())
            output["objective"] = (float(d0 @ x) if x is not None else None)
            output["gate_full"] = {
                "min_y": float(y.min()),
                "dual_residual_inf": float(np.abs(B0.T @ y - d0).max())
                / max(1.0, float(np.abs(d0).max())),
            }
            if x is not None:
                sigma = B0 @ x - b0
                output["gate_full"].update({
                    "primal_min_sigma": float(sigma.min())
                    / max(1.0, float(np.abs(b0).max())),
                    "gap_relative": abs(float(d0 @ x) - float(b0 @ y))
                    / max(1.0, abs(float(d0 @ x))),
                })

    if not quiet:
        print(json.dumps(output, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--staged-seconds", type=float, default=60.0)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--stage-factor", type=float,
                        default=sn.STAGE_FACTOR)
    parser.add_argument("--aim-factor", type=float, default=sn.AIM_FACTOR)
    parser.add_argument("--viol-cap", type=float, default=sn.VIOL_CAP)
    parser.add_argument("--max-stages", type=int, default=sn.MAX_STAGES)
    parser.add_argument("--maxiter", type=int, default=newt.MAXITER)
    parser.add_argument("--kernel",
                        choices=("svd", "fast", "fast+unitcore", "fast+tik",
                                 "fast+pchol", "fast+tik+pchol"),
                        default=DEFAULT_KERNEL,
                        help="'fast' is the pre-S1/S2 lane; '+tik' adds the "
                             "Tikhonov-Woodbury face/step route behind its "
                             "unit-row-fraction gate, '+pchol' the pivoted "
                             "Cholesky step route for m <= 700")
    parser.add_argument("--f2-repair", action="store_true",
                        help="opt in to the F2 repair of incumbent twin-LSQR "
                             "steps that miss the honest-residual budget "
                             "(the CHECK is always on and always counted; "
                             "see newton_oracle.F2_REPAIR for why the repair "
                             "is off by default)")
    parser.add_argument("--f2-dense", action="store_true",
                        help="with --f2-repair, allow the forced dense-SVD "
                             "second stage of the repair")
    parser.add_argument("--screen", action="store_true",
                        help="enable G1 gap-safe row screening")
    parser.add_argument("--screen-eps", type=float, default=sn.SCREEN_EPS)
    parser.add_argument("--screen-after", type=int, default=sn.SCREEN_AFTER)
    parser.add_argument("--no-pchol-terminal", action="store_true",
                        help="restore the pre-S2 terminal-test face solver "
                             "(dense piece_y for the m<=700 cohort); the "
                             "baseline arm of the terminal-bucket A/B")
    parser.add_argument("--no-face-cache", action="store_true")
    parser.add_argument("--predictive-skip", action="store_true")
    parser.add_argument("--g-skip-max", type=int, default=sn.G_SKIP_MAX)
    # ---- the four promoted items, each individually reversible ------------
    parser.add_argument("--route", choices=list(ROUTES), default=DEFAULT_ROUTE,
                        help="certificate-detail route installed for the "
                             "solve; 'frozen' restores "
                             "exp23_path_primal_dual._certificate_detail "
                             "(default: %s)" % DEFAULT_ROUTE)
    parser.add_argument("--no-equilibrate", action="store_true",
                        help="solve the RAW program instead of the Ruiz "
                             "power-of-two equilibrated one (the "
                             "pre-promotion pipeline)")
    parser.add_argument("--ruiz-rounds", type=int,
                        default=DEFAULT_RUIZ_ROUNDS)
    parser.add_argument("--no-candidates", action="store_true",
                        help="restore the pre-promotion c1_const_cache / "
                             "c5_pchol_screen kernel settings")
    parser.add_argument("--no-alloc-patches", action="store_true",
                        help="restore the pre-promotion a1/a2/a3/b1 "
                             "allocation code paths")
    args = parser.parse_args()
    if args.f2_repair:
        newt.F2_REPAIR = True
    if args.f2_dense:
        newt.F2_DENSE_FALLBACK = True
    run(args.model, seconds=args.seconds, staged_seconds=args.staged_seconds,
        reference=args.reference, t0=args.t0, stage_factor=args.stage_factor,
        aim_factor=args.aim_factor, viol_cap=args.viol_cap,
        max_stages=args.max_stages, maxiter=args.maxiter, kernel=args.kernel,
        screen_rows=args.screen, screen_eps=args.screen_eps,
        screen_after=args.screen_after, face_cache=not args.no_face_cache,
        predictive_skip=args.predictive_skip, g_skip_max=args.g_skip_max,
        pchol_terminal=not args.no_pchol_terminal,
        route=args.route, equilibrate=not args.no_equilibrate,
        ruiz_rounds=args.ruiz_rounds,
        candidates=not args.no_candidates,
        alloc_patches=not args.no_alloc_patches)


if __name__ == "__main__":
    main()
