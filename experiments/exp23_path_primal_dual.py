"""exp23: track the path in y-SPACE.  The dual path is a strictly convex
parametric QP, so y(t) is UNIQUE at every t -- track it, not u(t).

DIAGNOSIS (sharper than "V decouples from u")
---------------------------------------------
Program (4)'s minimiser u*(t) is NON-UNIQUE exactly at degeneracy: on a piece
with rank(B_V) < m the minimiser set is the affine family u* + ker(B_V), and
different members of that family have different residual sign patterns.  So
"the violated set of the computed u" is not a well-defined object.  exp21 and
exp22 both failed for the same reason: they bookkeep a label V for a point u
that does not exist uniquely.  No bookkeeping discipline can fix that.

But the RECOVERED DUAL is unique for every t.  Take Prop 2's QP
    min_{u,w} 1/2||w||^2 + d'u   s.t.  w <= Bu - tb
and form its Wolfe dual (multiplier mu >= 0; minimise over w: w = -mu;
over u: B'mu = d):
    y*_t = argmin { 1/2||y - t b||^2 :  B'y = d,  y >= 0 }        (QP_t)
-- STRICTLY convex, unique solution, piecewise affine in t, and equal to the
penalty recovery y*_t = -(Bu*_t - tb)_- for any minimiser u*_t of (4).
All of exp21/exp22's pathology is the shadow of u-non-uniqueness on a path
that is itself single-valued.

Two structural gifts of (QP_t):

  (a) On a support W = supp(y), y_W(t) is the orthogonal projection of t*b_W
      onto the affine set {B_W'y = d}:
          y_W(t) = t*g + h,   g = b_W - Pi_W b_W,   h = argmin-norm{B_W'y = d},
    with Pi_W = projection onto range(B_W).  Both terms are pseudoinverse
      expressions with NO rank assumption on B_W.  (exp21 died solving
      B_V'B_V u = ... when
      B_V'B_V went singular; that system never appears here.)

  (b) Comparing the variational inequalities of (QP_t) at t1 < t2:
          (t2 - t1) * (b'y(t2) - b'y(t1))  >=  ||y(t2) - y(t1)||^2  >=  0,
      so MONOTONE DUAL ASCENT is a theorem for the true path.

THE PRIMAL SIDE, AND WHERE THE HEURISTIC LIVES
----------------------------------------------
The multiplier u of the equality constraint B'y = d satisfies
B_W u = y_W - t b_W, and x := -u/t is the primal iterate: complementary on W
(B_i x = b_i - y_i/t) and feasible off W (B_i x >= b_i).  The off-support
slacks r_i(t) = -(t b_i + B_i u(t)) drive the ENTERING test.  When
rank(B_W) < m, u is non-unique -- that non-uniqueness IS primal degeneracy --
and we take the min-norm u.  That is the heuristic.  Every piece is validated
by a sign-repair loop ("settle": drop i with y_i < 0, add i with r_i < 0,
re-solve).  A fixed-t corrector normally uses NNLS.  Its stronger repair keeps
a sparse equality basis, applies product-form basis exchanges and rank-one
reduced-Gram updates, and uses a checked sparse augmented-KKT solve on an
ill-conditioned exceptional face.  If that active-set projection stalls at a
vertex, it projects the negative objective gradient onto the tangent cone,
takes the exact decreasing step to the first bound, rebuilds the support, and
continues.  If an ordinary path breakpoint cannot settle, a separate
critical-cone derivative selects the right-hand face from the exact affine
endpoint and immediately returns control to this walker. Neither exceptional
repair is allowed to certify the original LP by itself.

PRACTICAL SAFETY NET
--------------------
The corrector and sign-repair steps are heuristics, and real degenerate problems
can return an empty support or cycle among incompatible supports.  ``follow``
therefore limits resynchronization and critical repairs by default and falls
back first to an explicitly labelled supporting-face simplex crossover when
optional ``highspy`` and a certified projection point are available, then to
the direct HiGHS dual-simplex solve.  These are ordinary LP escape branches,
not path successes.  Every success, including a terminal affine piece, must
pass a separate componentwise backward-error certificate for the original LP.
Set ``fallback=False`` to require path-only success.

Run:  ~/.venvs/claude/bin/python exp23_path_primal_dual.py
"""

import numpy as np
from pathlib import Path
from scipy.optimize import linprog, nnls
from instrumented import lpgen2

TOL_Y = 1e-9        # dual-support threshold (certificate)
TOL_FEAS = 1e-7     # KKT certificate tolerance
TIE = 1e-9          # breakpoint tie window (relative)
FWD = 1e-12         # strict-forward filter for breakpoints
MAXPIV = 2000
TMAX = 1e8
ACTIVE_PROJECTION_CEILING = 600  # includes lotfi; avoids large-instance churn


def _emit_trace(trace, event, **fields):
    """Emit one structured diagnostic record without affecting normal runs."""
    if trace is None:
        return
    record = {"event": event, **fields}
    if callable(trace):
        trace(record)
    else:
        trace.append(record)


# ------------------------------------------------------------ piece in y-space

def piece_y(B, b, d, W):
    """y_W(t) = t*g + h and min-norm multiplier u(t) = t*ua + uc.  Rank-free.

    One SVD of B_W replaces the four lstsq calls of the original version (each
    lstsq is itself an SVD-based solve of the same matrix); all four quantities
    are pseudoinverse expressions in U, s, V'.  Same rank cutoff as
    lstsq(rcond=None): s_i <= s_max * max(shape) * eps."""
    BW = B[W]
    bW = b[W]
    U, s, Vt = np.linalg.svd(BW, full_matrices=False)
    if s.size:
        keep = s > s[0] * max(BW.shape) * np.finfo(BW.dtype).eps
        Ur, sr, Vr = U[:, keep], s[keep], Vt[keep]
    else:
        Ur, sr, Vr = U, s, Vt
    Vd = Vr @ d
    h = Ur @ (Vd / sr) if sr.size else np.zeros(BW.shape[0])   # min-norm B_W'h = d
    dres = float(np.linalg.norm(BW.T @ h - d)) / max(1.0, float(np.linalg.norm(d)))
    c = Ur.T @ bW
    g = bW - Ur @ c                                       # (I - U U') b_W
    g = g - Ur @ (Ur.T @ g)                               # re-orthogonalize
    ua = -(Vr.T @ (c / sr)) if sr.size else np.zeros(BW.shape[1])  # min-norm BW ua = g - bW
    uc = Vr.T @ (Vd / sr**2) if sr.size else np.zeros(BW.shape[1])  # min-norm BW uc = h
    return g, h, ua, uc, dres


def settle(B, b, d, W, t, rounds=20, diagnostics=None, piece_solver=None):
    """Validate/repair W at fixed t: drop y_i < 0, add r_i < 0, re-solve."""
    W = W.copy()
    seen = set()
    for iteration in range(rounds):
        key = np.packbits(W).tobytes()
        if key in seen:
            if diagnostics is not None:
                diagnostics.update(reason="support cycle", rounds=iteration)
            return W, None, False
        seen.add(key)
        if not W.any():
            if diagnostics is not None:
                diagnostics.update(reason="empty support", rounds=iteration)
            return W, None, False
        coefficients = (piece_y(B, b, d, W) if piece_solver is None
                        else piece_solver(W))
        g, h, ua, uc, dres = coefficients
        yW = t * g + h
        r = -(t * (b + B @ ua) + B @ uc)                  # off-support slacks
        y_scale = 1e-8 * max(1.0, float(np.abs(yW).max()) if yW.size else 1.0)
        r_scale = 1e-8 * max(1.0, float(np.abs(r).max()) if r.size else 1.0)
        add = (~W) & (r < -r_scale)
        idxW = np.where(W)[0]
        drop = idxW[yW < -y_scale]
        if not add.any() and drop.size == 0:
            ok = dres <= TOL_FEAS
            if diagnostics is not None:
                diagnostics.update(reason="settled" if ok else "dual equality",
                                   rounds=iteration + 1, added=0, dropped=0)
            return W, (g, h, ua, uc), ok
        if diagnostics is not None:
            diagnostics.update(added=int(add.sum()), dropped=int(drop.size))
        W[add] = True
        W[drop] = False
    if diagnostics is not None:
        diagnostics.update(reason="round cap", rounds=rounds)
    return W, None, False


# ------------------------------------------------------------------- corrector

def qp_corrector(B, b, d, t, attempt=0, active_set_repair=False,
                 diagnostics=None, retain_projection_working_set=False):
    """Solve (QP_t) directly: min ||y - tb||^2 + w^2||B'y - d||^2, y >= 0 via
    NNLS.  Lawson-Hanson returns exact zeros -> clean support identification.
    On ill-conditioned (real, sparse) instances LH can stall at large w, so
    cascade the weight down; last resort is a newton4 solve of program (4).
    For large n Lawson-Hanson is far too slow (minutes on degen2), so there
    the cheap newton4 corrector is used exclusively and escalation happens
    through the caller's t-doubling; NNLS remains primary for n <= 400."""
    n, m = B.shape
    if active_set_repair and n <= ACTIVE_PROJECTION_CEILING:
        try:
            from lex_active_set_projection import project_lex_active_set
            projected = project_lex_active_set(B, b, d, t=t, maxiter=2000,
                                               max_tangent_escapes=20)
            if diagnostics is not None:
                diagnostics.update(
                    backend="tangent active-set projection",
                    projection_status=projected["status"],
                    projection_iterations=projected.get("iterations", 0),
                    projection_perturbations=projected.get("perturbations", 0),
                    tangent_escapes=projected.get("tangent_escapes", 0),
                    projection_basis_refactorizations=projected.get(
                        "basis_refactorizations", 0),
                    projection_basis_updates=projected.get("basis_updates", 0),
                    projection_sparse_kkt_repairs=projected.get(
                        "sparse_kkt_repairs", 0),
                    projection_error=max(
                        projected.get("equality_error", np.inf),
                        projected.get("nonnegative_error", np.inf),
                        projected.get("stationarity_error", np.inf)))
            if projected["status"] == "PROJECTED":
                if diagnostics is not None:
                    diagnostics["projection_y"] = projected["y"].copy()
                    diagnostics["projection_working_set_size"] = int(
                        np.sum(projected.get("working_set",
                                             projected["support"])))
                if retain_projection_working_set:
                    return np.asarray(projected.get(
                        "working_set", projected["support"]),
                        dtype=bool).copy()
                return projected["support"].copy()
        except Exception as error:
            if diagnostics is not None:
                diagnostics.update(backend="tangent active-set projection",
                                   projection_status="EXCEPTION",
                                   projection_detail=type(error).__name__)
    if n > 400:
        u = newton4(B, b, d, t)
        y = np.maximum(t * b - B @ u, 0.0)
        thr = 1e-9 * max(1.0, float(y.max()) if y.size else 0.0)
        return y > thr
    sc = max(1.0, float(np.abs(B).max()))
    for w in (1e6 / sc, 1e5 / sc, 1e4 / sc, 1e3 / sc):
        A = np.vstack([np.eye(n), w * B.T])
        rhs = np.concatenate([t * b, w * d])
        try:
            y, _ = nnls(A, rhs, maxiter=30 * n)
        except Exception:
            continue
        thr = 1e-9 * max(1.0, float(y.max()) if y.size else 0.0)
        return y > thr
    u = newton4(B, b, d, t)
    y = np.maximum(t * b - B @ u, 0.0)
    thr = 1e-9 * max(1.0, float(y.max()) if y.size else 0.0)
    return y > thr


# ----------------------------------------------------------------- certificate

def _certificate_detail(B, b, d, y):
    """t-free KKT test, INCLUDING dual feasibility.  Passing => exact optimum.

    When rank(B_A) < m (routine on real sparse LPs), the lstsq x is not pinned
    by B_A x = b_A alone; complete it in null(B_A) by minimizing the
    PARAMETER-FREE violation penalty min_z ||(B(x+Nz) - b)_-||^2 -- program (4)
    with d = 0 -- which preserves B_A x = b_A and stays t-free."""
    n, m = B.shape
    sc_d = max(1.0, float(np.abs(d).max()))
    dres = float(np.max(np.abs(B.T @ y - d)))
    if dres > TOL_FEAS * sc_d or (y.size and float(y.min()) < -TOL_FEAS):
        reason = "dual equality" if dres > TOL_FEAS * sc_d else "negative dual"
        return False, None, dres / sc_d, reason
    A = np.where(y > TOL_Y * max(1.0, float(y.max()) if y.size else 1.0))[0]
    if A.size == 0:
        return False, None, np.inf, "empty support"
    x, _, rank, _ = np.linalg.lstsq(B[A], b[A], rcond=None)
    consist = float(np.max(np.abs(B[A] @ x - b[A])))
    viol = max(0.0, float(-(B @ x - b).min()))
    sc_b = max(1.0, float(np.abs(b).max()))
    if consist <= TOL_FEAS * sc_b and viol > TOL_FEAS * sc_b and rank < m:
        _, sv, Vt = np.linalg.svd(B[A])
        r = int((sv > (sv.max() if sv.size else 0.0) * 1e-10).sum())
        N = Vt[r:].T                                   # null(B_A), m x (m-r)
        if N.shape[1] > 0:
            z = newton4(B @ N, b - B @ x, np.zeros(N.shape[1]), 1.0)
            x2 = x + N @ z
            viol2 = max(0.0, float(-(B @ x2 - b).min()))
            if viol2 < viol:
                x, viol = x2, viol2
                consist = float(np.max(np.abs(B[A] @ x - b[A])))
    if consist > TOL_FEAS * sc_b:
        return False, x, consist / sc_b, "active consistency"
    if viol > TOL_FEAS * sc_b:
        return False, x, viol / sc_b, "primal violation"
    ok = True
    if ok:
        # Explicit duality-gap check.  Without it the certificate is NOT
        # airtight: consist/viol are scaled by the GLOBAL max|b|, so one
        # benign huge-scale row loosens the absolute tolerance on every row,
        # and with a large-scale d the complementarity leak y'(Bx - b) can be
        # unbounded (constructed failure: |d'x - f*| = 5e7 with ok=True).
        px, dy = float(d @ x), float(b @ y)
        gap = abs(px - dy) / max(1.0, abs(px), abs(dy))
        if gap > TOL_FEAS:
            return False, x, gap, "duality gap"
    pair_ok, pair_detail = certificate_pair(B, b, d, x, y)
    if not pair_ok:
        pair_err = max(pair_detail.get("primal", 0.0),
                       pair_detail.get("dual", 0.0),
                       pair_detail.get("nonnegative", 0.0),
                       pair_detail.get("gap", 0.0))
        return False, x, pair_err, "componentwise " + pair_detail["reason"]
    return True, x, max(consist, viol) / sc_b, "passed"


def certificate(B, b, d, y):
    """Compatibility wrapper returning the historical three-value result."""
    ok, x, err, _ = _certificate_detail(B, b, d, y)
    return ok, x, err


def _support_diagnostics(B, b, d, W, t, piece_solver=None):
    """Describe a support even when settle could not validate it."""
    if not W.any():
        return np.inf, "empty support"
    coefficients = (piece_y(B, b, d, W) if piece_solver is None
                    else piece_solver(W))
    g, h, _, _, _ = coefficients
    y = np.zeros(B.shape[0])
    y[W] = np.maximum(t * g + h, 0.0)
    dres = (float(np.linalg.norm(B.T @ y - d))
            / max(1.0, float(np.linalg.norm(d))))
    _, _, _, reason = _certificate_detail(B, b, d, y)
    return dres, reason


def critical_vertex_escape(B, b, d, y, t, trace=None, rounds=50,
                           piece_solver=None, settle_solver=None):
    """Use the exact affine endpoint to select the right-hand degenerate face.

    This is an exceptional transition oracle.  It does not project at fixed t
    and does not continue the path itself: once a right-hand support settles,
    control returns immediately to the ordinary piecewise-affine walker.
    """
    from critical_cone_path import _prepare, critical_derivative

    y = np.asarray(y, dtype=float).copy()
    yscale = max(1.0, float(np.abs(y).max()) if y.size else 1.0)
    endpoint_tol = 1e-8 * yscale
    y[np.abs(y) <= endpoint_tol] = 0.0
    if y.size and float(y.min()) < -100.0 * endpoint_tol:
        return None, {"reason": "negative endpoint", "minimum": float(y.min())}
    y = np.maximum(y, 0.0)

    E, _f, _selected, _row_scale, consistency = _prepare(B, d)
    if consistency > TOL_FEAS:
        return None, {"reason": "equality reduction", "error": consistency}
    zero = y <= endpoint_tol
    qnormal = t * b - y
    derivative = critical_derivative(E, b, qnormal, zero)
    if "Solved" not in derivative["status"]:
        return None, {"reason": "critical QP", "status": derivative["status"]}
    v = derivative["v"]

    vscale = max(1.0, float(np.abs(v).max()) if v.size else 1.0)
    equality_error = (float(np.linalg.norm(E @ v, np.inf))
                      / max(1.0, float(np.linalg.norm(v, np.inf))))
    tangent_error = (max(0.0, float(-v[zero].min())) / vscale
                     if zero.any() else 0.0)
    orthogonality_error = abs(float(qnormal @ v)) / max(
        1.0, float(np.linalg.norm(qnormal)) * float(np.linalg.norm(v)))
    if max(equality_error, tangent_error, orthogonality_error) > TOL_FEAS:
        return None, {"reason": "critical residual", "equality": equality_error,
                      "tangent": tangent_error,
                      "orthogonality": orthogonality_error}

    right_support = (y > endpoint_tol) | (v > 1e-8 * vscale)
    if not right_support.any():
        return None, {"reason": "zero right derivative",
                      "vnorm": float(np.linalg.norm(v))}

    # Try successively larger right-limit probes.  Each is tiny relative to t;
    # settle is the acceptance gate, so a probe that crosses an adjacent event
    # is rejected rather than silently accepted.
    last_detail = {}
    for exponent in (-10, -9, -8, -7, -6):
        epsilon = (10.0 ** exponent) * max(1.0, abs(t))
        tp = t + epsilon
        diagnostics = {}
        if settle_solver is None:
            W, pc, ok = settle(B, b, d, right_support, tp, rounds=rounds,
                               diagnostics=diagnostics,
                               piece_solver=piece_solver)
        else:
            W, pc, ok = settle_solver(
                right_support, tp, rounds=rounds, diagnostics=diagnostics)
        last_detail = diagnostics
        if ok:
            detail = {"reason": "settled", "t_from": float(t),
                      "t_to": float(tp), "epsilon": float(epsilon),
                      "support": int(W.sum()),
                      "entered": int(np.sum(zero & W)),
                      "vnorm": float(np.linalg.norm(v)),
                      "equality": equality_error,
                      "tangent": tangent_error,
                      "orthogonality": orthogonality_error}
            _emit_trace(trace, "critical escape", **detail)
            return (W, pc, tp), detail
    return None, {"reason": "right face did not settle",
                  "settle_reason": last_detail.get("reason"),
                  "vnorm": float(np.linalg.norm(v)),
                  "equality": equality_error,
                  "tangent": tangent_error,
                  "orthogonality": orthogonality_error}


def certificate_pair(B, b, d, x, y):
    """Backward-error KKT certificate for an explicit primal-dual pair.

    Every residual is scaled componentwise by the data and candidate magnitudes;
    this avoids allowing one large row or column to loosen every other test.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if (x.shape != (B.shape[1],) or y.shape != (B.shape[0],)
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        return False, {"reason": "nonfinite or wrong-shaped solution"}
    primal = B @ x - b
    dual = B.T @ y - d
    pscale = 1.0 + np.abs(b) + np.abs(B) @ np.abs(x)
    dscale = 1.0 + np.abs(d) + np.abs(B).T @ np.abs(y)
    primal_err = float(np.max(np.maximum(-primal, 0.0) / pscale))
    dual_err = float(np.max(np.abs(dual) / dscale))
    nonneg_err = max(0.0, float(-y.min())) / (1.0 + float(np.abs(y).max()))
    px, dy = float(d @ x), float(b @ y)
    gap_err = abs(px - dy) / (1.0 + abs(px) + abs(dy))
    errors = dict(primal=primal_err, dual=dual_err, nonnegative=nonneg_err,
                  gap=gap_err)
    for reason, err in errors.items():
        if err > TOL_FEAS:
            return False, {"reason": reason, **errors}
    return True, {"reason": "passed", **errors}


def _highs_fallback(B, b, d):
    """Solve and independently certify the LP when path heuristics do not settle."""
    m = B.shape[1]
    result = linprog(d, A_ub=-B, b_ub=-b,
                     bounds=[(None, None)] * m, method="highs-ds")
    if not result.success:
        return None, None, result
    x = np.asarray(result.x, dtype=float)
    y = -np.asarray(result.ineqlin.marginals, dtype=float)
    passed, detail = certificate_pair(B, b, d, x, y)
    if not passed:
        result.certificate_detail = detail
        return None, None, result
    result.certificate_detail = detail
    return x, y, result


# ------------------------------------------------------------------ the solver

def _follow_selector_engine(B, b, d, t=1.0, maxpiv=MAXPIV, tmax=TMAX,
                            trace=None, progress=None, fallback=True):
    """Promoted selector-gated engine (2026-08-06); record in agent_reports/11.

    Lazy import avoids a cycle: selector_gated_walker imports this module at
    its top level.  The returned dict is mapped onto ``follow``'s historical
    contract; ``critical_repairs``/``critical_failures`` mirror the selector
    walker's ``critical_transitions``/``critical_rejections``.
    """
    from selector_gated_walker import follow_selector_gated
    st = follow_selector_gated(
        B, b, d, t=t, maxpiv=maxpiv, tmax=tmax, trace=trace,
        progress=progress, legacy_repair=False,
        ratio_multiplier="min_norm", dual_ascent_repair=True,
        repair_order="adaptive", drift_guard=True)
    st.setdefault("corr", 0)
    st.setdefault("jumps", 0)
    st.setdefault("mono", True)
    st["critical_repairs"] = st.get("critical_transitions", 0)
    st["critical_failures"] = st.get("critical_rejections", 0)
    if st.get("status") == "CERTIFIED" or not fallback:
        return st
    xh, yh, result = _highs_fallback(B, b, d)
    _emit_trace(trace, "fallback", pivot=st.get("pivots", -1),
                reason="selector engine did not certify",
                solver_status=int(result.status), passed=xh is not None)
    st["fallback_reason"] = "selector engine did not certify"
    if xh is not None:
        st["fallback_certificate"] = result.certificate_detail
        st.update(status="CERTIFIED", x=xh, y=yh, lb=float(b @ yh),
                  backend="HiGHS dual-simplex fallback")
        return st
    status = "fallback failed"
    if result.status == 2:
        status = "INFEASIBLE"
    elif result.status == 3:
        status = "UNBOUNDED"
    st.update(status=status, backend="HiGHS dual-simplex fallback")
    return st


def follow(B, b, d, t=1.0, maxpiv=MAXPIV, tmax=TMAX, trace=None,
           fallback=True, resync_attempts=4, max_critical_repairs=20,
           use_path_factor=False, path_shadow=False, path_shadow_tol=1e-8,
           path_factor_mode="qr",
           primal_face_pricing=False, primal_face_pricing_candidates=12,
           primal_face_pricing_rounds=30,
           lexicographic_center_escape=False,
           lexicographic_center_candidates=12,
           lexicographic_center_rounds=30,
           simplex_crossover=True, polish_fallback=False,
           polish_maxiter=5000, objective_level_escape=False,
           max_objective_level_jumps=4, objective_level_fraction=0.05,
           objective_level_attempts=16,
           objective_level_min_fraction=1e-8,
           objective_level_maxiter=2000, progress=None, engine="selector"):
    """Follow the dual path.

    ``trace`` may be a list or a callable.  When supplied, every initialization
    and resynchronization attempt emits a structured record with ``t``, support
    sizes, settle outcome, relative dual residual, and certificate reason.
    The default safety net first attempts an explicitly labelled supporting-
    face ``SIMPLEX_CROSSOVER`` when optional ``highspy`` is available and a
    certified projection point was retained.  It then uses the direct HiGHS LP
    fallback on any unavailable or failed crossover.  Neither branch counts as
    Mangasarian-path success.  With ``polish_fallback=True``, a certified
    fallback pair is used to identify the complementary optimal face and the
    sparse active-set projector restores Mangasarian's minimum-norm dual
    selector.  A failed polish returns the original certified fallback pair.
    With ``objective_level_escape=True``, a certified projection whose working
    face cannot settle may impose a temporary higher ``b'y`` level.  Its
    multiplier must produce a forward ``t`` and independently certify a future
    point of the original unsliced projection path before ordinary walking
    resumes.  This branch precedes every original-LP fallback.
    With ``lexicographic_center_escape=True``, a failed breakpoint face is
    followed at fixed ``t`` through one global objective-center perturbation
    ladder.  ``B`` and its persistent factor remain exact; the transition is
    accepted only after returning to the unperturbed center and passing the
    original fixed-t and forward-affine KKT gates.
    Set ``fallback=False`` to retain path-only mode.

    PROMOTION (2026-08-06): ``engine`` selects the walking engine.  The
    default ``"selector"`` dispatches to the promoted selector-gated walker
    (``selector_gated_walker.follow_selector_gated``: adaptive repair order,
    M5 dual-ascent repair, drift guard, min-norm ratio multiplier; exact
    certificates unchanged).  Classic-only options (``use_path_factor``,
    ``primal_face_pricing``, ``lexicographic_center_escape``,
    ``objective_level_escape``, ``polish_fallback``, ``simplex_crossover``,
    ``resync_attempts``, ``max_critical_repairs``, ``path_shadow`` and the
    associated tuning knobs) are accepted but ignored by that engine;
    ``fallback`` keeps its meaning for both engines.  ``engine="classic"``
    preserves the historical path unchanged.  Record: agent_reports/11.
    """
    if engine == "selector":
        return _follow_selector_engine(
            B, b, d, t=t, maxpiv=maxpiv, tmax=tmax, trace=trace,
            progress=progress, fallback=fallback)
    if engine != "classic":
        raise ValueError("engine must be 'selector' or 'classic'")
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    if B.ndim != 2:
        raise ValueError("B must be a two-dimensional array")
    n, m = B.shape
    if n == 0 or m == 0 or b.shape != (n,) or d.shape != (m,):
        raise ValueError("expected B.shape == (len(b), len(d)) with nonzero dimensions")
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(b))
            and np.all(np.isfinite(d))):
        raise ValueError("B, b, and d must contain only finite values")
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("t must be finite and strictly positive")
    if not np.isfinite(tmax) or tmax < t:
        raise ValueError("tmax must be finite and no smaller than t")
    if path_factor_mode not in ("qr", "skeleton"):
        raise ValueError("path_factor_mode must be 'qr' or 'skeleton'")
    if (maxpiv < 0 or resync_attempts < 1 or max_critical_repairs < 0
            or primal_face_pricing_candidates < 1
            or primal_face_pricing_rounds < 1
            or lexicographic_center_candidates < 1
            or lexicographic_center_rounds < 1
            or polish_maxiter < 1 or max_objective_level_jumps < 0
            or objective_level_attempts < 1 or objective_level_maxiter < 1
            or not np.isfinite(objective_level_fraction)
            or objective_level_fraction <= 0.0
            or not np.isfinite(objective_level_min_fraction)
            or objective_level_min_fraction <= 0.0):
        raise ValueError("iteration limits must be nonnegative and resync attempts positive")
    st = dict(corr=0, jumps=0, mono=True, critical_repairs=0,
              critical_failures=0, path_factor_fallbacks=0,
              path_factor_condition_fallbacks=0,
              path_shadow_failures=0, path_shadow_max_error=0.0)
    st.update(simplex_crossover_attempts=0, simplex_crossover_failures=0)
    st.update(fallback_polish_attempts=0, fallback_polish_successes=0,
              fallback_polish_failures=0, fallback_polished=False)
    st.update(objective_level_jump_calls=0,
              objective_level_jump_successes=0,
              objective_level_jump_failures=0,
              objective_level_jump_backtracks=0,
              objective_level_jump_infeasible_slices=0,
              objective_level_jump_handoff_failures=0)
    st.update(primal_pricing_calls=0, primal_pricing_affine_lp_calls=0,
              primal_pricing_affine_accepts=0,
              primal_pricing_candidate_evaluations=0,
              primal_pricing_steps=0,
              primal_pricing_objective_descent_steps=0,
              primal_pricing_settled=0,
              primal_pricing_cycle_failures=0,
              primal_pricing_no_proposal_failures=0,
              primal_pricing_exhausted_failures=0,
              primal_pricing_round_cap_failures=0)
    st.update(lexicographic_center_calls=0,
              lexicographic_center_successes=0,
              lexicographic_center_failures=0,
              lexicographic_center_face_changes=0)

    def record_progress(stage, pivot=-1, mask=None, coefficients=None,
                        certified_face=False, **fields):
        """Expose interrupt-safe diagnostic state without affecting decisions."""
        if progress is None:
            return
        record = dict(stage=str(stage), pivot=int(pivot), t=float(t))
        if mask is not None:
            mask_copy = np.asarray(mask, dtype=bool).copy()
            record["working_set"] = mask_copy
            record["support"] = int(mask_copy.sum())
        if coefficients is not None:
            record["coefficients"] = tuple(
                np.asarray(value, dtype=float).copy()
                if isinstance(value, np.ndarray) else value
                for value in coefficients)
        record.update(fields)
        progress["updates"] = int(progress.get("updates", 0)) + 1
        progress["current"] = record
        if certified_face:
            progress["last_certified_face"] = record
    path_factor = None
    path_factor_active = False
    path_condition_error = None
    if use_path_factor:
        try:
            from path_piece_factor import (PathPieceConditionError,
                                           PathPieceFactor)
            if path_factor_mode == "skeleton":
                from path_piece_skeleton import PathPieceSkeletonFactor
                path_factor = PathPieceSkeletonFactor(B, b, d)
            else:
                path_factor = PathPieceFactor(B, b, d)
                path_condition_error = PathPieceConditionError
            path_factor_active = True
        except (ImportError, ValueError, np.linalg.LinAlgError) as error:
            st["path_factor_fallbacks"] += 1
            _emit_trace(trace, "path factor disabled", reason=type(error).__name__)

    def solve_path_piece(mask):
        nonlocal path_factor_active
        reference = None
        if path_factor_active:
            try:
                candidate = path_factor.solve(mask)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                if (path_condition_error is not None
                        and isinstance(error, path_condition_error)):
                    st["path_factor_condition_fallbacks"] += 1
                    _emit_trace(trace, "path factor face fallback",
                                reason="conditioning", detail=str(error),
                                support=int(np.sum(mask)),
                                rank=int(path_factor.rank))
                    return piece_y(B, b, d, mask)
                path_factor_active = False
                st["path_factor_fallbacks"] += 1
                _emit_trace(trace, "path factor disabled",
                            reason=type(error).__name__, detail=str(error))
            else:
                if path_shadow:
                    reference = piece_y(B, b, d, mask)
                    errors = []
                    for proposed, baseline in zip(candidate[:4], reference[:4]):
                        errors.append(float(np.max(
                            np.abs(proposed - baseline) / (1.0 + np.abs(baseline)))))
                    error = max(max(errors, default=0.0),
                                abs(candidate[4] - reference[4]))
                    st["path_shadow_max_error"] = max(
                        st["path_shadow_max_error"], error)
                    _emit_trace(trace, "path factor shadow",
                                support=int(np.sum(mask)), error=error,
                                rank=int(path_factor.rank))
                    if error > path_shadow_tol:
                        path_factor_active = False
                        st["path_shadow_failures"] += 1
                        _emit_trace(trace, "path factor disabled",
                                    reason="shadow mismatch", error=error)
                        return reference
                return candidate
        return (reference if reference is not None
                else piece_y(B, b, d, mask))

    def settle_path(mask, parameter, rounds=20, diagnostics=None):
        """Run the historical settle, then the opt-in priced escape."""
        detail = diagnostics if diagnostics is not None else {}
        original_mask = np.asarray(mask, dtype=bool).copy()
        settled_mask, coefficients, settled = settle(
            B, b, d, mask, parameter, rounds=rounds,
            diagnostics=detail, piece_solver=solve_path_piece)
        if settled or not primal_face_pricing:
            return settled_mask, coefficients, settled
        baseline_reason = detail.get("reason")
        baseline_rounds = detail.get("rounds")
        try:
            from primal_face_pricing import priced_settle_escape
            priced_mask, priced_coefficients, priced = priced_settle_escape(
                B, b, d, original_mask, parameter,
                piece_solver=solve_path_piece,
                max_candidates=primal_face_pricing_candidates,
                rounds=primal_face_pricing_rounds,
                diagnostics=detail, stats=st, tol=TOL_FEAS)
        except (ImportError, ValueError, RuntimeError,
                np.linalg.LinAlgError) as error:
            detail.update(pricing_used=True,
                          pricing_exception=type(error).__name__)
            return settled_mask, coefficients, False
        detail["baseline_settle_reason"] = baseline_reason
        detail["baseline_settle_rounds"] = baseline_rounds
        return priced_mask, priced_coefficients, priced

    corrector_detail = {}
    W = qp_corrector(B, b, d, t, diagnostics=corrector_detail); st["corr"] += 1
    last_projected_y = corrector_detail.get("projection_y")
    last_projected_t = float(t) if last_projected_y is not None else None
    raw_size = int(W.sum())
    record_progress("initial settle attempt", mask=W,
                    corrector_backend=corrector_detail.get("backend"))
    settle_detail = {}
    W, pc, ok = settle_path(W, t, diagnostics=settle_detail)
    record_progress("initialization", mask=W, coefficients=pc,
                    certified_face=bool(ok), settle_ok=bool(ok),
                    settle_reason=settle_detail.get("reason"))
    dres, cert_reason = _support_diagnostics(
        B, b, d, W, t, piece_solver=solve_path_piece)
    _emit_trace(trace, "initialization", attempt=0, t=float(t),
                raw_support=raw_size, support=int(W.sum()), settle_ok=bool(ok),
                settle_reason=settle_detail.get("reason"),
                settle_rounds=settle_detail.get("rounds"),
                pricing_used=settle_detail.get("pricing_used", False),
                pricing_steps=settle_detail.get("pricing_steps", 0),
                pricing_endpoint_error=settle_detail.get(
                    "pricing_endpoint_error"),
                dual_residual=float(dres), certificate_reason=cert_reason)
    prev_lb = -np.inf

    def out(k, status, x, lb, y=None, backend="dual path"):
        if (st["objective_level_jump_successes"]
                and backend.startswith("dual path")
                and "OBJECTIVE_LEVEL_JUMP" not in backend):
            backend += " + OBJECTIVE_LEVEL_JUMP"
        if (st["lexicographic_center_successes"]
                and backend.startswith("dual path")
                and "LEXICOGRAPHIC_CENTER" not in backend):
            backend += " + LEXICOGRAPHIC_CENTER"
        st.update(pivots=k, status=status, x=x, y=y, lb=lb, t=t,
                  backend=backend)
        st["path_factor_active"] = bool(path_factor_active)
        st["path_factor_mode"] = path_factor_mode
        if path_factor is not None:
            st.update(path_factor.diagnostics())
        record_progress("return", pivot=k, mask=W, coefficients=pc,
                        certified_face=bool(ok and pc is not None),
                        status=status, backend=backend)
        return st

    def polish_pair(x, y, backend, k, fallback_reason):
        """Fail closed to an already-certified fallback pair."""
        if not polish_fallback:
            return x, y, backend
        st["fallback_polish_attempts"] += 1
        try:
            from optimal_face_polish import polish_dual_optimum
            polished = polish_dual_optimum(
                B, b, d, y, x0=x, maxiter=polish_maxiter)
        except Exception as error:
            polished = {"status": "EXCEPTION",
                        "exception": type(error).__name__}
        detail_keys = (
            "status", "projection_status", "iterations", "crash_status",
            "crash_iterations", "crash_backtracked", "face_mode",
            "face_coordinates", "fixed_zero_coordinates",
            "selected_slack_error", "initial_norm", "polished_norm",
            "norm_reduction", "relative_norm_reduction", "face_error",
            "dual_error", "objective_error", "nonnegative_error",
            "secondary_kkt_error", "lp_certificate_pass", "exception",
        )
        polish_detail = {key: polished.get(key) for key in detail_keys
                         if polished.get(key) is not None}
        _emit_trace(trace, "fallback polish", pivot=k,
                    fallback_reason=fallback_reason,
                    source_backend=backend, **polish_detail)
        st["fallback_polish_detail"] = polish_detail
        if polished.get("status") == "POLISHED":
            polished_y = np.asarray(polished["y"], dtype=float)
            pair_ok, pair_detail = certificate_pair(B, b, d, x, polished_y)
            polish_detail["final_lp_certificate"] = pair_detail
            if pair_ok:
                st["fallback_polish_successes"] += 1
                st["fallback_polished"] = True
                return x, polished_y, backend + " + MANGASARIAN_POLISH"
            polish_detail["status"] = "FINAL LP CERTIFICATE FAILED"
        st["fallback_polish_failures"] += 1
        return x, y, backend

    def fallback_out(k, reason):
        # REVISIT: replace this ordinary-simplex escape with a certified
        # coalesced transition on the genuine Mangasarian projection path.
        # Keep its backend separate in all claims and benchmark summaries.
        if (simplex_crossover and last_projected_y is not None
                and last_projected_t is not None):
            st["simplex_crossover_attempts"] += 1
            try:
                from supporting_face_crossover import supporting_face_crossover
                crossed = supporting_face_crossover(
                    B, b, d, last_projected_y, last_projected_t)
            except Exception as error:
                crossed = {"status": "UNAVAILABLE",
                           "reason": "crossover wrapper exception",
                           "exception": type(error).__name__}
            crossover_detail = dict(crossed.get("detail", {}))
            crossover_detail.update(
                status=crossed.get("status"),
                reason=crossed.get("reason"),
                projection_t=float(last_projected_t))
            _emit_trace(trace, "simplex crossover", pivot=k,
                        fallback_reason=reason, **crossover_detail)
            if crossed.get("status") == "CANDIDATE":
                pair_ok, pair_detail = certificate_pair(
                    B, b, d, crossed["x"], crossed["y"])
                crossover_detail["certificate"] = pair_detail
                if pair_ok:
                    st["fallback_reason"] = reason
                    st["simplex_crossover_detail"] = crossover_detail
                    polished_x, polished_y, polished_backend = polish_pair(
                        crossed["x"], crossed["y"],
                        "SIMPLEX_CROSSOVER (supporting-face crash)", k,
                        reason)
                    return out(
                        k, "CERTIFIED", polished_x,
                        float(b @ polished_y), polished_y,
                        backend=polished_backend)
                crossover_detail["status"] = "CERTIFICATE FAILED"
            st["simplex_crossover_failures"] += 1
            st["simplex_crossover_detail"] = crossover_detail

        xh, yh, result = _highs_fallback(B, b, d)
        _emit_trace(trace, "fallback", pivot=k, reason=reason,
                    solver_status=int(result.status), passed=xh is not None)
        st["fallback_reason"] = reason
        if xh is not None:
            st["fallback_certificate"] = result.certificate_detail
            polished_x, polished_y, polished_backend = polish_pair(
                xh, yh, "HiGHS dual-simplex fallback", k, reason)
            return out(k, "CERTIFIED", polished_x,
                       float(b @ polished_y), polished_y,
                       backend=polished_backend)
        status = "fallback failed"
        if result.status == 2:
            status = "INFEASIBLE"
        elif result.status == 3:
            status = "UNBOUNDED"
        return out(k, status, None, prev_lb,
                   backend="HiGHS dual-simplex fallback")

    for k in range(maxpiv):
        if not ok:
            record_progress("resync required", pivot=k, mask=W,
                            settle_ok=False)
            # resync: re-solve (QP_t) at current t; escalate t if that fails
            # Historical path-only mode exhausts many resynchronizations.  The
            # explicitly bounded objective-level escape must honor the caller's
            # stated limit even when original-LP fallback is disabled; otherwise
            # a failed exceptional jump silently turns into a 60-corrector churn.
            attempts = (resync_attempts
                        if (fallback or objective_level_escape) else 60)
            for att in range(attempts):
                corrector_detail = {}
                W = qp_corrector(B, b, d, t, attempt=att,
                                 active_set_repair=(att == 0),
                                 diagnostics=corrector_detail,
                                 retain_projection_working_set=(
                                     primal_face_pricing and att == 0))
                st["corr"] += 1
                projected_y = corrector_detail.get("projection_y")
                if projected_y is not None:
                    last_projected_y = np.asarray(projected_y).copy()
                    last_projected_t = float(t)
                    projected_lp, projected_x, _, projected_reason = (
                        _certificate_detail(B, b, d, projected_y))
                    _emit_trace(trace, "projected LP certificate", pivot=k,
                                attempt=att, t=float(t),
                                passed=bool(projected_lp),
                                reason=projected_reason)
                    if projected_lp:
                        return out(k, "CERTIFIED", projected_x,
                                   float(b @ projected_y), projected_y,
                                   backend="dual path with tangent projection repair")
                raw_size = int(W.sum())
                record_progress("resync settle attempt", pivot=k, mask=W,
                                attempt=int(att),
                                corrector_backend=corrector_detail.get(
                                    "backend"))
                settle_detail = {}
                W, pc, ok = settle_path(W, t, diagnostics=settle_detail)
                record_progress("resync", pivot=k, mask=W,
                                coefficients=pc, certified_face=bool(ok),
                                attempt=int(att), settle_ok=bool(ok),
                                settle_reason=settle_detail.get("reason"))
                dres, cert_reason = _support_diagnostics(
                    B, b, d, W, t, piece_solver=solve_path_piece)
                _emit_trace(trace, "resync", attempt=att, t=float(t),
                            raw_support=raw_size, support=int(W.sum()),
                            settle_ok=bool(ok),
                            settle_reason=settle_detail.get("reason"),
                            settle_rounds=settle_detail.get("rounds"),
                            pricing_used=settle_detail.get(
                                "pricing_used", False),
                            pricing_steps=settle_detail.get(
                                "pricing_steps", 0),
                            pricing_endpoint_error=settle_detail.get(
                                "pricing_endpoint_error"),
                            dual_residual=float(dres),
                            certificate_reason=cert_reason,
                            corrector_backend=corrector_detail.get("backend"),
                            projection_status=corrector_detail.get("projection_status"),
                            projection_iterations=corrector_detail.get("projection_iterations"),
                            projection_perturbations=corrector_detail.get("projection_perturbations"),
                            projection_tangent_escapes=corrector_detail.get("tangent_escapes"),
                            projection_basis_refactorizations=corrector_detail.get(
                                "projection_basis_refactorizations"),
                            projection_basis_updates=corrector_detail.get(
                                "projection_basis_updates"),
                            projection_sparse_kkt_repairs=corrector_detail.get(
                                "projection_sparse_kkt_repairs"),
                            projection_working_set_size=corrector_detail.get(
                                "projection_working_set_size"),
                            projection_error=corrector_detail.get("projection_error"))
                if (not ok and objective_level_escape
                        and projected_y is not None
                        and (st["objective_level_jump_calls"]
                             < max_objective_level_jumps)):
                    level = float(b @ projected_y)
                    initial_delta = (objective_level_fraction
                                     * max(1.0, abs(level)))
                    minimum_delta = (objective_level_min_fraction
                                     * max(1.0, abs(level)))
                    st["objective_level_jump_calls"] += 1
                    try:
                        from objective_level_jump import (
                            objective_level_jump_backtracking)
                        level_jump = objective_level_jump_backtracking(
                            B, b, d, projected_y, t,
                            initial_delta=initial_delta,
                            attempts=objective_level_attempts,
                            min_delta=minimum_delta,
                            maxiter=objective_level_maxiter)
                    except Exception as error:
                        level_jump = {
                            "status": "EXCEPTION",
                            "attempt": 1,
                            "exception": type(error).__name__,
                        }
                    st["objective_level_jump_backtracks"] += int(
                        level_jump.get("infeasible_backtracks", 0))
                    st["objective_level_jump_infeasible_slices"] += int(
                        level_jump.get("infeasible_slices", 0))
                    jump_detail = {
                        "status": level_jump.get("status"),
                        "attempt": level_jump.get("attempt"),
                        "old_t": float(t),
                        "new_t": level_jump.get("new_t"),
                        "old_level": level,
                        "new_level": level_jump.get("new_level"),
                        "delta": level_jump.get("delta"),
                        "movement": level_jump.get("movement"),
                        "level_error": level_jump.get("level_error"),
                        "path_certificate_error": level_jump.get(
                            "path_certificate_error"),
                        "slice_iterations": level_jump.get("slice_iterations"),
                        "slice_status": level_jump.get("slice_status"),
                        "minimum_delta": level_jump.get("minimum_delta"),
                        "infeasible_slices": level_jump.get(
                            "infeasible_slices"),
                        "infeasible_backtracks": level_jump.get(
                            "infeasible_backtracks"),
                        "backtracking_stop_reason": level_jump.get(
                            "backtracking_stop_reason"),
                        "exception": level_jump.get("exception"),
                    }
                    if (level_jump.get("status") == "JUMPED"
                            and float(level_jump["new_t"]) <= tmax):
                        jump_t = float(level_jump["new_t"])
                        jump_y = np.asarray(level_jump["y"], dtype=float)
                        jump_W = np.asarray(
                            level_jump["working_set"], dtype=bool).copy()
                        jump_settle_detail = {}
                        jump_W, jump_pc, jump_ok = settle_path(
                            jump_W, jump_t, rounds=50,
                            diagnostics=jump_settle_detail)
                        reconstruction_error = np.inf
                        if jump_ok and jump_pc is not None:
                            jump_g, jump_h, _jump_ua, _jump_uc = jump_pc
                            reconstructed = np.zeros(n)
                            reconstructed[jump_W] = (
                                jump_t * jump_g + jump_h)
                            reconstruction_error = float(np.max(
                                np.abs(reconstructed - jump_y)
                                / (1.0 + np.abs(jump_y))))
                            jump_ok = reconstruction_error <= TOL_FEAS
                        jump_detail.update(
                            handoff_settled=bool(jump_ok),
                            handoff_reason=jump_settle_detail.get("reason"),
                            handoff_rounds=jump_settle_detail.get("rounds"),
                            handoff_reconstruction_error=reconstruction_error,
                            working_set=int(np.sum(jump_W)))
                        if jump_ok:
                            t, W, pc, ok = jump_t, jump_W, jump_pc, True
                            record_progress(
                                "objective level jump", pivot=k, mask=W,
                                coefficients=pc, certified_face=True,
                                settle_ok=True)
                            last_projected_y = jump_y.copy()
                            last_projected_t = jump_t
                            st["objective_level_jump_successes"] += 1
                            st["objective_level_jump_detail"] = jump_detail
                            _emit_trace(trace, "objective level jump",
                                        pivot=k, **jump_detail)
                            break
                        st["objective_level_jump_handoff_failures"] += 1
                    st["objective_level_jump_failures"] += 1
                    st["objective_level_jump_detail"] = jump_detail
                    _emit_trace(trace, "objective level jump failed",
                                pivot=k, **jump_detail)
                if ok:
                    break
                t = 2.0 * t + 1.0; st["jumps"] += 1
                if t > tmax:
                    if fallback:
                        return fallback_out(k, "resynchronization reached t cap")
                    return out(k, "t cap (resync)", None, prev_lb)
            if not ok:
                if fallback:
                    return fallback_out(k, "resynchronization did not settle")
                return out(k, "resync failed", None, prev_lb)

        g, h, ua, uc = pc
        record_progress("affine piece", pivot=k, mask=W,
                        coefficients=pc, certified_face=True,
                        settle_ok=True)
        idxW = np.where(W)[0]
        y = np.zeros(n)
        y[idxW] = np.maximum(t * g + h, 0.0)
        lb = float(b @ y)
        if lb < prev_lb - 1e-7 * max(1.0, abs(lb)):
            st["mono"] = False
        prev_lb = lb
        # Certificate gate: on the final piece y(t) is constant (the min-norm
        # point of the optimal dual face), so g = 0 there exactly; mid-path
        # ||g||^2 is the slope of b'y(t) and is strictly positive.  ||g|| ~ 0
        # is therefore a NECESSARY condition for optimality -- only pay for
        # the full KKT certificate when it holds.
        gate = float(g @ g) <= 1e-12 * max(1.0, float(b[idxW] @ b[idxW]))
        if gate:
            # On a terminal affine piece g=0, the stable dual is h and the
            # primal LP point is the slope of the equality multiplier, -ua.
            # Certify this pair directly instead of forming t*g+h, whose tiny
            # slope error is catastrophically magnified when t is large.
            terminal_y = np.zeros(n)
            terminal_y[idxW] = h
            terminal_x = -ua
            terminal_ok, terminal_detail = certificate_pair(
                B, b, d, terminal_x, terminal_y)
            _emit_trace(trace, "terminal pair certificate", pivot=k,
                        t=float(t), support=int(W.sum()),
                        passed=bool(terminal_ok), **terminal_detail)
            if terminal_ok:
                return out(k, "CERTIFIED", terminal_x,
                           float(b @ terminal_y), terminal_y,
                           backend="dual path with stabilized terminal pair")
            okc, x, _, cert_reason = _certificate_detail(B, b, d, y)
            _emit_trace(trace, "certificate", pivot=k, t=float(t),
                        support=int(W.sum()), gate=True, passed=bool(okc),
                        reason=cert_reason)
            if okc:
                return out(k, "CERTIFIED", x, lb, y)

        # breakpoints: uniform event array  v_i(t) = s_i*t + c_i  must stay >= 0
        s = np.empty(n); c = np.empty(n)
        s[idxW] = g; c[idxW] = h                          # leaving: y_i -> 0
        off = ~W
        s[off] = -(b[off] + B[off] @ ua)                  # entering: r_i -> 0
        c[off] = -(B[off] @ uc)
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = np.where(s < -1e-13, -c / s, np.inf)
        fmask = np.isfinite(tc) & (tc > t * (1 + FWD) + FWD)
        if not fmask.any():
            # final piece but certificate failed/gated -> certify ungated as
            # insurance, then escalate t and resync
            if not gate:
                okc, x, _, cert_reason = _certificate_detail(B, b, d, y)
                _emit_trace(trace, "certificate", pivot=k, t=float(t),
                            support=int(W.sum()), gate=False, passed=bool(okc),
                            reason=cert_reason)
                if okc:
                    return out(k, "CERTIFIED", x, lb, y)
            t = 2.0 * t + 1.0; st["jumps"] += 1
            if t > tmax:
                if fallback:
                    return fallback_out(k, "no forward event and certificate failed")
                return out(k, "no event, cert fails", None, prev_lb)
            ok = False
            continue
        tn = float(tc[fmask].min())
        if tn > tmax:
            if fallback:
                return fallback_out(k, "next breakpoint exceeds t cap")
            return out(k, "t cap (breakpoint)", None, prev_lb)
        tie = fmask & (np.abs(tc - tn) <= TIE * max(1.0, tn))
        # Preserve the exact endpoint supplied by the old affine face.  If the
        # ordinary toggle cannot settle, this point—not an IPM approximation—
        # defines the critical cone used by the exceptional repair.
        y_vertex = np.zeros(n)
        y_vertex[idxW] = tn * g + h
        W = W.copy()
        W[tie] = ~W[tie]                                  # discrete toggle
        t = tn
        record_progress("breakpoint settle attempt", pivot=k, mask=W,
                        tie_count=int(tie.sum()), settle_ok=None, pending=True)
        W, pc, ok = settle_path(W, t)
        record_progress("breakpoint settle", pivot=k, mask=W,
                        coefficients=pc, certified_face=bool(ok),
                        tie_count=int(tie.sum()), settle_ok=bool(ok))
        if not ok and lexicographic_center_escape:
            st["lexicographic_center_calls"] += 1
            try:
                from lexicographic_center_homotopy import (
                    follow_center_ladder)
                lex_result = follow_center_ladder(
                    B, b, d, W, t,
                    candidates=lexicographic_center_candidates,
                    rounds=lexicographic_center_rounds, tol=TOL_FEAS,
                    seconds_per_stage=None, return_state=True)
            except (ImportError, ValueError, RuntimeError,
                    np.linalg.LinAlgError) as error:
                lex_result = {
                    "status": "EXCEPTION",
                    "exception": type(error).__name__,
                    "returned_to_original": False,
                }
            lex_detail = {
                "status": lex_result.get("status"),
                "returned_to_original": lex_result.get(
                    "returned_to_original", False),
                "working_set_changed": lex_result.get(
                    "working_set_changed", False),
                "final_diff_from_initial": lex_result.get(
                    "final_diff_from_initial"),
                "final_support": lex_result.get("final_support"),
                "stages": len(lex_result.get("records", [])),
                "exception": lex_result.get("exception"),
            }
            if (lex_result.get("status") == "CERTIFIED"
                    and lex_result.get("returned_to_original")
                    and lex_result.get("working_set") is not None
                    and lex_result.get("coefficients") is not None):
                W = np.asarray(
                    lex_result["working_set"], dtype=bool).copy()
                pc = tuple(np.asarray(value, dtype=float).copy()
                           for value in lex_result["coefficients"])
                ok = True
                st["lexicographic_center_successes"] += 1
                st["lexicographic_center_face_changes"] += int(
                    lex_result.get("final_diff_from_initial", 0))
                record_progress(
                    "lexicographic center escape", pivot=k, mask=W,
                    coefficients=pc, certified_face=True, settle_ok=True)
                _emit_trace(trace, "lexicographic center escape", pivot=k,
                            t=float(t), **lex_detail)
            else:
                st["lexicographic_center_failures"] += 1
                _emit_trace(
                    trace, "lexicographic center escape failed", pivot=k,
                    t=float(t), **lex_detail)
        if not ok and st["critical_repairs"] < max_critical_repairs:
            repaired, repair_detail = critical_vertex_escape(
                B, b, d, y_vertex, t, trace=trace,
                piece_solver=solve_path_piece, settle_solver=settle_path)
            if repaired is not None:
                W, pc, t = repaired
                ok = True
                st["critical_repairs"] += 1
            else:
                st["critical_failures"] += 1
                _emit_trace(trace, "critical escape failed", pivot=k,
                            t=float(t), **repair_detail)
    if fallback:
        return fallback_out(maxpiv, "pivot cap")
    return out(maxpiv, "pivot cap", None, prev_lb)


# ------------------------------------------- exp21 baseline (same certificate)

def newton4(B, b, d, t, itmax=300, delta=1e-6, tol=1e-14):
    m = B.shape[1]
    u = np.zeros(m)
    for _ in range(itmax):
        r = B @ u - t * b
        neg = r < 0
        if neg.sum() == 0:
            break
        Bv = B[neg]
        un = u - np.linalg.solve(Bv.T @ Bv + delta * np.eye(m), Bv.T @ r[neg] + d)
        if np.linalg.norm(un - u, np.inf) <= tol * max(1.0, np.linalg.norm(u, np.inf)):
            return un
        u = un
    return u


def follow21(B, b, d, t0=1.0, maxpiv=400):
    """exp21's u-space follower (sign re-evaluation at a nudged t), with the
    same full KKT certificate as exp23 so the comparison is apples-to-apples."""
    u = newton4(B, b, d, t0)
    V = (B @ u - t0 * b) < 0
    t, prev_lb, mono = t0, -np.inf, True
    for k in range(maxpiv):
        if V.sum() == 0:
            return k, "V empty", mono
        BV = B[V]
        alpha, *_ = np.linalg.lstsq(BV, b[V], rcond=None)
        beta, *_ = np.linalg.lstsq(BV.T @ BV, -d, rcond=None)
        p = B @ alpha - b
        q = B @ beta
        y = np.maximum(-(t * p + q), 0.0)
        lb = float(b @ y)
        if lb < prev_lb - 1e-7 * max(1.0, abs(lb)):
            mono = False
        prev_lb = lb
        okc, *_ = certificate(B, b, d, y)
        if okc:
            return k, "CERTIFIED", mono
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = np.where(np.abs(p) > 1e-14, -q / p, np.inf)
        fwd = tc[tc > t * (1 + 1e-12) + 1e-12]
        if fwd.size == 0:
            return k, "no fwd bp", mono
        tn = float(fwd.min())
        tp = tn * (1 + 1e-9) + 1e-9
        Vn = (tp * p + q) < 0
        if np.array_equal(Vn, V):
            return k, "stall", mono
        V, t = Vn, tp
    return maxpiv, "pivot cap", mono


# ---------------------------------------------------------------- experiments

def sweep(m, S=20, ks=(0, 1, 3, 10), alpha_shape=3.0, margin=2e-1,
          with_baseline=False):
    print(f"\n### m={m}, n={int(alpha_shape*m)}, alpha={alpha_shape}, "
          f"margin={margin}, {S} seeds (same seeds as exp22)")
    hdr = (f"{'k=|A|-m':>8} | {'certified':>9} {'objerr(max)':>11} {'piv med':>7} "
           f"{'piv max':>7} {'corr med':>8} {'jumps':>5} {'monotone':>8} | exit")
    print(hdr)
    for kk in ks:
        a = (m + kk) / m
        cert = mono = 0
        piv, corr, jtot, why = [], [], 0, {}
        objerr = 0.0
        base = {"CERTIFIED": 0}
        for sd in range(S):
            rng = np.random.default_rng(9000 + 37 * sd)
            inst = lpgen2(m, alpha_shape, a, margin, rng)
            B, b, d = inst["B"], inst["b"], inst["d"]
            res = linprog(d, A_ub=-B, b_ub=-b, bounds=[(None, None)] * m,
                          method="highs")
            fstar = res.fun
            o = follow(B, b, d)
            piv.append(o["pivots"]); corr.append(o["corr"]); jtot += o["jumps"]
            why[o["status"]] = why.get(o["status"], 0) + 1
            mono += o["mono"]
            if o["status"] == "CERTIFIED":
                cert += 1
                objerr = max(objerr, abs(float(d @ o["x"]) - fstar)
                             / max(1.0, abs(fstar)))
            if with_baseline:
                _, w21, _ = follow21(B, b, d)
                base[w21] = base.get(w21, 0) + 1
        line = (f"{kk:>8d} | {cert:>6d}/{S} {objerr:>11.1e} "
                f"{int(np.median(piv)):>7d} {max(piv):>7d} "
                f"{int(np.median(corr)):>8d} {jtot:>5d} {mono:>5d}/{S} | {why}")
        print(line)
        if with_baseline:
            print(f"{'':>8}   exp21 baseline (same seeds, same certificate): "
                  f"{base.get('CERTIFIED', 0)}/{S} certified, exits {base}")


def run_netlib(name, fallback=True):
    from exp11_netlib_rank_certificate import to_Bxgeb
    netlib_path = Path(__file__).resolve().parent.parent / "netlib" / f"{name}.mps"
    B, b, d, _ = to_Bxgeb(netlib_path)
    n, m = B.shape
    res = linprog(d, A_ub=-B, b_ub=-b, bounds=[(None, None)] * m, method="highs")
    print(f"\n### netlib {name}: n={n} constraints, m={m} variables, "
          f"f*={res.fun:.9g}")
    o = follow(B, b, d, maxpiv=4000, fallback=fallback)
    msg = (f"  status={o['status']}  pivots={o['pivots']}  corr={o['corr']}  "
           f"jumps={o['jumps']}  critical={o['critical_repairs']}/"
           f"{o['critical_failures']}  monotone={o['mono']}  "
           f"final t={o['t']:.3g}  backend={o['backend']}")
    if o["status"] == "CERTIFIED":
        gap = float(d @ o["x"]) - o["lb"]
        msg += (f"\n  d'x={float(d @ o['x']):.9g}  b'y={o['lb']:.9g}  "
                f"gap={gap:.2e}  |d'x - f*|={abs(float(d @ o['x'])-res.fun):.2e}")
    else:
        msg += f"\n  best dual lower bound b'y={o['lb']:.9g}  (f*={res.fun:.9g})"
    print(msg)


if __name__ == "__main__":
    sweep(40, with_baseline=True)
    sweep(80)
    # afiro is a tiny calibration/smoke test; degen2 (n=1199, m=534, with
    # rank(B_V) < m routine) is the honest real-matrix test.
    for nm in ("afiro", "degen2"):
        run_netlib(nm)
