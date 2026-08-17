"""Decoupled LP generator + instrumented Theorem-8 solver.

WHY THIS EXISTS
---------------
The papers' lpgen confounds the two things we need to separate.  It plants ~phi*n
active constraints, so with phi = 1/2 every instance with alpha = n/m > 2 has MORE
ACTIVE CONSTRAINTS THAN VARIABLES -- degenerate by construction.  "Hard shape" and
"degenerate vertex" are the same knob, and no sweep over alpha can tell them apart.

lpgen2 below gives three independent knobs:

    alpha  = n/m                    shape
    a      = |active| / m           degeneracy   (a<1 primal non-unique at the optimum;
                                                  a=1 nondegenerate vertex;
                                                  a>1 degenerate vertex, dual non-unique
                                                      -- Mangasarian's "strong uniqueness"
                                                      condition (18) fails)
    margin = planted strict-complementarity gap  (min multiplier on the active set and
                                                  min slack off it)

The instrumented solver does the repair and the diagnosis with one instrument: the
guard that decides whether finite termination is safe IS the obstruction detector.

    1. stop on the TRUE fixed-point residual ||F(w)-w||, not on the step size.
       The shipped rule ||w_k - w||/max(1,||w||) < eps is a step test; it fires at
       residual ~ eps*rho/(1-rho), so as rho -> 1 it silently returns answers 3-5
       orders worse than eps advertises.  ||F(w)-w|| is already computed every
       iteration and costs nothing.
    2. track J+ = supp((z)_+) per iteration; declare identification when it holds
       still for `stab` iterations.
    3. on identification solve the affine system (I - M D_J) z = c on the face.
    4. guard on sign consistency + residual; accept, or fall back to iterating.

Logged per run: identification iteration, margin delta = min|z_i|/||z||, flats
= max(0, dimL + |J+| - n), |J+| against its two thresholds, and the recovered LP
residuals.
"""

import numpy as np


# ----------------------------------------------------------------- generator

def lpgen2(m, alpha, a, margin, rng, slack_hi=10.0):
    """LP  min d'x s.t. Bx >= b  with a planted, KKT-verified optimum.

    |active| = round(a*m), clipped to [1, n].  Active multipliers are drawn from
    U(margin, 1) and inactive slacks from U(margin, slack_hi), so `margin` directly
    sets the strict-complementarity gap: margin -> 0 makes complementarity weak.
    """
    n = int(round(alpha * m))
    k = int(np.clip(round(a * m), 1, n))
    B = rng.uniform(-50, 50, (n, m))
    xs = rng.standard_normal(m)
    J = rng.choice(n, k, replace=False)
    off = np.ones(n, bool); off[J] = False
    s = np.zeros(n)
    s[off] = rng.uniform(margin, slack_hi, off.sum())
    b = B @ xs - s
    yJ = rng.uniform(margin, 1.0, k)
    d = B[J].T @ yJ                       # KKT: B_J' y_J = d, y >= 0, complementary
    return dict(B=B, b=b, d=d, xs=xs, J=J, n=n, m=m, k=k)


def build1(B, b, d, t):
    """Theorem 8 item 1 (primal recovery). L = ker(B'), dim n-m."""
    n = B.shape[0]
    S = np.linalg.inv(np.eye(n) + B @ B.T)
    q = S @ (t * (B @ d) + b)
    return S, q


def build2(B, b, d, t):
    """Theorem 8 item 2 (dual recovery). L = range(B), dim m."""
    Pm = np.linalg.solve(B.T @ B, B.T)
    R = B @ Pm
    h = Pm.T @ d + t * b
    r = (R @ h - h) - Pm.T @ d
    return R, r, Pm


# ------------------------------------------------------------------- solver

def solve(M, c, dimL, eps=1e-9, K=8000, stab=40, scheme="averaged",
          do_face=True, face_tol=1e-6, supp_tol=0.0):
    """Instrumented Theorem-8 iteration with guarded finite termination.

    scheme: 'averaged' = the paper's z_k = z_{k-1}/(k+1) + (k/(k+1))F(z_{k-1});
            'picard'   = z_k = F(z_{k-1}).
    Returns a dict of diagnostics.  `iters` counts map applications either way, so
    the face-solve and no-face runs are directly comparable.
    """
    n = M.shape[0]
    w = np.zeros(n)
    prev_supp = None
    stable = 0
    out = dict(ident_k=None, face_tries=0, face_ok=False, hit_cap=False)

    # NB: the penalty data scales with t (c carries t*B*d and t*b), so the residual
    # MUST be relative or the test silently demands t-times more precision as t grows.
    # An absolute test here produced a spurious "the law fails at t >= t_0" in exp07.
    cscale = max(1.0, float(np.linalg.norm(c)))

    for k in range(1, K + 1):
        pw = np.maximum(w, 0.0)
        fw = M @ pw + c
        res = np.linalg.norm(fw - w) / cscale   # TRUE fixed-point residual, RELATIVE
        if res < eps:
            out.update(iters=k, res=res, how="iteration")
            return _finish(out, w, M, c, dimL, n)

        supp = w > supp_tol
        if prev_supp is not None and np.array_equal(supp, prev_supp):
            stable += 1
        else:
            stable = 0
        prev_supp = supp

        if do_face and stable == stab:
            out["face_tries"] += 1
            z, ok = _face_solve(M, c, supp, face_tol)
            if ok:
                out.update(iters=k, res=np.linalg.norm(M @ np.maximum(z, 0.0) + c - z) / cscale,
                           how="face", ident_k=k, face_ok=True)
                return _finish(out, z, M, c, dimL, n)

        w = fw if scheme == "picard" else w / (k + 1) + (k / (k + 1)) * fw

    out.update(iters=K, res=np.linalg.norm(M @ np.maximum(w, 0.0) + c - w) / cscale,
               how="cap", hit_cap=True)
    return _finish(out, w, M, c, dimL, n)


def _face_solve(M, c, supp, face_tol):
    """Solve (I - M D_J) z = c, then GUARD: sign consistency + residual."""
    n = M.shape[0]
    D = np.zeros((n, n))
    idx = np.where(supp)[0]
    D[idx, idx] = 1.0
    z, *_ = np.linalg.lstsq(np.eye(n) - M @ D, c, rcond=None)
    res = np.linalg.norm(M @ np.maximum(z, 0.0) + c - z)
    scale = max(1.0, np.linalg.norm(z))
    signs_ok = bool(np.all(z[idx] > 0) and np.all(z[~supp] <= 0))
    return z, (signs_ok and res / scale < face_tol)


def _finish(out, z, M, c, dimL, n):
    nz = np.linalg.norm(z)
    Jp = int((z > 0).sum())
    out.update(
        Jplus=Jp,
        margin=float(np.min(np.abs(z)) / nz) if nz > 0 else 0.0,
        flats=max(0, dimL + Jp - n),
        z=z,
    )
    return out


# --------------------------------------------------- recovered LP quantities

def recover1(inst, S, t, z):
    B, b, d = inst["B"], inst["b"], inst["d"]
    x = B.T @ (S @ (np.maximum(z, 0.0) + t * (B @ d) + b)) - t * d
    return x, max(0.0, float(np.max(b - B @ x)))          # primal violation


def recover2(inst, R, Pm, t, z):
    B, b, d = inst["B"], inst["b"], inst["d"]
    u = np.maximum(-R @ np.maximum(z, 0.0) - t * (R @ b) + Pm.T @ d + t * b, 0.0)
    return u, float(np.max(np.abs(B.T @ u - d)))          # dual residual
