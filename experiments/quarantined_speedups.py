"""QUARANTINED speedup candidates C1-C5, installed at RUNTIME only.

Standing quarantine rule (commit 40e5a2d): a candidate lives in an isolated
experiment module until it has passed a full model-interleaved A/B and the
regression suite.  Nothing in this file edits a shipped file.  Every patch is
installed by rebinding a name on an already-imported module inside a
``try/finally`` context manager, and every patch is INDEPENDENTLY toggleable:

    c1_const_cache   TIER 1  per-model constant caching in TikhonovFace.__init__
    c2_face_share    TIER 1  share the face object across same-eps ladder stages
    c3_refine_break  TIER 2  adaptive early exit in TikhonovFace.pinv refinement
    c4_stall_window  TIER 2  newton_oracle.STALL_WINDOW 25 -> 8
    c5_pchol_screen  TIER 2  woodbury_face_solver.PCHOL_SCREEN 20 -> 8

**PROMOTION NOTE (c1 and c5 have left quarantine).**  After the model-
interleaved A/B (-3.233 s panel, 405 runs, no certificate component moved) and
the composition run that produced the 12.219 s benchmark of record, C1 and C5
are the SHIPPED defaults: ``woodbury_face_solver.TIK_CONST_CACHE = True`` and
``woodbury_face_solver.PCHOL_SCREEN = 8``.  This module therefore no longer
MIRRORS them -- it SETS them, and, crucially, it sets them to their
PRE-PROMOTION values when they are not named.  So ``active(())`` is a true
control arm (c1 off, PCHOL_SCREEN back to 20) and every historical A/B in this
project still compares what it compared.  C2, C3 and C4 remain quarantined and
are still installed by rebinding, exactly as before.

TIER 1 is a claim of BIT-IDENTICAL output; ``--self-check`` tests it.
TIER 2 changes the trajectory and is only claimed to preserve the frozen
certificate verdict (exp23.certificate_pair, componentwise on the original raw
(B, b, d), TOL_FEAS = 1e-7).

--------------------------------------------------------------------------
WHAT EACH PATCH ACTUALLY DOES, AND WHAT IT DELIBERATELY DOES NOT DO

C1.  ``TikhonovFace.__init__`` recomputes, on EVERY face construction:

       (a) ``rownorm = A.multiply(A).sum(axis=1)`` over the face's core rows.
           Row norms are a property of ``cls.A_core`` ROWS and are invariant
           under the row gather that builds the face, so the full-model vector
           is cached once per ``FaceClassifier`` and the face reads a slice.
           Bit-identical: a CSR row gather preserves within-row index order,
           so the per-row products and their summation order are unchanged,
           and ``max`` over the same multiset of doubles is the same double.

       (b) ``self.AiDe = sp.csr_matrix(AiDe)``, a COO->CSR conversion of a
           k x m sparse matrix.  ``AiDe`` is DEAD on the shipped call path:
           ``solve_e``/``apply_H``/``null_part``/``pinv`` read ``iDe``, ``A``,
           ``At`` and ``cho``, never ``AiDe`` (verified by grep over
           experiments/: the only other mention is
           ``tikhonov_factor_update.py:127``, which ASSIGNS it).  The patch
           makes it lazy through ``__getattr__``, so the attribute still
           exists for any reader and costs nothing when unread.  No
           arithmetic is touched: the ``M`` Cholesky still consumes the same
           COO ``AiDe`` object.

     NOT DONE, and the reason: the brief also lists ``csr_matrix(A.T)``
     (0.171 ms) as a per-model constant.  It is not one.  ``A`` is the face's
     core-row subset, so ``A.T`` is face-dependent and cannot be hoisted to
     ``__init__``-time model state.  The only bit-identical hoist available is
     to cache ``csr_matrix(cls.A_core.T)`` (m x K) once and take
     ``AcT[:, sel]`` per face; scipy's CSR minor-axis fancy index scans the
     FULL ``nnz(A_core)``, which is >= the face's ``nnz``, so it is a probable
     REGRESSION whenever a face holds a minority of the model's core rows.
     Timing is out of scope here (measurement is queued elsewhere), so the
     unmeasurable variant is not shipped even behind a flag.

C2.  ``FaceStepKernel._tik_step`` builds a fresh ``TikhonovFace`` for each of
     the four ``wfs.TIK_LADDER`` stages, but stages 0/1 share
     ``eps_rel = 1e-12`` and stages 2/3 share ``1e-15``; the two members of a
     pair differ ONLY in ``deflate``, and ``TikhonovFace.__init__`` funnels
     ``deflate`` into a single integer attribute (``self.deflate_rounds``)
     that no other constructed quantity depends on -- ``D``, ``A``, ``At``,
     ``scale``, ``eps``, ``De``, ``iDe``, ``k`` and the Cholesky factor are
     computed before it and are independent of it.  The patch keeps one face
     per distinct ``eps_rel`` within a single step call and mutates
     ``deflate_rounds`` in place.  Faces are NEVER shared across step calls:
     the support changes between Newton iterations and the dict is local.

C3.  ``TikhonovFace.pinv`` runs ``refine`` fixed rounds.  ``rho`` is already
     computed at the top of each round; the patch exits when
     ``|rho|_inf <= C3_REFINE_RTOL * |r|_inf``.  The test is on the RAW
     ``rho``, before null deflation, which is the conservative side (deflation
     can only shrink it).  ``eps_rel``, the shift, the deflation depth and the
     rank cut are untouched, so the EXACT-SPLIT LAW is untouched: what is
     returned is still ``H^+ r`` for the same cut, split by the same exact
     orthogonal projector.  Callers still judge the answer on the honest
     residual against the ORIGINAL sparse operator (``_residual <= tol`` in
     the step route, ``TIK_PIECE_TOL``/``TIK_A1_TOL``/``TIK_ORTHO_TOL`` in the
     terminal route), so this remains false-reject-only.

     SCOPE NOTE, and it is a NARROWING of what a naive class patch would do.
     ``TikhonovFace`` is shared by the STEP route (``newton_oracle._tik_step``,
     acceptance tolerance ``FAST_STEP_TOL = 1e-6``) and the TERMINAL route
     (``oracle_terminal._tik_stage``), whose own fail-closed checks are
     ``TIK_ORTHO_TOL = 1e-9``, ``TIK_A1_TOL = 1e-10`` and
     ``TIK_PIECE_TOL = 1e-8`` -- all AT OR BELOW ``C3_REFINE_RTOL = 1e-9``.
     The 1000x margin the candidate was derived from exists on the step route
     and does NOT exist on the terminal route, where an early exit could flip
     a fail-closed check and push the face onto later ladder stages or onto
     the incumbent.  So the exit is ARMED ONLY INSIDE THE STEP ROUTE
     (``_C3_ARMED`` / ``C3_SCOPE = "step"``); ``C3_SCOPE = "all"`` exists for
     a deliberate future experiment and is not the default.

C4.  ``newton_oracle.STALL_WINDOW`` 25 -> 8.  A pure STOP rule inside
     ``newton_projection``: it can only end a stage earlier, and every stage
     result still has to pass the same terminal test and the same frozen
     certificate.  It cannot change what a PASSING pair certifies to; it can
     change WHICH pair is offered.

C5.  ``woodbury_face_solver.PCHOL_SCREEN`` 20 -> 8.  Routing only.  The
     hysteresis band (``PCHOL_RETIRE_HI``/``LO``) and ``PCHOL_SCREEN_MAX = 60``
     are deliberately left alone, so a decline rate inside the band still
     extends the window instead of deciding on a coin flip.  Every step the
     route produces is still gated on the honest residual.

--------------------------------------------------------------------------
SOURCE PINNING.  C1, C2 and C3 mirror shipped code.  Their installers refuse
to run if the shipped source they mirror has changed (``_SOURCE_SHA``), so a
future edit to ``woodbury_face_solver`` or ``newton_oracle`` cannot silently
leave a stale copy in this file pretending to be equivalent.

Self-check (correctness only, no timing):

    env PYTHONPYCACHEPREFIX=/tmp/fable_lp_pycache OMP_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      .venv/bin/python experiments/quarantined_speedups.py --self-check
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import sys
import weakref
from pathlib import Path

import numpy as np
# scipy is no longer needed here: the only code that used it was the C1 mirror
# of ``TikhonovFace.__init__``, which was PROMOTED into
# ``woodbury_face_solver`` and deleted from this file.

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import newton_oracle as newt              # noqa: E402
import woodbury_face_solver as wfs        # noqa: E402

CANDIDATES = ("c1_const_cache", "c2_face_share", "c3_refine_break",
              "c4_stall_window", "c5_pchol_screen")

TIER1 = ("c1_const_cache", "c2_face_share")
TIER2 = ("c3_refine_break", "c4_stall_window", "c5_pchol_screen")

# ---- tunables of the TIER 2 candidates, all read at install time ----------
# C3: exit refinement when the pre-deflation residual of H x = rr has fallen
# this far below |r|_inf.  Three decades below the acceptance tolerance the
# callers actually use (FAST_STEP_TOL = 1e-6, TIK_PIECE_TOL = 1e-8).
C3_REFINE_RTOL = 1e-9
# C3 SCOPE -- "step" (default) or "all".  See the SCOPE NOTE in the module
# docstring: the terminal route gates on TIK_ORTHO_TOL = 1e-9, TIK_A1_TOL =
# 1e-10 and TIK_PIECE_TOL = 1e-8, all AT OR BELOW C3_REFINE_RTOL, so an early
# exit there is not covered by the 1e-6 margin the candidate was derived from.
C3_SCOPE = "step"
C4_STALL_WINDOW = 8         # newton_oracle.STALL_WINDOW under c4
C5_PCHOL_SCREEN = 8         # wfs.PCHOL_SCREEN under c5

# sha256 of the shipped source this module mirrors.  Recomputed by
# ``--print-source-sha``.
# ``TikhonovFace.__init__`` is deliberately absent: C1 was PROMOTED into it and
# is now selected by ``wfs.TIK_CONST_CACHE``, so this module no longer keeps a
# copy of that body to go stale.
_SOURCE_SHA = {
    "TikhonovFace.pinv":
        "e18a3a2313ecc9851c93338be6a746ef0960b9fe2fd943a9a242577fb76fba0c",
    "FaceStepKernel._tik_step":
        "e04199132dbc94942f10769bd3f4d35dd145a54a6ff85de7053186ca46695f63",
}

_MIRRORED = {
    "TikhonovFace.pinv": lambda: wfs.TikhonovFace.pinv,
    "FaceStepKernel._tik_step": lambda: newt.FaceStepKernel._tik_step,
}


def source_sha():
    """sha256 of every shipped function this module mirrors."""
    out = {}
    for label, getter in _MIRRORED.items():
        try:
            out[label] = hashlib.sha256(
                inspect.getsource(getter()).encode()).hexdigest()
        except (OSError, TypeError):        # already patched, or no source
            out[label] = None
    return out


def _require_pristine(labels):
    """Refuse to install a mirror of source that has changed under us."""
    current = source_sha()
    drift = [k for k in labels if current.get(k) != _SOURCE_SHA[k]]
    if drift:
        raise RuntimeError(
            "quarantined_speedups mirrors shipped source that has CHANGED: %s"
            "; re-derive the patch and update _SOURCE_SHA before running"
            % ", ".join(drift))


# ==========================================================================
# C1 / C3 -- the patched TikhonovFace
# ==========================================================================

# Keyed on the FaceClassifier INSTANCE so nothing outlives the model and the
# shipped object is never mutated.
_ROWNORM_CACHE = weakref.WeakKeyDictionary()

# Counters, for evidence rather than control flow.
STATS = {"c1_faces": 0, "c1_rownorm_hits": 0, "c1_rownorm_misses": 0,
         "c2_builds": 0, "c2_reuses": 0,
         "c3_pinv_calls": 0, "c3_rounds_run": 0, "c3_rounds_possible": 0,
         "c3_early_exits": 0}


def reset_stats():
    for key in STATS:
        STATS[key] = 0


def _core_rownorms(cls):
    """Squared row norms of EVERY core row of the model, cached per model.

    KEPT for the self-check only.  The shipped cache is now
    ``wfs.FaceClassifier.core_rownorms`` (C1 promoted); this weak-keyed copy
    computes the same expression and is used by ``_face_level_checks`` to show
    the two agree.
    """
    cached = _ROWNORM_CACHE.get(cls)
    if cached is None:
        core = cls.A_core
        cached = np.asarray(core.multiply(core).sum(axis=1)).ravel()
        _ROWNORM_CACHE[cls] = cached
        STATS["c1_rownorm_misses"] += 1
    else:
        STATS["c1_rownorm_hits"] += 1
    return cached


@contextlib.contextmanager
def c1_scope(enabled):
    """Set the SHIPPED ``wfs.TIK_CONST_CACHE``; restore it in ``finally``."""
    saved = wfs.TIK_CONST_CACHE
    wfs.TIK_CONST_CACHE = bool(enabled)
    try:
        yield
    finally:
        wfs.TIK_CONST_CACHE = saved


class PatchedTikhonovFace(wfs.TikhonovFace):
    """``TikhonovFace`` with C3 enabled by CLASS flag.

    The flag defaults OFF, and with it off every method delegates to the
    shipped implementation, so installing the class without enabling a
    candidate is a no-op by construction (that is exactly what the ``none``
    arm of the A/B driver relies on when it is asked to install nothing).

    C1 is no longer here: it was promoted into the shipped
    ``TikhonovFace.__init__`` and is selected by ``wfs.TIK_CONST_CACHE``.
    """

    c3_refine_break = False

    # ------------------------------------------------------------ C3 ------
    def pinv(self, r, refine=wfs.TIK_REFINE):
        if not (type(self).c3_refine_break and _C3_ARMED):
            return super().pinv(r, refine=refine)
        # ---- MIRROR of TikhonovFace.pinv with ONE added exit test ---------
        r = np.asarray(r, dtype=float)
        rn = self.null_part(r)
        rr = r - rn
        x = self.solve_e(rr)
        x = x - self.null_part(x)
        gnorm = float(np.abs(r).max()) if r.size else 0.0
        floor = C3_REFINE_RTOL * gnorm
        STATS["c3_pinv_calls"] += 1
        STATS["c3_rounds_possible"] += int(refine)
        for _ in range(refine):
            rho = rr - self.apply_H(x)
            if float(np.abs(rho).max()) <= floor:
                STATS["c3_early_exits"] += 1
                break
            STATS["c3_rounds_run"] += 1
            rho = rho - self.null_part(rho)
            x = x + self.solve_e(rho)
            x = x - self.null_part(x)
        return x, rn


# --------------------------------------------------------------------------
# C3 SCOPE GATE.  ``_C3_ARMED`` is True only inside the STEP route, which is
# the route whose acceptance tolerance (FaceStepKernel.tol = FAST_STEP_TOL =
# 1e-6) is three decades above C3_REFINE_RTOL.  Under ``C3_SCOPE = "all"`` it
# is armed for the whole process, terminal route included -- available for a
# deliberate experiment, never the default.
_C3_ARMED = False


@contextlib.contextmanager
def c3_scope():
    """Arm the C3 early exit for the duration of the block."""
    global _C3_ARMED
    previous = _C3_ARMED
    _C3_ARMED = True
    try:
        yield
    finally:
        _C3_ARMED = previous


def _scoped(base):
    """Wrap a ``_tik_step`` implementation so C3 is armed only inside it."""

    def wrapper(self, S, grad, gnorm):
        with c3_scope():
            return base(self, S, grad, gnorm)

    wrapper.__name__ = base.__name__ + "_c3scoped"
    return wrapper


# ==========================================================================
# C2 -- share the face across same-eps ladder stages
# ==========================================================================

def _tik_step_shared(self, S, grad, gnorm):
    """MIRROR of ``FaceStepKernel._tik_step`` with the face reused per eps.

    Identical control flow, identical acceptance test, identical stats.  The
    ONLY difference is that a ladder stage whose ``eps_rel`` was already built
    in THIS call reuses that face and sets ``deflate_rounds`` in place.
    """
    best = None
    shared = {}
    for stage, (eps_rel, deflate, refine) in enumerate(wfs.TIK_LADDER):
        try:
            face = shared.get(eps_rel)
            if face is None:
                face = wfs.TikhonovFace(self._tik_cls, S, eps_rel=eps_rel,
                                        deflate=deflate)
                shared[eps_rel] = face
                STATS["c2_builds"] += 1
            else:
                STATS["c2_reuses"] += 1
            face.deflate_rounds = int(deflate)
            delta, r_null = face.pinv(grad, refine=refine)
            if not np.all(np.isfinite(delta)):
                raise ArithmeticError("nonfinite")
            residual = self._residual(S, delta, r_null, grad, gnorm)
        except (ArithmeticError, ValueError, MemoryError,
                np.linalg.LinAlgError):
            continue
        if residual <= self.tol:
            self.stats["tik_stage_hist"][stage] += 1
            return delta, r_null, residual
        if best is None or residual < best[2]:
            best = (delta, r_null, residual)
    if best is None:
        raise ArithmeticError("all tik ladder stages failed")
    return best


# ==========================================================================
# install / restore
# ==========================================================================

_ACTIVE = None


def normalize(names):
    """Accept a name, a sequence, 'all', 'tier1', 'tier2', or None."""
    if names is None:
        return ()
    if isinstance(names, str):
        names = (names,)
    out = []
    for name in names:
        if name == "all":
            out.extend(CANDIDATES)
        elif name == "tier1":
            out.extend(TIER1)
        elif name == "tier2":
            out.extend(TIER2)
        elif name in ("none", ""):
            continue
        elif name in CANDIDATES:
            out.append(name)
        else:
            raise ValueError("unknown candidate %r (known: %s)"
                             % (name, ", ".join(CANDIDATES)))
    return tuple(sorted(set(out), key=CANDIDATES.index))


def installed():
    """The installed candidate tuple, or ``None`` when nothing is installed."""
    return _ACTIVE


@contextlib.contextmanager
def active(names):
    """Install EXACTLY the named candidates; restore everything in ``finally``.

    "Exactly" is load-bearing since C1 and C5 were promoted: a candidate that
    is NOT named is forced to its PRE-PROMOTION value for the duration of the
    block, not left at the shipped default.  Without that, an A/B whose control
    arm is ``active(())`` would silently measure treatment against treatment.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("quarantined_speedups.active is already installed "
                           "with %s; nesting is not supported" % (_ACTIVE,))
    wanted = normalize(names)
    need_face_class = "c3_refine_break" in wanted
    mirrors = []
    if "c3_refine_break" in wanted:
        mirrors.append("TikhonovFace.pinv")
    if "c2_face_share" in wanted:
        mirrors.append("FaceStepKernel._tik_step")
    _require_pristine(mirrors)

    global _C3_ARMED
    saved = {
        "TikhonovFace": wfs.TikhonovFace,
        "_tik_step": newt.FaceStepKernel.__dict__["_tik_step"],
        "STALL_WINDOW": newt.STALL_WINDOW,
        "PCHOL_SCREEN": wfs.PCHOL_SCREEN,
        "TIK_CONST_CACHE": wfs.TIK_CONST_CACHE,
        "c3_flag": PatchedTikhonovFace.c3_refine_break,
        "c3_armed": _C3_ARMED,
    }
    _ACTIVE = wanted
    try:
        PatchedTikhonovFace.c3_refine_break = "c3_refine_break" in wanted
        if need_face_class:
            wfs.TikhonovFace = PatchedTikhonovFace
        step = saved["_tik_step"]
        if "c2_face_share" in wanted:
            step = _tik_step_shared
        if "c3_refine_break" in wanted:
            if C3_SCOPE == "all":
                _C3_ARMED = True
            else:
                step = _scoped(step)
        if step is not saved["_tik_step"]:
            newt.FaceStepKernel._tik_step = step
        if "c4_stall_window" in wanted:
            newt.STALL_WINDOW = int(C4_STALL_WINDOW)
        # PROMOTED: set, do not merely raise.  Absent means pre-promotion.
        wfs.TIK_CONST_CACHE = "c1_const_cache" in wanted
        wfs.PCHOL_SCREEN = (int(C5_PCHOL_SCREEN)
                            if "c5_pchol_screen" in wanted
                            else int(wfs.PCHOL_SCREEN_PRE_PROMOTION))
        yield wanted
    finally:
        wfs.TikhonovFace = saved["TikhonovFace"]
        newt.FaceStepKernel._tik_step = saved["_tik_step"]
        newt.STALL_WINDOW = saved["STALL_WINDOW"]
        wfs.PCHOL_SCREEN = saved["PCHOL_SCREEN"]
        wfs.TIK_CONST_CACHE = saved["TIK_CONST_CACHE"]
        PatchedTikhonovFace.c3_refine_break = saved["c3_flag"]
        _C3_ARMED = saved["c3_armed"]
        _ACTIVE = None


def installed_state():
    """What is live RIGHT NOW, read off the modules -- not off intent."""
    return {
        "active": list(_ACTIVE or ()),
        "TikhonovFace": wfs.TikhonovFace.__name__,
        "TikhonovFace_c1": bool(wfs.TIK_CONST_CACHE),
        "TikhonovFace_c3": bool(getattr(wfs.TikhonovFace,
                                        "c3_refine_break", False)),
        "tik_step": newt.FaceStepKernel.__dict__["_tik_step"].__name__,
        "STALL_WINDOW": int(newt.STALL_WINDOW),
        "PCHOL_SCREEN": int(wfs.PCHOL_SCREEN),
        "C3_REFINE_RTOL": C3_REFINE_RTOL,
        "C3_SCOPE": C3_SCOPE,
        "C3_ARMED_NOW": bool(_C3_ARMED),
    }


# ==========================================================================
# SELF-CHECK
# ==========================================================================
#
# afiro and sc50a -- the only two models this task is permitted to run -- have
# unit-row fractions 0.5075 and 0.4068, both BELOW wfs.TIK_MIN_UNIT = 0.70, so
# the S1 classifier is never installed on either, and m = 32 / 48 are both
# below newton_oracle.FAST_MIN_COLUMNS = 100, so FaceStepKernel._active is
# False and the S2 piv-chol route never runs either.  On those two models C1,
# C2, C3 and C5 are structurally unreachable: an end-to-end comparison there
# proves the harness is INERT, and nothing about the candidates.
#
# So the candidate-level evidence is produced two ways:
#
#   FACE LEVEL -- a synthetic unit-heavy model, faces drawn from it, and a
#   direct comparison of TikhonovFace outputs and of _tik_step outputs,
#   byte-for-byte (``tobytes()``), patched vs shipped, in ONE process with the
#   call order controlled.
#
#   PIPELINE LEVEL -- the same synthetic model driven through
#   ``staged_newton.follow_staged`` (which reaches every route: S1 step, S1
#   terminal, S2, the stall guard and the frozen certificate), plus real
#   ``exp84`` runs on afiro and sc50a.
#
# NATIVE NONDETERMINISM.  The noise-floor harness measured 5 distinct
# control-flow fingerprints for sc50a across 9 runs; the cause sits BELOW
# control flow (BLAS/FP path selection).  Bit-identity is therefore only
# claimed where it is checkable against that: the face-level checks are pure
# functions of fixed inputs, run back-to-back in one process, and the
# CONTROL for the claim is a baseline-vs-baseline repeat -- if the shipped
# path does not reproduce itself in that same process, the comparison is
# reported as INCONCLUSIVE rather than as a pass.

def synthetic_model(m=120, core=30, seed=7, core_scale=1.0, nnz=5):
    """A unit-heavy bounded LP in the (B x >= b) form the lane consumes.

    ``2m`` unit rows (a box) + ``core`` sparse core rows; unit fraction
    2m/(2m+core) = 0.889 >= TIK_MIN_UNIT, and m >= FAST_MIN_COLUMNS, so BOTH
    the S1 and S2 routes install -- which afiro and sc50a cannot do.

    ``core_scale`` reproduces the regime the shift ladder exists for: unit
    rows all carry value 1 while core rows are much larger, so
    ``eps = 1e-12 * scale`` climbs toward real face spectrum and the
    refinement loop has real work to do (fit1d's ratio is 9.65e3; the hard
    probe below uses 1e3-1e4).
    """
    rng = np.random.default_rng(seed)
    rows, rhs = [], []
    for j in range(m):                      # x_j >= -1 and -x_j >= -1
        e = np.zeros(m)
        e[j] = 1.0
        rows.append(e.copy())
        rhs.append(-1.0)
        rows.append(-e)
        rhs.append(-1.0)
    for _ in range(core):
        row = np.zeros(m)
        cols = rng.choice(m, size=nnz, replace=False)
        row[cols] = rng.standard_normal(nnz) * core_scale * rng.uniform(
            0.1, 10.0)
        rows.append(row)
        rhs.append(-float(rng.uniform(0.5, 2.0)) * core_scale)
    B = np.asarray(rows)
    b = np.asarray(rhs)
    d = rng.standard_normal(m)
    return B, b, d


EASY = dict(m=120, core=30, seed=7, core_scale=1.0, nnz=5)
# HARD: core rows ~1e3 larger than the unit rows and denser, so the shift sits
# near real spectrum -- the fit1d TRUNCATION regime the ladder was built for,
# and the regime in which C3's early exit is a real decision rather than a
# formality.
HARD = dict(m=140, core=60, seed=19, core_scale=1.0e3, nnz=14)
# BIG: face-level probe only.  The point is the CORE-ROW COUNT: the traces the
# candidates were derived from have ship04s faces at k = 325..478, and C1's
# cached row norms and C2's shared Cholesky both scale with k, so the identity
# claims are worth testing at that k rather than only at k <= 30.  Unit
# fraction 1200/1700 = 0.706, just over TIK_MIN_UNIT.
BIG = dict(m=600, core=500, seed=31, core_scale=1.0e3, nnz=25)
# WIDE: face-level probe only.  Large k AND well conditioned, so refinement
# actually converges and C3's early exit is a live decision at ship04s-like
# face size -- the regime the C3 measurement came from (rho/|g| after rounds
# 0/1/2 = 1.5e-4 / 1e-7 / 5e-11).  BIG is its ill-conditioned twin, where
# refinement never converges and C3 must (and does) run every round.
WIDE = dict(m=600, core=500, seed=37, core_scale=1.0, nnz=25)


def _face_probes(B, count=12, seed=3):
    """A few honest faces: supports of the rows active at random points."""
    rng = np.random.default_rng(seed)
    n, m = B.shape
    out = []
    for _ in range(count):
        x = rng.standard_normal(m) * rng.uniform(0.2, 3.0)
        v = B @ x
        S = v > np.quantile(v, rng.uniform(0.1, 0.5))
        if S.sum() >= 5:
            out.append(S)
    return out


def _same_bytes(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _honest_residual(cls, face, x, rn, r):
    """``||B_S'(B_S x) + r_null - r||_inf / ||r||_inf`` on the ORIGINAL rows.

    Exactly the quantity ``FaceStepKernel._residual`` gates on (tol = 1e-6)
    and, up to the right-hand side, the quantity ``TIK_PIECE_TOL`` gates on.
    """
    BS = cls.Bs[face.idx]
    return float(np.abs(BS.T @ (BS @ x) + rn - r).max()) / max(
        float(np.abs(r).max()), 1e-300)


def _face_level_checks(model_kwargs=None, label="synth"):
    """C1/C2/C3 at the face level, on a synthetic unit-heavy model."""
    B, b, d = synthetic_model(**(model_kwargs or EASY))
    del b, d
    cls = wfs.FaceClassifier(B)
    faces = _face_probes(B)
    report = {"label": label, "unit_fraction": float(wfs.unit_row_fraction(B)),
              "faces": len(faces), "shape": list(B.shape),
              "model_kwargs": dict(model_kwargs or EASY)}

    def rng_fixed(B, S):
        # deterministic right-hand side per face
        h = int(hashlib.sha256(np.packbits(S).tobytes()).hexdigest()[:8], 16)
        return np.random.default_rng(h).standard_normal(B.shape[1])

    # ---- baseline: pinv over the ladder settings, twice (the CONTROL) -----
    def sweep(build):
        out = []
        for S in faces:
            r = rng_fixed(B, S)
            for eps_rel, deflate, refine in wfs.TIK_LADDER:
                face = build(cls, S, eps_rel, deflate)
                x, rn = face.pinv(r, refine=refine)
                out.append((x.copy(), rn.copy(), float(face.scale),
                            float(face.eps), int(face.k),
                            _honest_residual(cls, face, x, rn, r)))
        return out

    # The BASELINE is the PRE-PROMOTION face: C1 is now the shipped default, so
    # the control arm has to switch it off explicitly or it would be the
    # treatment.
    shipped = wfs.TikhonovFace
    with c1_scope(False):
        base_a = sweep(lambda c, S, e, dfl: shipped(c, S, eps_rel=e,
                                                    deflate=dfl))
        base_b = sweep(lambda c, S, e, dfl: shipped(c, S, eps_rel=e,
                                                    deflate=dfl))
    control = all(_same_bytes(p[0], q[0]) and _same_bytes(p[1], q[1])
                  for p, q in zip(base_a, base_b))
    report["control_baseline_reproduces_itself"] = bool(control)

    # ---- C1 (now the shipped default; the flag selects it) ----------------
    PatchedTikhonovFace.c3_refine_break = False
    with c1_scope(True):
        c1 = sweep(lambda c, S, e, dfl: shipped(c, S, eps_rel=e, deflate=dfl))
    # The per-model cache the shipped classifier builds must be the same vector
    # this module computes independently.
    report["c1_cache_matches_recompute"] = bool(_same_bytes(
        cls.core_rownorms(), _core_rownorms(cls)))
    report["c1_bit_identical"] = bool(control and all(
        _same_bytes(p[0], q[0]) and _same_bytes(p[1], q[1])
        and p[2:] == q[2:] for p, q in zip(base_a, c1)))
    report["c1_scale_eps_identical"] = bool(
        all(p[2:] == q[2:] for p, q in zip(base_a, c1)))

    # ---- C2: mutate deflate_rounds on a reused face vs rebuild ------------
    same, checked = True, 0
    with c1_scope(False):
        for S in faces:
            r = rng_fixed(B, S)
            for eps_rel in (wfs.TIK_EPS_REL, 1e-15):
                deflates = [dfl for e, dfl, _ in wfs.TIK_LADDER
                            if e == eps_rel]
                if len(deflates) < 2:
                    continue
                shared_face = shipped(cls, S, eps_rel=eps_rel,
                                      deflate=deflates[0])
                for dfl in deflates:
                    fresh = shipped(cls, S, eps_rel=eps_rel, deflate=dfl)
                    shared_face.deflate_rounds = int(dfl)
                    for refine in (wfs.TIK_REFINE, 8):
                        xa, ra = fresh.pinv(r, refine=refine)
                        xb, rb = shared_face.pinv(r, refine=refine)
                        checked += 1
                        if not (_same_bytes(xa, xb) and _same_bytes(ra, rb)):
                            same = False
    report["c2_reused_face_bit_identical"] = bool(same)
    report["c2_comparisons"] = checked

    # ---- C3: not bit-identical by design; bound the drift -----------------
    # Measured against ``base_a``, so C1 is held at the baseline value.
    PatchedTikhonovFace.c3_refine_break = True
    try:
        with c3_scope(), c1_scope(False):
            c3 = sweep(lambda c, S, e, dfl:
                       PatchedTikhonovFace(c, S, eps_rel=e, deflate=dfl))
    finally:
        PatchedTikhonovFace.c3_refine_break = False
    drift = 0.0
    for pa, pb in zip(base_a, c3):
        scale = max(1.0, float(np.abs(pa[0]).max()))
        drift = max(drift, float(np.abs(pa[0] - pb[0]).max()) / scale)
    report["c3_max_relative_step_drift"] = drift
    report["c3_identical_fraction"] = float(np.mean(
        [_same_bytes(p[0], q[0]) for p, q in zip(base_a, c3)]))
    # The gate the callers actually apply.  FAST_STEP_TOL = 1e-6 in the step
    # route, TIK_PIECE_TOL = 1e-8 in the terminal route.
    report["honest_residual_worst_baseline"] = max(p[5] for p in base_a)
    report["honest_residual_worst_c3"] = max(p[5] for p in c3)
    report["c3_pass_1e-6_baseline"] = int(sum(p[5] <= 1e-6 for p in base_a))
    report["c3_pass_1e-6_c3"] = int(sum(p[5] <= 1e-6 for p in c3))
    report["c3_pass_1e-8_baseline"] = int(sum(p[5] <= 1e-8 for p in base_a))
    report["c3_pass_1e-8_c3"] = int(sum(p[5] <= 1e-8 for p in c3))
    report["c3_acceptance_unchanged_1e-6"] = bool(all(
        (p[5] <= 1e-6) == (q[5] <= 1e-6) for p, q in zip(base_a, c3)))
    report["c3_acceptance_unchanged_1e-8"] = bool(all(
        (p[5] <= 1e-8) == (q[5] <= 1e-8) for p, q in zip(base_a, c3)))
    # Margin of the worst C3 residual against the tolerance of each ROUTE.
    # This is the number that decided C3_SCOPE = "step".
    worst = report["honest_residual_worst_c3"]
    report["c3_margin_step_route_1e-6"] = (1e-6 / worst) if worst else None
    report["c3_margin_terminal_route_1e-8"] = (1e-8 / worst) if worst else None
    report["c3_stats"] = {k: STATS[k] for k in STATS if k.startswith("c3_")}
    return report


def _sparse_bytes(matrix):
    return (matrix.data.tobytes(), matrix.indices.tobytes(),
            matrix.indptr.tobytes(), matrix.shape)


def _face_state(face):
    """Every constructed quantity of a face, as comparable bytes."""
    state = {
        "idx": face.idx.tobytes(),
        "D": np.asarray(face.D).tobytes(),
        "De": np.asarray(face.De).tobytes(),
        "iDe": np.asarray(face.iDe).tobytes(),
        "scale": float(face.scale).hex(),
        "eps": float(face.eps).hex(),
        "k": int(face.k),
        "deflate_rounds": int(face.deflate_rounds),
        "A": _sparse_bytes(face.A.tocsr()),
        "At": _sparse_bytes(face.At.tocsr()),
    }
    state["cho"] = (None if face.cho is None
                    else (np.asarray(face.cho[0]).tobytes(), bool(face.cho[1])))
    return state


def _replay_check(model_kwargs, label, limit=400):
    """The strongest available C1/C2 test: REAL faces, byte-for-byte.

    A recording subclass captures the ``(classifier, support, eps_rel,
    deflate)`` of every ``TikhonovFace`` the shipped pipeline builds on a
    synthetic unit-heavy model -- both the step route's faces and the terminal
    route's -- and each captured face is then rebuilt twice, shipped vs
    patched, and compared on EVERY constructed quantity plus the ``pinv``
    outputs.  This does not depend on the probe faces of ``_face_probes``
    being representative, because these ARE the faces the run used.
    """
    import staged_newton as sn

    B, b, d = synthetic_model(**model_kwargs)
    captured = []
    shipped = wfs.TikhonovFace

    class _Recorder(shipped):
        def __init__(self, cls, S, eps_rel=wfs.TIK_EPS_REL,
                     deflate=wfs.TIK_DEFLATE):
            if len(captured) < limit:
                captured.append((cls, np.array(S, copy=True), float(eps_rel),
                                 int(deflate)))
            super().__init__(cls, S, eps_rel=eps_rel, deflate=deflate)

    wfs.TikhonovFace = _Recorder
    try:
        sn.follow_staged(B, b, d, staged_seconds=60.0, staged_diagnostics={},
                         kernel="fast+tik+pchol")
    finally:
        wfs.TikhonovFace = shipped

    report = {"label": label, "faces_captured": len(captured),
              "distinct_eps_rel": sorted({e for _c, _s, e, _d in captured}),
              "k_range": []}
    if not captured:
        report["result"] = "no TikhonovFace was constructed"
        return report

    c1_state_ok = c1_pinv_ok = True
    c2_ok = True
    ks = []
    PatchedTikhonovFace.c3_refine_break = False
    try:
        for cls, S, eps_rel, deflate in captured:
            # C1 is now a shipped flag: the BASE arm is the pre-promotion
            # recompute, the PATCHED arm the per-model cache.
            with c1_scope(False):
                base = shipped(cls, S, eps_rel=eps_rel, deflate=deflate)
            with c1_scope(True):
                patched = shipped(cls, S, eps_rel=eps_rel, deflate=deflate)
            ks.append(int(base.k))
            if _face_state(base) != _face_state(patched):
                c1_state_ok = False
            rhs = np.linspace(-1.0, 1.0, base.cls.m) * 0.5 + 0.25
            for refine in (wfs.TIK_REFINE, 8):
                xa, ra = base.pinv(rhs, refine=refine)
                xb, rb = patched.pinv(rhs, refine=refine)
                if not (_same_bytes(xa, xb) and _same_bytes(ra, rb)):
                    c1_pinv_ok = False
            # ---- C2: same face, deflate mutated in place vs rebuilt --------
            for other in {dfl for e, dfl, _ in wfs.TIK_LADDER
                          if e == eps_rel}:
                with c1_scope(False):
                    fresh = shipped(cls, S, eps_rel=eps_rel, deflate=other)
                base.deflate_rounds = int(other)
                if _face_state(base) != _face_state(fresh):
                    c2_ok = False
                for refine in (wfs.TIK_REFINE, 8):
                    xa, ra = fresh.pinv(rhs, refine=refine)
                    xb, rb = base.pinv(rhs, refine=refine)
                    if not (_same_bytes(xa, xb) and _same_bytes(ra, rb)):
                        c2_ok = False
            base.deflate_rounds = int(deflate)
    finally:
        PatchedTikhonovFace.c3_refine_break = False

    report["k_range"] = [min(ks), max(ks)]
    report["c1_constructed_state_identical"] = bool(c1_state_ok)
    report["c1_pinv_bit_identical"] = bool(c1_pinv_ok)
    report["c2_mutated_face_identical"] = bool(c2_ok)
    return report


def _pipeline_check(B, b, d, arms, label, seconds=60.0):
    """Drive ``follow_staged`` once per arm and compare frozen gate output."""
    import exp23_path_primal_dual as exp23
    import staged_newton as sn

    rows = []
    for arm in arms:
        diag = {}
        reset_stats()
        with active(arm):
            state = installed_state()
            result = sn.follow_staged(
                B, b, d, staged_seconds=seconds, staged_diagnostics=diag,
                kernel="fast+tik+pchol", maxiter=newt.MAXITER)
        y = result.get("y")
        x = result.get("x")
        detail = None
        if x is not None and y is not None:
            _ok, detail = exp23.certificate_pair(B, b, d,
                                                 np.asarray(x, float),
                                                 np.asarray(y, float))
        lb = result.get("lb")
        stages = diag.get("stages") or []
        rows.append({
            "model": label,
            "arm": ",".join(arm) or "none",
            "installed": state,
            "status": result.get("status"),
            "lb": None if lb is None else float(lb),
            "lb_hex": None if lb is None else float(lb).hex(),
            "newton_iters_total": diag.get("newton_iters_total"),
            "stages_run": diag.get("stages_run"),
            "accepted_stage": diag.get("accepted_stage"),
            "accepted_support": diag.get("accepted_support"),
            "max_step_residual": max(
                [float(s.get("max_step_residual") or 0.0) for s in stages]
                or [0.0]),
            "certificate": detail,
            "certificate_max": (None if not detail else max(
                float(detail.get(k, 0.0))
                for k in ("primal", "dual", "nonnegative", "gap"))),
            "tik_terminal": diag.get("tik_terminal"),
            "tik_unit_fraction": diag.get("tik_unit_fraction"),
            "patch_stats": dict(STATS),
        })
    return rows


def _model_check(name, arms):
    """Real netlib model through exp84.run, one row per arm."""
    import exp84_staged_newton as exp84
    import quarantined_pivchol_alloc as qpa

    rows = []
    for arm in arms:
        with active(arm), qpa.active(()):
            state = installed_state()
            # PRE-PROMOTION pipeline, so this check isolates the candidate:
            # frozen route, raw program, kernel flags exactly as ``active``
            # and ``qpa.active`` set them.
            record = exp84.run(name, seconds=120, quiet=True,
                               route="frozen", equilibrate=False,
                               candidates=None, alloc_patches=None)
        detail = record.get("certificate_detail") or {}
        lb = record.get("lb")
        stages = record.get("stage_table") or []
        rows.append({
            "model": name,
            "arm": ",".join(arm) or "none",
            "installed": state,
            "status": record.get("status"),
            "lb": lb,
            "lb_hex": None if lb is None else float(lb).hex(),
            "newton_iters_total": record.get("newton_iters_total"),
            "stages_run": record.get("stages_run"),
            "accepted_stage": record.get("accepted_stage"),
            "accepted_face_size": record.get("accepted_face_size"),
            "max_step_residual": max(
                [float(s.get("max_step_residual") or 0.0) for s in stages]
                or [0.0]),
            "certificate": {k: detail.get(k) for k in
                            ("reason", "primal", "dual", "nonnegative",
                             "gap", "gate_x", "certificate_detail_skipped",
                             "certificate_detail_ok",
                             "certificate_detail_error")},
            # See exp_quarantined_ab._cert_row: when oracle_terminal._gate
            # accepts _certificate_detail's OWN x, the four componentwise
            # numbers left in the block belong to the REJECTED candidate
            # (afiro: primal 1.64e-2 next to reason "passed").  They describe
            # the accepted pair only when gate_x is "min-norm" / the detail
            # step was skipped.
            "certificate_numbers_authoritative": bool(
                detail.get("certificate_detail_skipped") is True
                or detail.get("gate_x") in (None, "min-norm")),
            "certificate_max": max(
                [float(detail.get(k) or 0.0)
                 for k in ("primal", "dual", "nonnegative", "gap")] or [0.0]),
        })
    return rows


def self_check():
    reset_stats()
    out = {"source_sha_matches": source_sha() == _SOURCE_SHA,
           "source_sha": source_sha()}
    out["face_level"] = []
    for label, kwargs in (("easy", EASY), ("hard", HARD), ("big", BIG),
                          ("wide", WIDE)):
        reset_stats()
        out["face_level"].append(_face_level_checks(kwargs, label))

    out["replay"] = [_replay_check(kwargs, label)
                     for label, kwargs in (("easy", EASY), ("hard", HARD))]

    arms = [(), ("c1_const_cache",), ("c2_face_share",),
            ("c3_refine_break",), ("c4_stall_window",),
            ("c5_pchol_screen",), CANDIDATES, ()]
    out["pipeline_synthetic"] = []
    for label, kwargs in (("easy", EASY), ("hard", HARD)):
        B, b, d = synthetic_model(**kwargs)
        out["pipeline_synthetic"].extend(
            _pipeline_check(B, b, d, arms, "synth-" + label))
    out["netlib"] = {}
    for name in ("afiro", "sc50a"):
        out["netlib"][name] = _model_check(name, arms)

    # ---- verdicts ---------------------------------------------------------
    def verdict(rows):
        base = [r for r in rows if r["arm"] == "none"]
        control_ok = (len(base) < 2
                      or (base[0]["lb_hex"] == base[-1]["lb_hex"]
                          and base[0]["newton_iters_total"]
                          == base[-1]["newton_iters_total"]))
        return {
            "baseline_repeats_itself_in_process": bool(control_ok),
            "all_certified": all(r["status"] == "CERTIFIED" for r in rows),
            "statuses": sorted({r["status"] for r in rows}),
            # None (not 0.0) when EVERY row's four numbers belong to a
            # rejected candidate, as on afiro: silence, not a false zero.
            "max_certificate_error": (
                max([r["certificate_max"] for r in rows
                     if r.get("certificate_numbers_authoritative", True)])
                if any(r.get("certificate_numbers_authoritative", True)
                       for r in rows) else None),
            "certificate_reasons": sorted({str((r["certificate"] or {}).get(
                "reason") if isinstance(r["certificate"], dict)
                else None) for r in rows}),
            "rows_with_non_authoritative_certificate_numbers": sum(
                not r.get("certificate_numbers_authoritative", True)
                for r in rows),
            "max_step_residual": max(r["max_step_residual"] for r in rows),
            "lb_hex_by_arm": {r["arm"]: r["lb_hex"] for r in rows},
            "tier1_bit_identical_vs_baseline": {
                r["arm"]: bool(control_ok
                               and r["lb_hex"] == base[0]["lb_hex"]
                               and r["newton_iters_total"]
                               == base[0]["newton_iters_total"]
                               and r["accepted_stage"]
                               == base[0]["accepted_stage"])
                for r in rows
                if r["arm"] in ("c1_const_cache", "c2_face_share")},
        }

    out["verdict"] = {
        label: verdict([r for r in out["pipeline_synthetic"]
                        if r["model"] == label])
        for label in ("synth-easy", "synth-hard")}
    out["verdict"].update({k: verdict(v) for k, v in out["netlib"].items()})
    out["note"] = (
        "afiro/sc50a cannot reach C1/C2/C3 (unit fraction 0.51/0.41 < 0.70) "
        "nor C5 (m 32/48 < FAST_MIN_COLUMNS 100); their rows prove the patch "
        "harness is inert, not that the candidates are correct.  The "
        "candidate evidence is face_level + pipeline_synthetic.")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--print-source-sha", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.print_source_sha:
        print(json.dumps(source_sha(), indent=2, sort_keys=True))
        return
    if args.self_check:
        report = self_check()
        text = json.dumps(report, indent=2, sort_keys=True, default=str)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        return
    print(json.dumps(installed_state(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
