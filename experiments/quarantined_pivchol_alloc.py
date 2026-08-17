"""Allocation / numpy-overhead patches for the pivoted-Cholesky face solver.

*** PROMOTED.  a1_tril_slice, a2_u_gather, a3_diag_scalar and b1_churn_mask
LEFT QUARANTINE and are the shipped implementation. ***  They live in
``woodbury_face_solver.piv_chol_step`` (selected by ``ALLOC_TRIL_SLICE`` /
``ALLOC_U_GATHER`` / ``ALLOC_DIAG_SCALAR``) and in
``newton_oracle.FaceStepKernel.step`` (``CHURN_MASK``), each behind a flag
whose False branch is the PRE-PROMOTION body line for line.

This module therefore no longer SWAPS those functions -- it SETS those flags,
and it sets them to their PRE-PROMOTION values when a target is not named.  So
``active(())`` is a true control arm and ``bench_pivchol_alloc``'s ``equil``
cell still measures what it measured.  ``a4_u_free`` is still quarantined and
is still installed by rebinding ``wfs.piv_chol_step``.

The differential replay below is unchanged in meaning: it now runs the SHIPPED
function twice on copies, once with the flags off (the reference) and once with
the named targets on (the treatment), and demands byte equality of ``delta``,
``r_null``, ``rank`` and ``clean``.

ORIGINAL QUARANTINE DISCIPLINE (commit 40e5a2d).  Every install is made inside
a context manager and restored in a ``finally``, so a crash cannot leave the
solver in a non-default state.  Every target is individually toggleable.

--------------------------------------------------------------------------
THE TARGETS

A LAPACK-calibration census over the 27-model panel attributed, of a 12.920 s
panel, 0.43 s to ``np.tril``, 0.30 s to ``np.setdiff1d``, 0.15 s to
``np.abs``, and 4.58 GB / 0.587 s to the ``U = zeros(m, rank); U[P] = L``
materialisation inside ``woodbury_face_solver.piv_chol_step``.  Each patch
below removes memory traffic ONLY; none of them changes a floating-point
operation, an operand order, or the order in which any sum is accumulated.

  a1_tril_slice   ``np.tril(c)[:, :rank]``  ->  ``np.tril(c[:, :rank])``
        ``np.tril`` keeps element (i, j) iff ``i >= j``, and slicing columns
        to ``[0, rank)`` changes neither index.  So the two expressions have
        identical elements; the first materialises an ``m x m`` array and
        returns a strided view of it, the second materialises ``m x rank``.
        Traffic: ``m*m`` writes -> ``m*rank`` writes, and every later read of
        ``L`` becomes contiguous instead of strided.
        WHAT CHANGES BESIDES SIZE: ``L``'s memory LAYOUT.  ``L`` is then fed
        to ``sla.solve_triangular`` (twice) and read by the ``U`` build.
        ``solve_triangular`` copies its operands to the layout LAPACK wants
        and calls the same ``dtrtrs``, so the values it returns should be
        bit-identical -- but that is an assumption about scipy/LAPACK, not
        algebra, so it is CHECKED by replay rather than asserted.

  a2_u_gather     ``U = zeros((m, rank)); U[P] = L``  ->  ``U = L[Q]``
        ``P`` is a full permutation of ``0..m-1`` (``dpstrf`` pivots every
        row), so ``U[P[i], :] = L[i, :]`` defines EVERY row of ``U``: with
        ``Q`` the inverse permutation (``Q[P[i]] = i``), ``U[j, :] =
        L[Q[j], :]``, i.e. ``U = L[Q]`` exactly.  The zero-fill is therefore
        dead: it is overwritten in full.  ``U`` is a pure permutation COPY of
        ``L`` either way -- no arithmetic -- so byte-equality of ``U`` here is
        a COPY check, not a recompute check, and it is exact by construction.
        Traffic: ``3 * m*rank*8`` bytes (zero-fill + scattered write + read)
        -> ``2 * m*rank*8`` (gathered read + sequential write).  Building
        ``Q`` costs one ``O(m)`` integer scatter.

  a3_diag_scalar  ``dl = np.abs(np.diag(c)[:rank])``  ->  two scalars
        ``dl`` is consumed ONLY as ``float(dl[-1])`` and ``float(dl[0])``
        inside ``clean``.  ``float(np.abs(x)) == abs(float(x))`` exactly for
        every finite double (``abs`` only clears the sign bit), so ``clean``
        is computed from the same two Python floats by the same Python
        expression.  Small: this one is included because the census names
        ``np.abs``, not because it can be large.

  b1_churn_mask   ``np.setdiff1d`` x2  ->  one boolean XOR count
        See TARGET B below.

  a4_u_free       EXPERIMENTAL, NOT ADMITTED, off by default.
        Drops ``U`` entirely and applies the permutation to the VECTORS
        instead (``a = L' grad[P]``, ``delta[P] = L gsolve(z)``, ...).  This
        is algebraically exact but it REORDERS the accumulation of three
        ``dgemv`` reductions of length ``m``, and floating-point addition is
        not associative.  It is provided so the replay harness can MEASURE
        how far it moves, and it is refused by ``active()`` unless
        ``allow_experimental=True``.  It is not part of the A/B arm.

--------------------------------------------------------------------------
TARGET B -- ``newton_oracle.FaceStepKernel.step`` support-churn accounting

    moved  = np.setdiff1d(idx, self._previous, assume_unique=True).size
    moved += np.setdiff1d(self._previous, idx, assume_unique=True).size
    self.stats["support_changes"].append(int(moved))

DIAGNOSTIC, not load-bearing.  ``support_changes`` is written here and read
in exactly one place, ``FaceStepKernel.counters()`` (newton_oracle.py:1064,
1100-1109), which turns it into the ``support_change`` summary block of the
stats dict.  No routing decision, tolerance, cadence or gate anywhere reads
it -- the ``max_changes`` cadence rule of ``_FastQR`` computes its OWN
``changes`` from its own additions/removals (newton_oracle.py:333).  So the
value only has to be preserved, and it is:

    ``moved`` is |A \\ B| + |B \\ A| = |A XOR B| for the sets A = idx,
    B = previous idx.  The call site is ``kernel.step(S, np.where(S)[0],
    ...)`` (newton_oracle.py:1309-1310, and identically exp98:84), so ``idx``
    is exactly the support of the boolean mask ``S`` and ``S`` has the same
    fixed length ``n`` on every call.  Hence ``|A XOR B| ==
    np.count_nonzero(S != S_prev)``, an exact integer identity -- one O(n)
    pass instead of two sort-based ``setdiff1d`` calls.

HOW IT IS INSTALLED WITHOUT COPYING THE 180-LINE ``step``.  The churn block
is a prologue that runs iff ``self._active and self._previous is not None``.
The wrapper does the accounting from its own mask, sets ``self._previous =
None`` so the shipped prologue takes its no-op branch (it then re-assigns
``self._previous = idx`` itself), and delegates.  ``self._previous`` is read
NOWHERE else in the file (definitions at :617, :887; reads at :881-885 only),
so nothing downstream can observe the difference.  No verbatim copy of
``step`` is kept here, so this patch cannot go stale against the shipped
body; the shipped source is still SHA-pinned below so that a future edit to
the prologue is caught rather than silently mis-patched.

--------------------------------------------------------------------------
VALIDATION -- ``--differential``

Byte-equality is established by DIFFERENTIAL REPLAY on real inputs captured
from live solves of cheap Netlib models under the A/B's own configuration
(``qr_cod`` + ``c1_const_cache`` + ``c5_pchol_screen`` + Ruiz).  For every
``piv_chol_step`` call the wrapper:

  * copies ``H`` before the shipped call consumes it (``dpstrf`` overwrites),
  * lets the SHIPPED function run on the real ``H`` and returns ITS result to
    the solver, unmodified -- the solver's trajectory is never touched,
  * re-runs the SHIPPED function on a fresh copy (``ref``) and the PATCHED
    function on another fresh copy (``new``), and compares ``ref`` vs ``new``
    byte for byte.

Both compared arms are computed from copies, so any buffer-alignment effect
applies to both.  The extra ``live vs ref`` comparison is reported separately
as the COPY BASELINE: it measures how much the incumbent moves when it is
merely handed a different (identical-valued) buffer.  A non-zero copy
baseline would mean a byte-equality verdict on this quantity is not
meaningful; a zero baseline means the arm comparison is trustworthy.

WHICH CHECKS ARE COPY CHECKS AND WHICH ARE RECOMPUTES
  * ``L`` values, ``U`` values, ``rank``, ``clean``  -- copy/slice/abs only;
    byte-equality here is exact by construction.
  * ``delta``, ``r_null`` -- RECOMPUTED through ``dtrtrs``/``dgemv``.  Under
    a1/a2/a3 the operands handed to BLAS are byte-identical arrays with
    identical shapes, so identity is expected but it is MEASURED, not
    assumed.  Under a4 the reduction order genuinely changes and identity is
    NOT expected.

Canonical invocations -- thread env vars MUST be set before python starts::

    env PYTHONPYCACHEPREFIX=/tmp/fable_lp_pycache OMP_NUM_THREADS=1 \\
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
      NUMEXPR_NUM_THREADS=1 \\
      ~/.venvs/claude/bin/python experiments/quarantined_pivchol_alloc.py \\
        --differential --models afiro sc50a adlittle beaconfd scorpion

    ... experiments/quarantined_pivchol_alloc.py --self-check
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.linalg import lapack

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import newton_oracle as newt              # noqa: E402
import woodbury_face_solver as wfs        # noqa: E402

TARGETS = ("a1_tril_slice", "a2_u_gather", "a3_diag_scalar",
           "b1_churn_mask", "a4_u_free")

# Everything admitted for the A/B.  a4 is deliberately absent.
DEFAULT = ("a1_tril_slice", "a2_u_gather", "a3_diag_scalar", "b1_churn_mask")

EXPERIMENTAL = ("a4_u_free",)

# The pivoted-Cholesky targets; b1 lives in newton_oracle.  Post-promotion the
# only one of these still installed by rebinding is ``a4_u_free``.
_PCHOL_TARGETS = ("a1_tril_slice", "a2_u_gather", "a3_diag_scalar",
                  "a4_u_free")

# sha256 of the shipped source this module patches.  Recomputed by
# ``--print-source-sha``.
_SOURCE_SHA = {
    "piv_chol_step":
        "20c7f769fe14e0a81e9584c5780a273baa92755632cbc517516156063b73c8e9",
    "FaceStepKernel.step":
        "40a8146b4afdada9e0fe18c036f33712f5cfd2c22fd795f8ca9924ff82a9eb70",
    "_gram_apply":
        "1d3911258e3f3f19a71669a76f1f56717c756fccb887367ab7c882b62b738a7c",
}

_MIRRORED = {
    "piv_chol_step": lambda: wfs.piv_chol_step,
    "FaceStepKernel.step": lambda: newt.FaceStepKernel.step,
    "_gram_apply": lambda: wfs._gram_apply,
}

# The pristine objects, bound at import so a nested install can never lose
# them.  Post-promotion these ARE the shipped implementations of a1/a2/a3 and
# b1; which branch they take is decided by the module flags below.
FROZEN_PIV_CHOL_STEP = wfs.piv_chol_step
FROZEN_KERNEL_STEP = newt.FaceStepKernel.__dict__["step"]

# The shipped flags each target now sets, as (module, attribute).
_FLAGS = {
    "a1_tril_slice": (wfs, "ALLOC_TRIL_SLICE"),
    "a2_u_gather": (wfs, "ALLOC_U_GATHER"),
    "a3_diag_scalar": (wfs, "ALLOC_DIAG_SCALAR"),
    "b1_churn_mask": (newt, "CHURN_MASK"),
}


@contextlib.contextmanager
def flag_scope(names):
    """Set EXACTLY the named shipped flags; restore all of them in ``finally``.

    Unnamed flags are forced False -- the pre-promotion code path -- which is
    what makes a control arm a control arm.
    """
    wanted = set(names)
    saved = [(mod, attr, getattr(mod, attr))
             for mod, attr in _FLAGS.values()]
    try:
        for target, (mod, attr) in _FLAGS.items():
            setattr(mod, attr, target in wanted)
        yield
    finally:
        for mod, attr, value in saved:
            setattr(mod, attr, value)

STATS = {"pchol_calls": 0, "a2_gathers": 0, "b1_churn_calls": 0}


def reset_stats():
    for key in STATS:
        STATS[key] = 0


def source_sha():
    out = {}
    for label, getter in _MIRRORED.items():
        try:
            fn = getter()
            fn = getattr(fn, "__wrapped__", fn)
            out[label] = hashlib.sha256(
                inspect.getsource(fn).encode()).hexdigest()
        except (OSError, TypeError):
            out[label] = None
    return out


def _require_pristine(labels):
    current = source_sha()
    drift = [k for k in labels if current.get(k) != _SOURCE_SHA[k]]
    if drift:
        raise RuntimeError(
            "quarantined_pivchol_alloc patches shipped source that has "
            "CHANGED: %s; re-derive the patch and update _SOURCE_SHA before "
            "running" % ", ".join(drift))


# ==========================================================================
# TARGET A -- piv_chol_step
# ==========================================================================

# Read at every call; set by ``active()`` and restored in its ``finally``.
_A1 = False
_A2 = False
_A3 = False
_A4 = False


def patched_piv_chol_step(H, grad, m):
    """``piv_chol_step`` with the allocation targets toggled by module flag.

    With every flag off this is the shipped body line for line (which is what
    the ``none`` arm of the A/B relies on).
    """
    STATS["pchol_calls"] += 1
    c, piv, rank, info = lapack.dpstrf(H, lower=1)
    rank = int(rank)
    # info > 0 is the NORMAL rank-deficient return of dpstrf, not a failure.
    if info < 0 or rank <= 0:
        raise np.linalg.LinAlgError("dpstrf rank 0 (info %d)" % info)
    if _A1:
        # a1: tril the m x rank SLICE.  tril keeps (i, j) iff i >= j, so
        # restricting j to [0, rank) commutes with tril exactly.
        L = np.tril(c[:, :rank])
    else:
        L = np.tril(c)[:, :rank]
    if _A3:
        # a3: only dl[0] and dl[-1] are ever read.  float(np.abs(x)) ==
        # abs(float(x)) exactly (abs clears the sign bit).
        first = abs(float(c[0, 0]))
        last = abs(float(c[rank - 1, rank - 1]))
    else:
        dl = np.abs(np.diag(c)[:rank])
        first = float(dl[0])
        last = float(dl[-1])
    clean = ((last ** 2)
             / max(m * np.finfo(float).eps * first ** 2, 1e-300))
    P = np.asarray(piv, dtype=int) - 1
    if _A4:
        U = None
    elif _A2:
        # a2: U[P] = L with P a full permutation is the gather U = L[Q] with
        # Q the inverse permutation; the zero-fill is dead.
        Q = np.empty(m, dtype=P.dtype)
        Q[P] = np.arange(m, dtype=P.dtype)
        U = np.take(L, Q, axis=0)
        STATS["a2_gathers"] += 1
    else:
        U = np.zeros((m, rank))
        U[P] = L
    p = m - rank
    if p > 0:
        W = sla.solve_triangular(L[:rank, :rank], L[rank:, :rank].T,
                                 lower=True, trans="T",
                                 check_finite=False).T
        core = W @ W.T
        core[np.diag_indices_from(core)] += 1.0
        chol = sla.cho_factor(core, lower=True, check_finite=False)
    else:
        W, chol = None, None
    gsolve = wfs._gram_apply(L[:rank, :rank], W, chol)
    if _A4:
        # EXPERIMENTAL.  Algebraically exact, but each dgemv now reduces over
        # m in pivot order instead of row order.
        a = L.T @ grad[P]
        z = gsolve(a)
        scattered = np.empty(m)
        scattered[P] = L @ z
        r_null = grad - scattered
        delta = np.empty(m)
        delta[P] = L @ gsolve(z)
    else:
        a = U.T @ grad
        z = gsolve(a)
        r_null = grad - U @ z
        delta = U @ gsolve(z)
    if not (np.all(np.isfinite(delta)) and np.all(np.isfinite(r_null))):
        raise np.linalg.LinAlgError("nonfinite piv-chol step")
    return delta, r_null, rank, clean


# ==========================================================================
# TARGET B -- FaceStepKernel.step support-churn accounting
# ==========================================================================

_B1 = False
_PREV_MASK = "_qpa_prev_mask"


def patched_kernel_step(self, S, idx, grad, gnorm, iter_lim):
    """RETAINED FOR REFERENCE; no longer installed by ``active``.

    b1_churn_mask was promoted into ``newton_oracle.FaceStepKernel.step``
    (flag ``newton_oracle.CHURN_MASK``), so the count is now done there and
    this wrapper is kept only as the derivation it came from.

    Do the DIAGNOSTIC churn count in O(n), then delegate verbatim.

    Sets ``self._previous = None`` so the shipped prologue skips its two
    ``setdiff1d`` calls and takes the branch that merely records ``idx``;
    nothing else in ``newton_oracle`` reads ``_previous``.
    """
    if _B1 and self._active:
        prev = getattr(self, _PREV_MASK, None)
        if prev is not None and self._previous is not None:
            STATS["b1_churn_calls"] += 1
            self.stats["support_changes"].append(
                int(np.count_nonzero(S != prev)))
        setattr(self, _PREV_MASK, S)
        # make the shipped prologue a no-op; it re-assigns _previous itself
        self._previous = None
    return FROZEN_KERNEL_STEP(self, S, idx, grad, gnorm, iter_lim)


patched_kernel_step.__wrapped__ = FROZEN_KERNEL_STEP


# ==========================================================================
# install / restore
# ==========================================================================

_ACTIVE = None


def normalize(names):
    """Accept a name, a sequence, 'all', 'default', 'none', or None."""
    if names is None:
        return ()
    if isinstance(names, str):
        names = (names,)
    out = []
    for name in names:
        if name == "all":
            out.extend(DEFAULT)
        elif name == "default":
            out.extend(DEFAULT)
        elif name in ("none", ""):
            continue
        elif name in TARGETS:
            out.append(name)
        else:
            raise ValueError("unknown target %r (known: %s)"
                             % (name, ", ".join(TARGETS)))
    return tuple(sorted(set(out), key=TARGETS.index))


def installed():
    """The installed target tuple, or ``None`` when nothing is installed."""
    return _ACTIVE


@contextlib.contextmanager
def active(names, allow_experimental=False):
    """Install EXACTLY the named targets; restore everything in ``finally``.

    Post-promotion a1/a2/a3/b1 are SET on the shipped modules rather than
    swapped in, and a target that is not named is forced to its PRE-PROMOTION
    value for the duration of the block.  ``a4_u_free`` is still quarantined
    and is still installed by rebinding ``wfs.piv_chol_step``.
    """
    global _ACTIVE, _A1, _A2, _A3, _A4, _B1
    if _ACTIVE is not None:
        raise RuntimeError("quarantined_pivchol_alloc.active is already "
                           "installed with %s; nesting is not supported"
                           % (_ACTIVE,))
    wanted = normalize(names)
    bad = [n for n in wanted if n in EXPERIMENTAL]
    if bad and not allow_experimental:
        raise RuntimeError(
            "%s is EXPERIMENTAL and is NOT byte-identical by construction "
            "(it reorders three dgemv reductions); pass "
            "allow_experimental=True to install it anyway"
            % ", ".join(bad))
    if wanted and ("a4_u_free" in wanted) and ("a2_u_gather" in wanted):
        raise RuntimeError("a4_u_free removes U; a2_u_gather rebuilds it -- "
                           "the two are mutually exclusive")
    labels = []
    if "a4_u_free" in wanted:
        labels += ["piv_chol_step", "_gram_apply"]
    _require_pristine(labels)

    saved = {"piv_chol_step": wfs.piv_chol_step,
             "a1": _A1, "a2": _A2, "a3": _A3, "a4": _A4, "b1": _B1}
    _ACTIVE = wanted
    try:
        with flag_scope(wanted):
            _A1 = "a1_tril_slice" in wanted
            _A2 = "a2_u_gather" in wanted
            _A3 = "a3_diag_scalar" in wanted
            _A4 = "a4_u_free" in wanted
            _B1 = "b1_churn_mask" in wanted
            if _A4:
                wfs.piv_chol_step = patched_piv_chol_step
            yield wanted
    finally:
        wfs.piv_chol_step = saved["piv_chol_step"]
        _A1, _A2, _A3 = saved["a1"], saved["a2"], saved["a3"]
        _A4, _B1 = saved["a4"], saved["b1"]
        _ACTIVE = None


def installed_state():
    """What is live RIGHT NOW, read off the modules -- not off intent."""
    return {
        "active": list(_ACTIVE or ()),
        "piv_chol_step": wfs.piv_chol_step.__name__,
        "kernel_step": newt.FaceStepKernel.__dict__["step"].__name__,
        "a1_tril_slice": bool(wfs.ALLOC_TRIL_SLICE),
        "a2_u_gather": bool(wfs.ALLOC_U_GATHER),
        "a3_diag_scalar": bool(wfs.ALLOC_DIAG_SCALAR),
        "a4_u_free": bool(_A4),
        "b1_churn_mask": bool(newt.CHURN_MASK),
    }


def assert_restored():
    if wfs.piv_chol_step is not FROZEN_PIV_CHOL_STEP:
        raise SystemExit("piv_chol_step not restored")
    if newt.FaceStepKernel.__dict__["step"] is not FROZEN_KERNEL_STEP:
        raise SystemExit("FaceStepKernel.step not restored")
    if _ACTIVE is not None:
        raise SystemExit("targets not restored")


# ==========================================================================
# byte-equality machinery
# ==========================================================================

def same_bytes(a, b):
    """Byte equality, strict about dtype, shape and NaN payload."""
    if isinstance(a, (int, float, np.floating, np.integer)) or a is None:
        if type(a) is not type(b):
            return False
        if a is None:
            return True
        return np.float64(a).tobytes() == np.float64(b).tobytes() \
            if isinstance(a, (float, np.floating)) else a == b
    a = np.asarray(a)
    b = np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def _compare_outputs(ref, new):
    """Field-by-field byte comparison of two ``piv_chol_step`` returns."""
    d_r, rn_r, rank_r, clean_r = ref
    d_n, rn_n, rank_n, clean_n = new
    return {
        "delta": bool(same_bytes(d_r, d_n)),
        "r_null": bool(same_bytes(rn_r, rn_n)),
        "rank": bool(rank_r == rank_n),
        "clean": bool(np.float64(clean_r).tobytes()
                      == np.float64(clean_n).tobytes()),
    }


def _max_abs_diff(ref, new):
    d_r, rn_r, _, clean_r = ref
    d_n, rn_n, _, clean_n = new
    if d_r.shape != d_n.shape:
        return None
    return {
        "delta": float(np.max(np.abs(d_r - d_n))) if d_r.size else 0.0,
        "r_null": float(np.max(np.abs(rn_r - rn_n))) if rn_r.size else 0.0,
        "clean": abs(float(clean_r) - float(clean_n)),
    }


def _two_arms(H0, grad, m, targets):
    """``(reference, treatment)`` for one captured ``H``, both on fresh copies.

    Reference = shipped ``piv_chol_step`` with every promoted flag OFF (the
    pre-promotion body).  Treatment = the same function with exactly
    ``targets`` on, except ``a4_u_free``, which is not shipped and is still
    served by this module's own implementation.
    """
    with flag_scope(()):
        ref = FROZEN_PIV_CHOL_STEP(np.array(H0, order="F", copy=True), grad, m)
    if "a4_u_free" in targets:
        global _A1, _A2, _A3, _A4
        saved = (_A1, _A2, _A3, _A4)
        _A1 = "a1_tril_slice" in targets
        _A2 = "a2_u_gather" in targets
        _A3 = "a3_diag_scalar" in targets
        _A4 = True
        try:
            new = patched_piv_chol_step(np.array(H0, order="F", copy=True),
                                        grad, m)
        finally:
            _A1, _A2, _A3, _A4 = saved
    else:
        with flag_scope(targets):
            new = FROZEN_PIV_CHOL_STEP(np.array(H0, order="F", copy=True),
                                       grad, m)
    return ref, new


class Differential:
    """Observation-only differential wrapper around ``piv_chol_step``.

    The SOLVER always receives the SHIPPED function's return value computed on
    the SHIPPED ``H`` buffer.  Two extra evaluations are made on copies and
    compared to each other; a third comparison (live vs shipped-on-a-copy) is
    the COPY BASELINE.
    """

    def __init__(self, targets, corpus_cap=0, corpus_max_m=512):
        self.targets = tuple(targets)
        self.calls = 0
        self.compared = 0
        self.exact = {"delta": 0, "r_null": 0, "rank": 0, "clean": 0}
        self.copy_baseline_exact = {"delta": 0, "r_null": 0, "rank": 0,
                                    "clean": 0}
        self.copy_baseline_n = 0
        self.structural = {"L_values_equal": 0, "U_values_equal": 0,
                           "n": 0}
        self.worst = {"delta": 0.0, "r_null": 0.0, "clean": 0.0}
        self.failures = []
        self.raises = 0
        self.shapes = {}
        self.corpus = []
        self.corpus_cap = int(corpus_cap)
        self.corpus_max_m = int(corpus_max_m)
        # churn (target B)
        self.churn_calls = 0
        self.churn_exact = 0
        self.churn_mismatch = []

    # -- structural identities, computed straight from a captured H --------
    def _structural(self, H, m):
        """Check the two COPY-level identities directly, on a copy of H.

        ``L`` values: ``np.tril(c)[:, :rank]`` vs ``np.tril(c[:, :rank])``.
        ``U`` values: ``zeros+scatter`` vs ``inverse-permutation gather``.
        Both are pure select/copy -- no arithmetic -- so ``array_equal`` here
        is a COPY check and is exact by construction if the algebra holds.
        """
        c, piv, rank, info = lapack.dpstrf(np.array(H, order="F", copy=True),
                                           lower=1)
        rank = int(rank)
        if info < 0 or rank <= 0:
            return
        L_old = np.tril(c)[:, :rank]
        L_new = np.tril(c[:, :rank])
        P = np.asarray(piv, dtype=int) - 1
        U_old = np.zeros((m, rank))
        U_old[P] = L_old
        Q = np.empty(m, dtype=P.dtype)
        Q[P] = np.arange(m, dtype=P.dtype)
        U_new = np.take(L_new, Q, axis=0)
        self.structural["n"] += 1
        self.structural["L_values_equal"] += int(
            L_old.shape == L_new.shape
            and L_old.tobytes() == np.ascontiguousarray(L_new).tobytes())
        self.structural["U_values_equal"] += int(
            U_old.tobytes() == np.ascontiguousarray(U_new).tobytes())
        # the permutation claim itself
        if not np.array_equal(P[Q[P]], P):
            self.failures.append({"why": "inverse permutation identity"})

    def wrapper(self, H, grad, m):
        self.calls += 1
        H0 = np.array(H, order="F", copy=True)
        try:
            live = FROZEN_PIV_CHOL_STEP(H, grad, m)
        except np.linalg.LinAlgError:
            self.raises += 1
            raise
        self.shapes[int(m)] = self.shapes.get(int(m), 0) + 1
        if self.corpus_cap and len(self.corpus) < self.corpus_cap \
                and m <= self.corpus_max_m:
            self.corpus.append((H0.copy(), np.array(grad, copy=True), int(m)))
        # --- both compared arms run on COPIES: alignment applies to both ---
        # REFERENCE: the shipped function with every promoted flag OFF, i.e.
        # the pre-promotion body.  TREATMENT: the same function with exactly
        # the named targets on.
        ref, new = _two_arms(H0, grad, m, self.targets)
        self.compared += 1
        verdict = _compare_outputs(ref, new)
        for key, ok in verdict.items():
            self.exact[key] += int(ok)
        diffs = _max_abs_diff(ref, new)
        if diffs is not None:
            for key in self.worst:
                self.worst[key] = max(self.worst[key], diffs[key])
        if not all(verdict.values()) and len(self.failures) < 12:
            self.failures.append({"m": int(m), "verdict": verdict,
                                  "max_abs_diff": diffs})
        # --- COPY BASELINE: shipped-on-real-H vs shipped-on-a-copy ---------
        self.copy_baseline_n += 1
        base = _compare_outputs(live, ref)
        for key, ok in base.items():
            self.copy_baseline_exact[key] += int(ok)
        self._structural(H0, m)
        return live

    # -- target B ---------------------------------------------------------
    def churn_check(self, kernel_self, S, idx):
        """Compute BOTH churn counts and compare.  Nothing is substituted."""
        if kernel_self._active and kernel_self._previous is not None:
            reference = int(
                np.setdiff1d(idx, kernel_self._previous,
                             assume_unique=True).size
                + np.setdiff1d(kernel_self._previous, idx,
                               assume_unique=True).size)
            prev_mask = getattr(kernel_self, "_qpa_diff_mask", None)
            if prev_mask is not None:
                candidate = int(np.count_nonzero(S != prev_mask))
                self.churn_calls += 1
                if candidate == reference:
                    self.churn_exact += 1
                elif len(self.churn_mismatch) < 12:
                    self.churn_mismatch.append(
                        {"reference": reference, "candidate": candidate})
        if kernel_self._active:
            kernel_self._qpa_diff_mask = S

    def report(self):
        return {
            "targets": list(self.targets),
            "piv_chol_calls": self.calls,
            "compared": self.compared,
            "byte_exact": dict(self.exact),
            "byte_exact_all_fields": min(self.exact.values())
            if self.compared else 0,
            "copy_baseline_n": self.copy_baseline_n,
            "copy_baseline_byte_exact": dict(self.copy_baseline_exact),
            "structural_copy_checks": dict(self.structural),
            "max_abs_diff_over_corpus": dict(self.worst),
            "linalg_errors_from_shipped": self.raises,
            "m_histogram": dict(sorted(self.shapes.items())),
            "failures": self.failures,
            "churn_compared": self.churn_calls,
            "churn_byte_exact": self.churn_exact,
            "churn_mismatch": self.churn_mismatch,
        }


def _make_step_wrapper(diff):
    """A PLAIN function (a bound method is not a descriptor and would not
    receive ``self`` when set as a class attribute)."""

    def traced(self, S, idx, grad, gnorm, iter_lim):
        diff.churn_check(self, S, idx)
        return FROZEN_KERNEL_STEP(self, S, idx, grad, gnorm, iter_lim)

    traced.__wrapped__ = FROZEN_KERNEL_STEP
    return traced


def _make_pchol_wrapper(diff):
    def traced(H, grad, m):
        return diff.wrapper(H, grad, m)
    traced.__wrapped__ = FROZEN_PIV_CHOL_STEP
    return traced


@contextlib.contextmanager
def differential(targets, corpus_cap=0):
    """Install the differential wrapper; restore in ``finally``."""
    diff = Differential(targets, corpus_cap=corpus_cap)
    saved_p = wfs.piv_chol_step
    saved_s = newt.FaceStepKernel.__dict__["step"]
    wfs.piv_chol_step = _make_pchol_wrapper(diff)
    newt.FaceStepKernel.step = _make_step_wrapper(diff)
    try:
        yield diff
    finally:
        wfs.piv_chol_step = saved_p
        newt.FaceStepKernel.step = saved_s


def replay(corpus, targets):
    """Offline replay over a captured corpus.  Returns the same shaped dict."""
    diff = Differential(targets)
    for H, grad, m in corpus:
        H0 = np.array(H, order="F", copy=True)
        try:
            ref, new = _two_arms(H0, grad, m, targets)
        except np.linalg.LinAlgError:
            diff.raises += 1
            continue
        diff.calls += 1
        diff.compared += 1
        verdict = _compare_outputs(ref, new)
        for key, ok in verdict.items():
            diff.exact[key] += int(ok)
        diffs = _max_abs_diff(ref, new)
        if diffs is not None:
            for key in diff.worst:
                diff.worst[key] = max(diff.worst[key], diffs[key])
        if not all(verdict.values()) and len(diff.failures) < 12:
            diff.failures.append({"m": int(m), "verdict": verdict,
                                  "max_abs_diff": diffs})
        diff._structural(H0, m)
    return diff.report()


# ==========================================================================
# drivers
# ==========================================================================

CHEAP_MODELS = ("afiro", "sc50a", "adlittle", "beaconfd", "scorpion")


def _solve_once(model, seconds=300, ruiz_rounds=None):
    """One solve of ``model`` under the A/B's own control configuration."""
    import bench_transforms as bt
    B, b, d = bt._load(model)
    rounds = bt.RUIZ_ROUNDS if ruiz_rounds is None else ruiz_rounds
    Bt, bt_, dt, r, c, _info = bt.equilibrate(B, b, d, rounds=rounds)
    row, x_t, y_t = bt._solve_transformed(model, Bt, bt_, dt, seconds, 0, 0)
    return row, x_t, y_t


def run_differential(models, targets, seconds=300, corpus_cap=0):
    out = {"models": {}, "targets": list(targets)}
    totals = Differential(targets).report()
    for key in ("piv_chol_calls", "compared", "churn_compared",
                "churn_byte_exact"):
        totals[key] = 0
    totals["byte_exact"] = {k: 0 for k in totals["byte_exact"]}
    totals["copy_baseline_byte_exact"] = {
        k: 0 for k in totals["copy_baseline_byte_exact"]}
    totals["copy_baseline_n"] = 0
    totals["structural_copy_checks"] = {"L_values_equal": 0,
                                        "U_values_equal": 0, "n": 0}
    totals["failures"] = []
    totals["churn_mismatch"] = []
    totals["m_histogram"] = {}
    totals["linalg_errors_from_shipped"] = 0
    corpus = []
    for model in models:
        started = time.perf_counter()
        with differential(targets, corpus_cap=corpus_cap) as diff:
            row, _x, _y = _solve_once(model, seconds=seconds)
        assert_restored()
        rep = diff.report()
        rep["solve_status"] = row.get("status")
        rep["solve_seconds_CONTAMINATED"] = round(
            time.perf_counter() - started, 3)
        out["models"][model] = rep
        corpus.extend(diff.corpus)
        for key in ("piv_chol_calls", "compared", "churn_compared",
                    "churn_byte_exact", "copy_baseline_n"):
            totals[key] += rep[key]
        for key in totals["byte_exact"]:
            totals["byte_exact"][key] += rep["byte_exact"][key]
            totals["copy_baseline_byte_exact"][key] += \
                rep["copy_baseline_byte_exact"][key]
        for key in totals["structural_copy_checks"]:
            totals["structural_copy_checks"][key] += \
                rep["structural_copy_checks"][key]
        for key in ("delta", "r_null", "clean"):
            totals["max_abs_diff_over_corpus"][key] = max(
                totals["max_abs_diff_over_corpus"][key],
                rep["max_abs_diff_over_corpus"][key])
        totals["failures"].extend(rep["failures"])
        totals["churn_mismatch"].extend(rep["churn_mismatch"])
        totals["linalg_errors_from_shipped"] += \
            rep["linalg_errors_from_shipped"]
        for key, count in rep["m_histogram"].items():
            totals["m_histogram"][key] = \
                totals["m_histogram"].get(key, 0) + count
    totals["byte_exact_all_fields"] = min(totals["byte_exact"].values()) \
        if totals["compared"] else 0
    totals["ALL_BYTE_EXACT"] = bool(
        totals["compared"] > 0
        and all(v == totals["compared"] for v in totals["byte_exact"].values())
        and totals["churn_compared"] > 0
        and totals["churn_byte_exact"] == totals["churn_compared"])
    totals["COPY_BASELINE_CLEAN"] = bool(
        totals["copy_baseline_n"] > 0
        and all(v == totals["copy_baseline_n"]
                for v in totals["copy_baseline_byte_exact"].values()))
    out["totals"] = totals
    out["corpus_size"] = len(corpus)
    return out, corpus


def self_check():
    """Cheap, model-free checks of every algebraic claim in the docstring."""
    rng = np.random.default_rng(11)
    report = {"cases": [], "n_exact": 0, "n": 0}
    for m, defic in ((40, 0), (60, 7), (90, 23), (25, 12)):
        k = m - defic
        A = rng.standard_normal((max(k, 1), m))
        if defic:
            A = A[:k]
        H = A.T @ A
        H = np.asfortranarray(0.5 * (H + H.T))
        grad = rng.standard_normal(m)
        ref = FROZEN_PIV_CHOL_STEP(np.array(H, order="F", copy=True), grad, m)
        for arm in (DEFAULT, ("a1_tril_slice",), ("a2_u_gather",),
                    ("a3_diag_scalar",), ()):
            global _A1, _A2, _A3, _A4
            saved = (_A1, _A2, _A3, _A4)
            _A1 = "a1_tril_slice" in arm
            _A2 = "a2_u_gather" in arm
            _A3 = "a3_diag_scalar" in arm
            _A4 = False
            try:
                new = patched_piv_chol_step(
                    np.array(H, order="F", copy=True), grad, m)
            finally:
                _A1, _A2, _A3, _A4 = saved
            verdict = _compare_outputs(ref, new)
            report["n"] += 1
            report["n_exact"] += int(all(verdict.values()))
            report["cases"].append({"m": m, "deficiency": defic,
                                    "arm": list(arm), "verdict": verdict,
                                    "rank": int(ref[2])})
    # target B identity, on random supports
    n = 5000
    churn_ok = 0
    for _ in range(200):
        S1 = rng.random(n) > 0.4
        S2 = rng.random(n) > 0.4
        i1, i2 = np.where(S1)[0], np.where(S2)[0]
        reference = int(np.setdiff1d(i1, i2, assume_unique=True).size
                        + np.setdiff1d(i2, i1, assume_unique=True).size)
        churn_ok += int(reference == int(np.count_nonzero(S1 != S2)))
    report["churn_identity"] = {"trials": 200, "exact": churn_ok}
    report["all_exact"] = bool(report["n_exact"] == report["n"]
                               and churn_ok == 200)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="quarantined allocation patches for the pivoted-Cholesky "
                    "face solver; correctness only, no timing")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--differential", action="store_true")
    parser.add_argument("--models", nargs="*", default=list(CHEAP_MODELS))
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT))
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--corpus-cap", type=int, default=0,
                        help="save up to N captured (H, grad, m) for replay")
    parser.add_argument("--corpus-out", type=Path, default=None)
    parser.add_argument("--replay", type=Path, default=None,
                        help="replay a corpus .npz written by --corpus-out")
    parser.add_argument("--print-source-sha", action="store_true")
    args = parser.parse_args()

    if args.print_source_sha:
        print(json.dumps(source_sha(), indent=2, sort_keys=True))
        return
    out = {"installed_before": installed_state()}
    if args.self_check:
        out["self_check"] = self_check()
    if args.replay is not None:
        data = np.load(args.replay)
        n = int(data["n"])
        corpus = [(data["H%d" % i], data["g%d" % i], int(data["m%d" % i]))
                  for i in range(n)]
        out["replay"] = replay(corpus, tuple(args.targets))
        out["replay"]["corpus"] = str(args.replay)
    if args.differential:
        targets = tuple(normalize(args.targets)) if "a4_u_free" not in \
            args.targets else tuple(args.targets)
        rep, corpus = run_differential(args.models, targets,
                                       seconds=args.seconds,
                                       corpus_cap=args.corpus_cap)
        out["differential"] = rep
        if args.corpus_out is not None and corpus:
            args.corpus_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {"n": len(corpus)}
            for i, (H, g, m) in enumerate(corpus):
                payload["H%d" % i] = H
                payload["g%d" % i] = g
                payload["m%d" % i] = m
            np.savez(args.corpus_out, **payload)
            out["corpus_written"] = str(args.corpus_out)
    assert_restored()
    out["installed_after"] = installed_state()
    out["source_sha"] = source_sha()
    print(json.dumps(out, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
