"""presolve_walker: the PRESOLVED-WALKER lane.

Run the promoted selector-gated dual-path walker on the HiGHS-PRESOLVED
program, lift the terminal primal-dual pair back to the original space with
HiGHS ``postsolve``, and certify with the FROZEN gates on the ORIGINAL
``to_Bxgeb`` data.

    original  (B, b, d)            min d'x  s.t. Bx >= b, x free
       |  highspy presolve
    reduced   HighsLp (general)    min c'x  s.t. rl <= Ax <= ru, cl <= x <= cu
       |  reduced_to_Bxgeb  (the SAME syntactic recipe as
       |                     exp11_netlib_rank_certificate.to_Bxgeb)
    reduced'  (B_r, b_r, d_r)      min d_r'x s.t. B_r x >= b_r, x free
       |  selector_gated_walker.follow_selector_gated(..., factor_update=True)
    pair      (x_r, y_r)
       |  reduced_pair_to_highs  (our rows -> HiGHS reduced row/col duals)
       |  highspy postsolve
    lifted    (x0, y0)  in the ORIGINAL space
       |  exp23_path_primal_dual.certificate_pair(B, b, d, x0, y0)   <-- FROZEN
    verdict

WHAT COUNTS AS A RESULT.  Only the last step.  ``certificate_pair`` /
``_certificate_detail`` evaluated on the ORIGINAL ``B, b, d`` is the sole
authority that may declare CERTIFIED.  The reduced-space certificate that this
module also computes (``reduced_certificate``) is an INTERMEDIATE diagnostic --
it is recorded so that lift degradation can be measured, and it is never
allowed to set the verdict.

SIGN / ORDER CONVENTIONS -- verified empirically, not assumed.

  * The HiGHS reduced LP satisfies ``A' lam + z = c`` with ``lam`` =
    ``solution.row_dual`` and ``z`` = ``solution.col_dual``.  Measured on
    afiro: ``||A' lam + z - c||inf = 5.6e-17`` from HiGHS's own reduced
    solution (``pw_probe2``).  This is the same unnegated convention that
    ``oracle_terminal._oracle_solve_highspy`` documents for the row-lower-only
    model, extended to two-sided rows and finite column bounds.

  * ``reduced_to_Bxgeb`` emits ``+a_i`` for a finite ``rl_i`` and ``-a_i`` for
    a finite ``ru_i``, ``+e_j`` for a finite ``cl_j`` and ``-e_j`` for a
    finite ``cu_j``.  Consequently ``B_r' y = d_r`` expands EXACTLY into
    ``A' lam + z = c`` under ``lam_i = y[row_lo_i] - y[row_up_i]`` and
    ``z_j = y[col_lo_j] - y[col_up_j]``.  No further sign work is needed.

  * ``Highs.postsolve`` accepts a manually built ``HighsSolution`` with no
    basis (overload 2), and returns an original-space pair through
    ``Highs.getSolution()``.  Verified on afiro: the round trip of HiGHS's own
    reduced solution certified on the original data at primal 1.1e-14,
    dual 1.7e-16, gap 1.2e-16 (``pw_probe2``).

FAIL-CLOSED.  Any presolve, conversion, walk, lift or gate failure falls back
to the raw walk on the original program and the route is recorded.
"""

from __future__ import annotations

import time

import numpy as np

import exp23_path_primal_dual as exp23

# The promoted engine's defaults, copied from
# exp23_path_primal_dual._follow_selector_engine (lines 441-455), plus the
# committed fast lane ``factor_update=True``.
ENGINE_DEFAULTS = dict(legacy_repair=False, ratio_multiplier="min_norm",
                       dual_ascent_repair=True, repair_order="adaptive",
                       drift_guard=True, factor_update=True)

# Bookkeeping row kinds emitted by ``reduced_to_Bxgeb``.
ROW_LO, ROW_UP, COL_LO, COL_UP = 0, 1, 2, 3


# --------------------------------------------------------------- HiGHS model

def build_highs_lp(B, b, d):
    """``min d'x s.t. Bx >= b``, x free, as a ``highspy.HighsLp``.

    Byte-for-byte the model of ``oracle_terminal._oracle_solve_highspy``
    (lines 220-238): free columns, ``row_lower_ = b``, ``row_upper_ = +inf``,
    column-wise CSC of ``B``, minimize.
    """
    import highspy
    from scipy.sparse import csc_matrix

    n, m = B.shape
    matrix = csc_matrix(np.asarray(B, dtype=float))
    lp = highspy.HighsLp()
    lp.num_col_ = m
    lp.num_row_ = n
    lp.col_cost_ = np.asarray(d, dtype=float)
    lp.col_lower_ = np.full(m, -highspy.kHighsInf)
    lp.col_upper_ = np.full(m, highspy.kHighsInf)
    lp.row_lower_ = np.asarray(b, dtype=float)
    lp.row_upper_ = np.full(n, highspy.kHighsInf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
    lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
    lp.a_matrix_.value_ = matrix.data.astype(float)
    lp.sense_ = highspy.ObjSense.kMinimize
    return lp


def presolve_program(B, b, d, threads=1):
    """Presolve the original program; return the handle, reduced LP and stats.

    The ``Highs`` handle is returned live: ``postsolve`` needs the same
    instance, because the postsolve stack lives inside it.
    """
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", int(threads))
    h.passModel(build_highs_lp(B, b, d))

    started = time.perf_counter()
    status = h.presolve()
    seconds = time.perf_counter() - started

    rlp = h.getPresolvedLp()
    presolve_status = str(h.getModelPresolveStatus())
    stats = {
        "presolve_seconds": seconds,
        "highs_status": str(status),
        "presolve_status": presolve_status,
        "rows_before": int(B.shape[0]),
        "cols_before": int(B.shape[1]),
        "highs_reduced_rows": int(rlp.num_row_),
        "highs_reduced_cols": int(rlp.num_col_),
        "reduced_offset": float(rlp.offset_),
        "reduced_sense_minimize": bool(rlp.sense_ == highspy.ObjSense.kMinimize),
    }
    return h, rlp, stats


def _reduced_matrix(rlp):
    """Dense ``A_r`` from the reduced LP, handling either storage format."""
    import highspy
    from scipy.sparse import csc_matrix, csr_matrix

    value = np.asarray(rlp.a_matrix_.value_, dtype=float)
    index = np.asarray(rlp.a_matrix_.index_, dtype=np.int64)
    start = np.asarray(rlp.a_matrix_.start_, dtype=np.int64)
    shape = (int(rlp.num_row_), int(rlp.num_col_))
    if shape[0] == 0 or shape[1] == 0:
        return np.zeros(shape)
    if rlp.a_matrix_.format_ == highspy.MatrixFormat.kColwise:
        A = csc_matrix((value, index, start[: shape[1] + 1]), shape=shape)
    else:
        A = csr_matrix((value, index, start[: shape[0] + 1]), shape=shape)
    return np.asarray(A.todense())


def structural_check(B_r, b_r, d_r):
    """Structural sanity of the converted reduced program.

    A structurally singular sparse pattern (a row or a column with no stored
    nonzero) is what makes ``scipy.sparse.linalg.splu`` abort the PROCESS
    rather than raise, so it is worth detecting even though this lane cannot
    reach ``splu``: with ``ENGINE_DEFAULTS`` the walk uses ``unit_core=False``
    (the only ``splu`` site in ``unit_core_face_solver``, line 274, is behind
    that flag) and ``init_seed=None`` (the ``splu`` in
    ``highs_phase1_basis``, line 66, is behind that one), while
    ``face_factor_updating`` / ``path_piece_factor`` / ``path_piece_skeleton``
    are dense LAPACK, ``dual_ascent_repair`` has no ``splu``, and
    ``critical_cone_path._prepare`` reaches only
    ``sparse_basis_projection.reduce_equalities_sparse``, which is a DENSE
    pivoted QR.  The counts are recorded so a future configuration change
    cannot silently walk into the sparse path with a bad pattern.
    """
    empty_rows = empty_cols = 0
    if B_r.size:
        empty_rows = int((np.abs(B_r).max(axis=1) == 0.0).sum())
        empty_cols = int((np.abs(B_r).max(axis=0) == 0.0).sum())
    finite = (bool(np.all(np.isfinite(B_r))) and bool(np.all(np.isfinite(b_r)))
              and bool(np.all(np.isfinite(d_r))))
    return {"struct_empty_rows": empty_rows, "struct_empty_cols": empty_cols,
            "struct_all_finite": finite,
            "struct_ok": bool(empty_rows == 0 and empty_cols == 0 and finite)}


# ------------------------------------------------- reduced LP -> Bx >= b form

def reduced_to_Bxgeb(rlp):
    """Convert the HiGHS reduced LP to ``B_r x >= b_r`` with x free.

    The recipe is the one ``exp11_netlib_rank_certificate.to_Bxgeb`` applies to
    an MPS file (lines 100-128), transcribed to the in-memory general form:

        row  a'x >= rl   (finite rl)   ->   ( +a, +rl)
        row  a'x <= ru   (finite ru)   ->   ( -a, -ru)
        col  x_j >= cl_j (finite cl)   ->   (+e_j, +cl_j)
        col  x_j <= cu_j (finite cu)   ->   (-e_j, -cu_j)

    An E row (rl == ru) therefore emits the pair (+a, +r), (-a, -r) exactly as
    to_Bxgeb does; a RANGE row (rl < ru, both finite) emits the same two rows
    with the two distinct right-hand sides, again matching to_Bxgeb.  Free rows
    and free columns emit nothing.  Zero rows are NOT dropped -- to_Bxgeb does
    not drop them either -- but they are counted.

    Returns ``(B_r, b_r, d_r, book)``.  ``book`` carries, per emitted row, the
    kind (ROW_LO / ROW_UP / COL_LO / COL_UP) and the HiGHS reduced row or
    column index, which is what makes the dual lift possible.
    """
    import highspy

    INF = highspy.kHighsInf
    n_r, m_r = int(rlp.num_row_), int(rlp.num_col_)
    A = _reduced_matrix(rlp)
    rl = np.asarray(rlp.row_lower_, dtype=float)[:n_r]
    ru = np.asarray(rlp.row_upper_, dtype=float)[:n_r]
    cl = np.asarray(rlp.col_lower_, dtype=float)[:m_r]
    cu = np.asarray(rlp.col_upper_, dtype=float)[:m_r]
    cost = np.asarray(rlp.col_cost_, dtype=float)[:m_r]

    rows, rhs, kinds, owners = [], [], [], []
    n_eq = n_range = n_free_row = 0
    for i in range(n_r):
        lo_ok, up_ok = rl[i] > -INF, ru[i] < INF
        if lo_ok:
            rows.append(A[i]); rhs.append(rl[i])
            kinds.append(ROW_LO); owners.append(i)
        if up_ok:
            rows.append(-A[i]); rhs.append(-ru[i])
            kinds.append(ROW_UP); owners.append(i)
        if lo_ok and up_ok:
            if abs(ru[i] - rl[i]) <= 1e-12 * max(1.0, abs(rl[i])):
                n_eq += 1
            else:
                n_range += 1
        elif not lo_ok and not up_ok:
            n_free_row += 1

    n_fixed_col = n_free_col = 0
    for j in range(m_r):
        lo_ok, up_ok = cl[j] > -INF, cu[j] < INF
        if lo_ok:
            e = np.zeros(m_r); e[j] = 1.0
            rows.append(e); rhs.append(cl[j])
            kinds.append(COL_LO); owners.append(j)
        if up_ok:
            e = np.zeros(m_r); e[j] = -1.0
            rows.append(e); rhs.append(-cu[j])
            kinds.append(COL_UP); owners.append(j)
        if lo_ok and up_ok and abs(cu[j] - cl[j]) <= 1e-12 * max(1.0, abs(cl[j])):
            n_fixed_col += 1
        elif not lo_ok and not up_ok:
            n_free_col += 1

    B_r = np.array(rows, dtype=float) if rows else np.zeros((0, m_r))
    b_r = np.array(rhs, dtype=float) if rhs else np.zeros(0)
    zero_rows = int((np.abs(A).max(axis=1) == 0.0).sum()) if n_r else 0
    struct = structural_check(B_r, b_r, cost)
    book = {
        "kind": np.array(kinds, dtype=np.int64),
        "owner": np.array(owners, dtype=np.int64),
        "A": A,
        "n_highs_rows": n_r,
        "n_highs_cols": m_r,
        "stats": {
            "our_rows": int(B_r.shape[0]),
            "our_cols": int(m_r),
            "highs_equality_rows": n_eq,
            "highs_range_rows": n_range,
            "highs_free_rows": n_free_row,
            "highs_fixed_cols": n_fixed_col,
            "highs_free_cols": n_free_col,
            "highs_zero_rows": zero_rows,
            "col_lower_finite": int((cl > -INF).sum()),
            "col_upper_finite": int((cu < INF).sum()),
            **struct,
        },
    }
    return B_r, b_r, cost, book


# ------------------------------------------------------------------- the lift

def reduced_pair_to_highs(book, x_r, y_r):
    """Map OUR reduced pair onto HiGHS reduced-LP solution vectors.

    ``B_r' y = d_r`` expands term by term into HiGHS's ``A' lam + z = c``:
    a G-side row contributes ``+a_i y_k`` and an L-side row ``-a_i y_k``, so
    ``lam_i = y[row_lo_i] - y[row_up_i]``; a lower-bound row contributes
    ``+e_j y_k`` and an upper-bound row ``-e_j y_k``, so
    ``z_j = y[col_lo_j] - y[col_up_j]``.  An E row's two duals therefore give
    the free-sign equality multiplier as their difference, which is the
    standard reduction and needs no special case.
    """
    kind, owner = book["kind"], book["owner"]
    lam = np.zeros(book["n_highs_rows"])
    z = np.zeros(book["n_highs_cols"])
    y = np.asarray(y_r, dtype=float)
    if y.size:
        np.add.at(lam, owner[kind == ROW_LO], y[kind == ROW_LO])
        np.add.at(lam, owner[kind == ROW_UP], -y[kind == ROW_UP])
        np.add.at(z, owner[kind == COL_LO], y[kind == COL_LO])
        np.add.at(z, owner[kind == COL_UP], -y[kind == COL_UP])
    x = np.asarray(x_r, dtype=float)
    row_value = book["A"] @ x if book["n_highs_rows"] else np.zeros(0)
    return x, z, row_value, lam


def postsolve_pair(h, x_r, z, row_value, lam):
    """Push a reduced-LP solution through HiGHS postsolve; read the original.

    Uses the no-basis overload (``postsolve(HighsSolution)``); the basis
    overload is unavailable to us because the walker returns no basis.
    """
    import highspy

    sol = highspy.HighsSolution()
    sol.col_value = list(np.asarray(x_r, dtype=float))
    sol.col_dual = list(np.asarray(z, dtype=float))
    sol.row_value = list(np.asarray(row_value, dtype=float))
    sol.row_dual = list(np.asarray(lam, dtype=float))
    sol.value_valid = True
    sol.dual_valid = True

    started = time.perf_counter()
    status = h.postsolve(sol)
    seconds = time.perf_counter() - started
    lifted = h.getSolution()
    x0 = np.asarray(lifted.col_value, dtype=float)
    y0 = np.asarray(lifted.row_dual, dtype=float)
    info = {
        "postsolve_seconds": seconds,
        "postsolve_status": str(status),
        "postsolve_model_status": h.modelStatusToString(h.getModelStatus()),
        "x0_size": int(x0.size),
        "y0_size": int(y0.size),
    }
    return x0, y0, info


# ---------------------------------------------------------------- residuals

def residuals(B, b, d, x, y):
    """The FROZEN gate's own componentwise error vector, pass or fail.

    Returns ``(passed, detail)`` straight from ``exp23.certificate_pair`` so
    every number in the report is the gate's number, not a re-derivation.
    """
    ok, detail = exp23.certificate_pair(B, b, d, x, y)
    out = {"pass": bool(ok), "reason": detail.get("reason")}
    for key in ("primal", "dual", "nonnegative", "gap"):
        if key in detail:
            out[key] = float(detail[key])
    return ok, out


def _min_y(y):
    y = np.asarray(y, dtype=float)
    return float(y.min()) if y.size else 0.0


# --------------------------------------------------------------- the walkers

def walk(B, b, d, maxpiv=4000, tmax=1e8, stats=None, progress=None,
         trace=None, **overrides):
    """One promoted-engine walk (no HiGHS fallback), timed."""
    from selector_gated_walker import follow_selector_gated

    kwargs = dict(ENGINE_DEFAULTS)
    kwargs.update(overrides)
    started = time.perf_counter()
    st = follow_selector_gated(B, b, d, t=1.0, maxpiv=maxpiv, tmax=tmax,
                               trace=trace, progress=progress, stats=stats,
                               **kwargs)
    st["walk_seconds"] = time.perf_counter() - started
    return st


def run_presolved(B, b, d, maxpiv=4000, tmax=1e8, stats=None,
                  reduced_progress=None):
    """Presolve -> convert -> walk reduced -> lift -> certify on ORIGINAL data.

    Returns a record.  ``record["status"]`` is CERTIFIED only when
    ``exp23.certificate_pair`` passes on the ORIGINAL ``B, b, d``.
    """
    record = {"route": "presolve-walk-postsolve", "lane": "presolved-walker"}
    total0 = time.perf_counter()

    h, rlp, pstats = presolve_program(B, b, d)
    record.update(pstats)

    started = time.perf_counter()
    B_r, b_r, d_r, book = reduced_to_Bxgeb(rlp)
    record["convert_seconds"] = time.perf_counter() - started
    record.update(book["stats"])
    record["reduced_shape"] = [int(B_r.shape[0]), int(B_r.shape[1])]
    record["row_reduction"] = (float(B_r.shape[0]) / B.shape[0]
                               if B.shape[0] else float("nan"))
    record["col_reduction"] = (float(B_r.shape[1]) / B.shape[1]
                               if B.shape[1] else float("nan"))

    # Presolve may finish the problem outright; there is then nothing to walk.
    if B_r.shape[1] == 0 or B_r.shape[0] == 0:
        record["route"] = "presolve-solved"
        record["reduced_pivots"] = 0
        record["reduced_walk_seconds"] = 0.0
        record["reduced_status"] = "EMPTY"
        x_r = np.zeros(int(B_r.shape[1]))
        y_r = np.zeros(int(B_r.shape[0]))
        record["reduced_certificate"] = {"pass": None,
                                         "reason": "empty reduced program"}
    else:
        st = walk(B_r, b_r, d_r, maxpiv=maxpiv, tmax=tmax, stats=stats,
                  progress=reduced_progress)
        record["reduced_status"] = st.get("status")
        record["reduced_pivots"] = int(st.get("pivots", -1))
        record["reduced_walk_seconds"] = float(st["walk_seconds"])
        record["reduced_backend"] = st.get("backend")
        record["reduced_t"] = (float(st["t"]) if st.get("t") is not None
                               else None)
        record["reduced_lb"] = (float(st["lb"]) if st.get("lb") is not None
                                else None)
        record["reduced_objective_plus_offset"] = (
            None if st.get("lb") is None
            else float(st["lb"]) + record["reduced_offset"])
        x_r = st.get("x")
        y_r = st.get("y")
        if x_r is None or y_r is None:
            record["status"] = "REDUCED-WALK-FAILED"
            record["failure"] = "reduced walk returned no pair"
            record["total_seconds"] = time.perf_counter() - total0
            return record
        x_r = np.asarray(x_r, dtype=float)
        y_r = np.asarray(y_r, dtype=float)
        # INTERMEDIATE ONLY -- never the verdict.  Recorded so the report can
        # quantify how much (if anything) the lift costs in accuracy.
        _, rdet = residuals(B_r, b_r, d_r, x_r, y_r)
        rdet["min_y"] = _min_y(y_r)
        record["reduced_certificate"] = rdet

    started = time.perf_counter()
    try:
        hx, hz, hrv, hlam = reduced_pair_to_highs(book, x_r, y_r)
        # Consistency of the mapping itself, before HiGHS ever sees it.
        c = np.asarray(rlp.col_cost_, dtype=float)[: book["n_highs_cols"]]
        stat = (float(np.max(np.abs(book["A"].T @ hlam + hz - c)))
                if book["n_highs_rows"] and book["n_highs_cols"] else 0.0)
        record["reduced_stationarity_residual"] = stat
        x0, y0, linfo = postsolve_pair(h, hx, hz, hrv, hlam)
        record.update(linfo)
    except Exception as error:                      # fail closed
        record["status"] = "LIFT-FAILED"
        record["failure"] = f"{type(error).__name__}: {error}"
        record["lift_seconds"] = time.perf_counter() - started
        record["total_seconds"] = time.perf_counter() - total0
        return record
    record["lift_seconds"] = time.perf_counter() - started

    if x0.size != B.shape[1] or y0.size != B.shape[0]:
        record["status"] = "LIFT-FAILED"
        record["failure"] = (f"postsolve shape {x0.size}/{y0.size} vs "
                             f"{B.shape[1]}/{B.shape[0]}")
        record["total_seconds"] = time.perf_counter() - total0
        return record

    started = time.perf_counter()
    ok, detail = residuals(B, b, d, x0, y0)          # <-- THE FROZEN GATE
    detail["min_y"] = _min_y(y0)
    rok, _rx, rerr, rreason = exp23._certificate_detail(B, b, d, y0)
    record["certify_seconds"] = time.perf_counter() - started
    record["certificate"] = {
        "pair_pass": bool(ok), "pair_detail": detail,
        "recheck_pass": bool(rok), "recheck_error": float(rerr),
        "recheck_reason": rreason,
        "objective": float(d @ x0), "lb": float(b @ y0),
    }
    record["status"] = "CERTIFIED" if ok else "GATE-FAILED"
    record["lb"] = float(b @ y0)
    record["x"] = x0
    record["y"] = y0
    record["total_seconds"] = time.perf_counter() - total0
    return record


def run_raw(B, b, d, maxpiv=4000, tmax=1e8, stats=None, progress=None):
    """The fail-closed comparison lane: the walk on the ORIGINAL program."""
    record = {"route": "raw-walk", "lane": "raw-walker"}
    total0 = time.perf_counter()
    st = walk(B, b, d, maxpiv=maxpiv, tmax=tmax, stats=stats,
              progress=progress)
    record["raw_status"] = st.get("status")
    record["raw_pivots"] = int(st.get("pivots", -1))
    record["raw_walk_seconds"] = float(st["walk_seconds"])
    x = st.get("x")
    y = st.get("y")
    if x is None or y is None:
        record["status"] = "RAW-WALK-FAILED"
        record["total_seconds"] = time.perf_counter() - total0
        return record
    ok, detail = residuals(B, b, d, np.asarray(x, float), np.asarray(y, float))
    detail["min_y"] = _min_y(y)
    rok, _rx, rerr, rreason = exp23._certificate_detail(B, b, d,
                                                        np.asarray(y, float))
    record["certificate"] = {
        "pair_pass": bool(ok), "pair_detail": detail,
        "recheck_pass": bool(rok), "recheck_error": float(rerr),
        "recheck_reason": rreason,
        "objective": float(d @ np.asarray(x, float)),
        "lb": float(b @ np.asarray(y, float)),
    }
    record["status"] = "CERTIFIED" if ok else "GATE-FAILED"
    record["lb"] = float(b @ np.asarray(y, float))
    record["x"] = np.asarray(x, float)
    record["y"] = np.asarray(y, float)
    record["total_seconds"] = time.perf_counter() - total0
    return record
