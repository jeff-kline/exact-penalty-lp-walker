"""exp11: the rank certificates on real (structured, sparse) Netlib LPs.

The theorem (assets/10) gives EXACT, genericity-free certificates:

    flats_1 = |A(x)|  - rank(B_{A(x)})        item 1 transverse  <=>  flats_1 = 0  (LICQ at x)
    flats_2 = m       - rank(B_{supp(y)})     item 2 transverse  <=>  flats_2 = 0  (Mangasarian (18))

Every COUNTING form of these (|A| vs m, alpha vs 2, a vs 1) relies on general position and
is therefore worthless on Netlib, where B is sparse and structured.  The rank forms are not.
So Netlib is the one place the theory can actually break, and Paper 2's Table 7 already
reports what Thm 8.1 / 8.2 did on afiro, degen2 and fit1d.

We convert each Netlib LP to the paper's form  min d'x s.t. Bx >= b  (x free in R^m):
    E row a'x = r    ->   a'x >= r  and  -a'x >= -r
    G row a'x >= r   ->   a'x >= r
    L row a'x <= r   ->   -a'x >= -r
    RANGES           ->   the second inequality of the pair, per MPS semantics
    bound l <= x_j   ->   e_j'x >= l ;   x_j <= u  ->  -e_j'x >= -u
Solve with HiGHS, take the active set at the optimum, and compute both certificates.

For flats_2 we need the LEAST-NORM dual.  Its support lies in A(x), and on that index set
the least-norm nonneg solution of B_A' y = d is the rho -> 0 limit of
    min ||B_A' y - d||^2 + rho ||y||^2 ,  y >= 0
which is NNLS on [[B_A'],[sqrt(rho) I]] vs [d, 0].  We also report the HiGHS dual for
comparison, and  m - rank(B_A)  as a certified LOWER BOUND on flats_2 (since supp(y) is a
subset of A(x), so rank(B_{supp y}) <= rank(B_A)).
"""

import sys
import numpy as np
from scipy.optimize import linprog, nnls

TOL_ACT = 1e-7
TOL_DUAL = 1e-9


# ------------------------------------------------------------------ MPS parse

def parse_mps(path):
    rows, row_type, cols, rhs, ranges, bounds = [], {}, {}, {}, {}, []
    obj_row, section = None, None
    for raw in open(path):
        if not raw.strip() or raw.lstrip().startswith('*'):
            continue
        if not raw[0].isspace():                       # section header
            section = raw.split()[0].upper()
            continue
        f = raw.split()
        if section == 'ROWS':
            typ, name = f[0].upper(), f[1]
            row_type[name] = typ
            if typ == 'N':
                if obj_row is None:
                    obj_row = name
            else:
                rows.append(name)
        elif section == 'COLUMNS':
            if len(f) >= 3 and f[2] == "'MARKER'":
                continue
            col = f[0]
            cols.setdefault(col, {})
            for i in range(1, len(f) - 1, 2):
                cols[col][f[i]] = cols[col].get(f[i], 0.0) + float(f[i + 1])
        elif section == 'RHS':
            for i in range(1, len(f) - 1, 2):
                rhs[f[i]] = rhs.get(f[i], 0.0) + float(f[i + 1])
        elif section == 'RANGES':
            for i in range(1, len(f) - 1, 2):
                ranges[f[i]] = float(f[i + 1])
        elif section == 'BOUNDS':
            bounds.append((f[0].upper(), f[2], float(f[3]) if len(f) > 3 else None))
    return rows, row_type, cols, rhs, ranges, bounds, obj_row


def to_Bxgeb(path):
    rows, row_type, cols, rhs, ranges, bounds, obj_row = parse_mps(path)
    colnames = sorted(cols)
    ci = {c: j for j, c in enumerate(colnames)}
    ri = {r: i for i, r in enumerate(rows)}
    m = len(colnames)

    A = np.zeros((len(rows), m))
    d = np.zeros(m)
    for c, ents in cols.items():
        for r, v in ents.items():
            if r == obj_row:
                d[ci[c]] += v
            elif r in ri:
                A[ri[r], ci[c]] += v

    lo = np.zeros(m); hi = np.full(m, np.inf)
    for typ, col, val in bounds:
        j = ci[col]
        if typ == 'UP':
            hi[j] = val
            if val < 0 and lo[j] == 0:
                lo[j] = -np.inf
        elif typ == 'LO': lo[j] = val
        elif typ == 'FX': lo[j] = hi[j] = val
        elif typ == 'FR': lo[j], hi[j] = -np.inf, np.inf
        elif typ == 'MI': lo[j] = -np.inf
        elif typ == 'PL': hi[j] = np.inf
        elif typ == 'BV': lo[j], hi[j] = 0.0, 1.0

    Brows, brhs = [], []
    for r in rows:
        a, t, r0 = A[ri[r]], row_type[r], rhs.get(r, 0.0)
        rg = ranges.get(r)
        if t == 'G':
            Brows.append(a); brhs.append(r0)
            if rg is not None: Brows.append(-a); brhs.append(-(r0 + abs(rg)))
        elif t == 'L':
            Brows.append(-a); brhs.append(-r0)
            if rg is not None: Brows.append(a); brhs.append(r0 - abs(rg))
        elif t == 'E':
            if rg is None:
                Brows.append(a); brhs.append(r0); Brows.append(-a); brhs.append(-r0)
            else:
                up = r0 + abs(rg) if rg > 0 else r0
                dn = r0 if rg > 0 else r0 - abs(rg)
                Brows.append(a); brhs.append(dn); Brows.append(-a); brhs.append(-up)
    for j in range(m):
        if np.isfinite(lo[j]):
            e = np.zeros(m); e[j] = 1.0; Brows.append(e); brhs.append(lo[j])
        if np.isfinite(hi[j]):
            e = np.zeros(m); e[j] = -1.0; Brows.append(e); brhs.append(-hi[j])
    return np.array(Brows), np.array(brhs), d, colnames


# ------------------------------------------------------------- certificates

def certificates(name, B, b, d):
    n, m = B.shape
    res = linprog(d, A_ub=-B, b_ub=-b, bounds=[(None, None)] * m, method='highs')
    if not res.success:
        print(f"{name}: LP FAILED -- {res.message}"); return
    x = res.x
    slack = B @ x - b
    scale = np.maximum(1.0, np.abs(b))
    A_idx = np.where(slack <= TOL_ACT * scale)[0]

    rank_A = int(np.linalg.matrix_rank(B[A_idx], tol=1e-8)) if len(A_idx) else 0
    flats1 = len(A_idx) - rank_A

    # HiGHS dual (a basic dual solution), then least-norm dual restricted to A(x)
    yh = np.zeros(n)
    if res.ineqlin is not None:
        yh = np.maximum(-res.ineqlin.marginals, 0.0)
    supp_h = np.where(yh > TOL_DUAL * max(1.0, yh.max() if yh.size else 1.0))[0]

    BA = B[A_idx]
    rho = 1e-8
    M = np.vstack([BA.T, np.sqrt(rho) * np.eye(len(A_idx))])
    rhsv = np.concatenate([d, np.zeros(len(A_idx))])
    try:
        yln, _ = nnls(M, rhsv, maxiter=20 * M.shape[1])
        eqres = float(np.linalg.norm(BA.T @ yln - d) / max(1.0, np.linalg.norm(d)))
    except Exception as e:
        yln, eqres = np.zeros(len(A_idx)), float('nan')
        print(f"  [nnls failed: {e}]")
    supp_ln = A_idx[yln > TOL_DUAL * max(1.0, yln.max() if yln.size else 1.0)]

    rank_ln = int(np.linalg.matrix_rank(B[supp_ln], tol=1e-8)) if len(supp_ln) else 0
    rank_h = int(np.linalg.matrix_rank(B[supp_h], tol=1e-8)) if len(supp_h) else 0
    flats2_ln = m - rank_ln
    flats2_lb = m - rank_A                     # certified lower bound: supp(y) subset of A

    print(f"\n=== {name} ===  n={n} constraints, m={m} variables, alpha=n/m={n/m:.2f}, "
          f"obj={res.fun:.9g}")
    print(f"  |A(x)| = {len(A_idx):<6d}  rank(B_A) = {rank_A:<6d}   a = |A|/m = {len(A_idx)/m:.2f}")
    print(f"  ITEM 1  flats_1 = |A| - rank(B_A)        = {flats1:<6d}"
          f"  -> {'TRANSVERSE (LICQ holds)' if flats1 == 0 else 'STALLS'}")
    print(f"  ITEM 2  flats_2 = m - rank(B_supp(y_ln)) = {flats2_ln:<6d}"
          f"  -> {'TRANSVERSE (Mangasarian (18) holds)' if flats2_ln == 0 else 'STALLS'}")
    print(f"          |supp y_leastnorm| = {len(supp_ln)}, rank = {rank_ln}, "
          f"eq-residual = {eqres:.1e}")
    print(f"          |supp y_HiGHS|     = {len(supp_h)}, rank = {rank_h}, "
          f"flats_2(HiGHS) = {m - rank_h}")
    print(f"          certified lower bound m - rank(B_A) = {flats2_lb}")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["afiro", "degen2", "fit1d"]):
        B, b, d, _ = to_Bxgeb(f"../netlib/{nm}.mps")
        certificates(nm, B, b, d)
