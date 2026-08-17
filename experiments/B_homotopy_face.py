"""Fixed-t, sparsity-preserving structural homotopy for a Mangasarian face."""

from __future__ import annotations

import signal

import numpy as np

from active_set_projection import multiplier_kkt
from exp23_path_primal_dual import piece_y
from primal_face_pricing import priced_settle_escape


class StageTimeout(Exception):
    pass


def _timeout(_signum, _frame):
    raise StageTimeout("B-homotopy stage ceiling reached")


def perturb_nonzeros(B, noise, delta):
    """Multiplicatively perturb only the stored nonzeros of a dense matrix."""
    B = np.asarray(B, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if noise.shape != B.shape:
        raise ValueError("noise must have the same shape as B")
    perturbed = B.copy()
    nonzero = B != 0.0
    perturbed[nonzero] *= 1.0 + float(delta) * noise[nonzero]
    return perturbed


def certify_face(B, b, d, W, t, coefficients, tol=1e-7):
    """Independently check fixed-t feasibility and multiplier KKT."""
    W = np.asarray(W, dtype=bool)
    g, h, _ua, _uc = coefficients[:4]
    y = np.zeros(B.shape[0])
    y[W] = t * g + h
    residual = B.T @ y - d
    dres = float(np.linalg.norm(residual)) / max(
        1.0, float(np.linalg.norm(d)))
    scale = 1.0 + np.abs(d) + np.abs(B).T @ np.abs(y)
    equality_error = float(np.max(np.abs(residual) / scale))
    nonnegative_error = max(
        0.0, float(np.max(-y / (1.0 + np.abs(y)))))
    kkt = multiplier_kkt(B, b, y, W, t, tol=tol)
    passed = (float(dres) <= tol and equality_error <= tol
              and nonnegative_error <= tol and bool(kkt.get("passed")))
    return passed, {
        "passed": bool(passed),
        "piece_dual_residual": float(dres),
        "equality_error": equality_error,
        "nonnegative_error": nonnegative_error,
        "multiplier_kkt": kkt,
    }


def follow_delta_ladder(B, b, d, W, t, deltas=(1e-6, 1e-8, 0.0),
                        seed=7601, candidates=12, rounds=30,
                        tol=1e-7, seconds_per_stage=8):
    """Track one face down delta to the exact original matrix; fail closed."""
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    W = np.asarray(W, dtype=bool).copy()
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1.0, 1.0, size=B.shape)
    noise[B == 0.0] = 0.0
    initial_W = W.copy()
    records = []
    old_handler = signal.signal(signal.SIGALRM, _timeout)
    try:
        for delta in deltas:
            Bd = perturb_nonzeros(B, noise, delta)
            sparsity_preserved = bool(np.array_equal(Bd != 0.0, B != 0.0))
            diagnostics = {}
            stats = {}
            signal.alarm(int(seconds_per_stage))
            try:
                settled_W, coefficients, settled = priced_settle_escape(
                    Bd, b, d, W, t,
                    piece_solver=lambda mask, matrix=Bd: piece_y(
                        matrix, b, d, mask),
                    max_candidates=candidates, rounds=rounds, tol=tol,
                    diagnostics=diagnostics, stats=stats)
            except StageTimeout as error:
                records.append({
                    "delta": float(delta), "status": "RESOURCE-LIMIT",
                    "detail": str(error),
                    "sparsity_preserved": sparsity_preserved,
                })
                return {
                    "status": "RESOURCE-LIMIT", "records": records,
                    "returned_to_original": False,
                    "working_set_changed": bool(np.any(W != initial_W)),
                }
            finally:
                signal.alarm(0)
            record = {
                "delta": float(delta),
                "sparsity_preserved": sparsity_preserved,
                "settled": bool(settled),
                "diagnostics": diagnostics,
                "stats": stats,
                "support_before": int(W.sum()),
                "support_after": int(settled_W.sum()),
                "working_set_changes": int(np.sum(settled_W != W)),
            }
            if not sparsity_preserved or not settled or coefficients is None:
                record["status"] = "FAILED"
                records.append(record)
                return {
                    "status": "FAILED", "records": records,
                    "returned_to_original": False,
                    "working_set_changed": bool(np.any(settled_W != initial_W)),
                }
            certified, certificate = certify_face(
                Bd, b, d, settled_W, t, coefficients, tol=tol)
            record["certificate"] = certificate
            record["status"] = "CERTIFIED" if certified else "FAILED"
            records.append(record)
            W = settled_W
            if not certified:
                return {
                    "status": "FAILED", "records": records,
                    "returned_to_original": False,
                    "working_set_changed": bool(np.any(W != initial_W)),
                }
        return {
            "status": "CERTIFIED",
            "records": records,
            "returned_to_original": bool(float(deltas[-1]) == 0.0),
            "working_set_changed": bool(np.any(W != initial_W)),
            "total_working_set_changes": int(np.sum(W != initial_W)),
            "final_support": int(W.sum()),
            "t": float(t),
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
