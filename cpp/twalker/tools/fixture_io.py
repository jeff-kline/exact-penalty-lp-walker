"""Portable binary fixtures for the correctness-first t-walker rewrite.

The format is intentionally simple enough to read from C++ without a third
party serialization dependency.  All integers are little-endian and all
floating-point values are IEEE-754 float64.

Header and model blocks::

    char[4]  "TWFX"
    uint32   version (1)
    uint32   n, m, nnz, face_count
    uint64   csr_indptr[n + 1]
    uint32   csr_indices[nnz]
    float64  csr_values[nnz], b[n], d[m]
    uint8    post_seed_support[n]
    float64  t0

Each face then stores its sparse row list and the SVD-oracle result::

    float64  t
    uint32   row_count
    uint32   rows[row_count]
    float64  g[row_count], h[row_count], ua[m], uc[m], dres
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import scipy.sparse as sp


MAGIC = b"TWFX"
VERSION = 1


def _bytes(array, dtype):
    return np.ascontiguousarray(array, dtype=dtype).tobytes()


def write_fixture(path, B, b, d, post_seed_support, t0, faces, metadata):
    """Write one model and its face-oracle records; return manifest fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csr = sp.csr_matrix(np.asarray(B, dtype=float))
    csr.sort_indices()
    n, m = csr.shape
    support = np.asarray(post_seed_support, dtype=bool)
    if support.shape != (n,):
        raise ValueError("post-seed support has the wrong shape")

    with path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<5I", VERSION, n, m, csr.nnz, len(faces)))
        stream.write(_bytes(csr.indptr, "<u8"))
        stream.write(_bytes(csr.indices, "<u4"))
        stream.write(_bytes(csr.data, "<f8"))
        stream.write(_bytes(b, "<f8"))
        stream.write(_bytes(d, "<f8"))
        stream.write(_bytes(support, "u1"))
        stream.write(struct.pack("<d", float(t0)))
        for face in faces:
            rows = np.asarray(face["rows"], dtype=np.uint32)
            g, h, ua, uc, dres = face["truth"]
            if len(g) != len(rows) or len(h) != len(rows):
                raise ValueError("face vector length mismatch")
            if len(ua) != m or len(uc) != m:
                raise ValueError("multiplier length mismatch")
            stream.write(struct.pack("<dI", float(face["t"]), len(rows)))
            stream.write(_bytes(rows, "<u4"))
            stream.write(_bytes(g, "<f8"))
            stream.write(_bytes(h, "<f8"))
            stream.write(_bytes(ua, "<f8"))
            stream.write(_bytes(uc, "<f8"))
            stream.write(struct.pack("<d", float(dres)))

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        **metadata,
        "fixture": path.name,
        "format_version": VERSION,
        "n": int(n),
        "m": int(m),
        "nnz": int(csr.nnz),
        "post_seed_support_size": int(support.sum()),
        "face_count": int(len(faces)),
        "sha256": digest,
    }
    path.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _read_array(stream, count, dtype):
    dtype = np.dtype(dtype)
    payload = stream.read(count * dtype.itemsize)
    if len(payload) != count * dtype.itemsize:
        raise ValueError("truncated fixture")
    return np.frombuffer(payload, dtype=dtype).copy()


def read_fixture(path):
    """Read a fixture into SciPy/NumPy objects for measurement scripts."""
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(4) != MAGIC:
            raise ValueError("bad fixture magic")
        version, n, m, nnz, face_count = struct.unpack("<5I", stream.read(20))
        if version != VERSION:
            raise ValueError("unsupported fixture version %d" % version)
        indptr = _read_array(stream, n + 1, "<u8").astype(np.int64)
        indices = _read_array(stream, nnz, "<u4").astype(np.int32)
        values = _read_array(stream, nnz, "<f8")
        b = _read_array(stream, n, "<f8")
        d = _read_array(stream, m, "<f8")
        support = _read_array(stream, n, "u1").astype(bool)
        t0, = struct.unpack("<d", stream.read(8))
        faces = []
        for _ in range(face_count):
            t, row_count = struct.unpack("<dI", stream.read(12))
            rows = _read_array(stream, row_count, "<u4").astype(np.int32)
            g = _read_array(stream, row_count, "<f8")
            h = _read_array(stream, row_count, "<f8")
            ua = _read_array(stream, m, "<f8")
            uc = _read_array(stream, m, "<f8")
            dres, = struct.unpack("<d", stream.read(8))
            faces.append({"t": t, "rows": rows,
                          "truth": (g, h, ua, uc, dres)})
        if stream.read(1):
            raise ValueError("unexpected trailing fixture data")
    B = sp.csr_matrix((values, indices, indptr), shape=(n, m))
    return {"B": B, "b": b, "d": d, "post_seed_support": support,
            "t0": t0, "faces": faces}
