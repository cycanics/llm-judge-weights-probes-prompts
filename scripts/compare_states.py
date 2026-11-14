#!/usr/bin/env python3
"""Are two state files for the same items bit-identical? If not, how many rows differ and by how much.

usage:
  compare_states.py A.npy B.npy
  compare_states.py --selftest

Typical use: final states of the same items from two model files (for example an edited and an unedited
model, where the edit touches only the output matrix and the states should therefore not move).
If A.keys.txt and B.keys.txt sit beside the files (as score_exact.py writes them), B's rows are matched
to A's by key; otherwise rows are compared in file order.

Prints: shapes and dtypes; whether the arrays are bit-identical (same bytes, NaN patterns included);
otherwise the number of rows with any differing bit, the number of differing values, and the largest
absolute difference (over values finite in both). Exit status 0 identical, 1 different, 2 not comparable.
"""
import argparse, os, sys

import numpy as np


def keys_for(path):
    p = path[:-4] + ".keys.txt" if path.endswith(".npy") else None
    if p and os.path.exists(p):
        return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]
    return None


def compare(A, B):
    """Returns a dict; A and B are 2-d arrays of the same shape."""
    if A.dtype != B.dtype:
        A, B = A.astype(np.float64), B.astype(np.float64)
    bitsA = np.ascontiguousarray(A).view(np.uint8).reshape(A.shape[0], -1)
    bitsB = np.ascontiguousarray(B).view(np.uint8).reshape(B.shape[0], -1)
    row_diff = (bitsA != bitsB).any(axis=1)
    a64, b64 = A.astype(np.float64), B.astype(np.float64)
    both = np.isfinite(a64) & np.isfinite(b64)
    nan_mismatch = int((np.isnan(a64) != np.isnan(b64)).sum())
    diff = np.abs(a64 - b64)
    vals = (a64 != b64) & both
    return {"identical": not row_diff.any(), "rows": int(A.shape[0]), "rows_differing": int(row_diff.sum()),
            "values_differing": int(vals.sum()), "nan_pattern_mismatch": nan_mismatch,
            "max_abs_diff": float(diff[both].max()) if both.any() else float("nan"),
            "rows_with_nan_a": int(np.isnan(a64).any(axis=1).sum()), "rows_with_nan_b": int(np.isnan(b64).any(axis=1).sum())}


def run(pa, pb):
    A = np.load(pa, mmap_mode="r"); B = np.load(pb, mmap_mode="r")
    print(f"A {pa}: shape {A.shape} {A.dtype}")
    print(f"B {pb}: shape {B.shape} {B.dtype}")
    A = A.reshape(A.shape[0], -1); B = B.reshape(B.shape[0], -1)   # one row per item
    ka, kb = keys_for(pa), keys_for(pb)
    if ka is not None and kb is not None:
        if sorted(ka) != sorted(kb) or len(ka) != A.shape[0] or len(kb) != B.shape[0]:
            print("not comparable: the keys files do not hold the same items")
            return 2
        pos = {k: i for i, k in enumerate(kb)}
        B = B[[pos[k] for k in ka]]
        print("rows matched by key" + ("" if ka == kb else " (B reordered)"))
    else:
        print("rows compared in file order (no keys files)")
    if A.shape != B.shape:
        print("not comparable: shapes differ")
        return 2
    r = compare(np.asarray(A), np.asarray(B))
    if r["identical"]:
        print(f"bit-identical: yes, {r['rows']}/{r['rows']} rows")
        return 0
    print(f"bit-identical: no; rows differing {r['rows_differing']}/{r['rows']}, values differing {r['values_differing']}, "
          f"max |A-B| {r['max_abs_diff']:.6g}, NaN-pattern mismatches {r['nan_pattern_mismatch']}")
    return 1


def selftest():
    import tempfile, shutil
    fails = []

    def check(cond, what):
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            fails.append(what)

    rng = np.random.default_rng(0)
    A = rng.normal(0, 50, (300, 64)).astype(np.float32)
    r = compare(A, A.copy())
    check(r["identical"] and r["rows_differing"] == 0, "copy is bit-identical")
    B = A.copy(); B[[3, 17, 250], [0, 5, 63]] += np.float32(0.25); B[100, 10] = np.nextafter(B[100, 10], np.float32(np.inf))
    r = compare(A, B)
    check(not r["identical"] and r["rows_differing"] == 4 and r["values_differing"] == 4, "4 changed rows found, incl. a 1-ulp change")
    check(abs(r["max_abs_diff"] - 0.25) < 1e-4, f"largest difference 0.25 (got {r['max_abs_diff']:.6g})")
    C = A.copy(); C[7, 1] = np.nan
    r = compare(A, C)
    check(not r["identical"] and r["nan_pattern_mismatch"] == 1 and r["rows_with_nan_b"] == 1, "NaN in one file detected")
    Z = A.copy(); Z[9, 9] = 0.0; Z2 = Z.copy(); Z2[9, 9] = -0.0
    r = compare(Z, Z2)
    check(not r["identical"] and r["max_abs_diff"] == 0.0, "+0 vs -0: not bit-identical, max difference 0")
    tmp = tempfile.mkdtemp(prefix="compare_states_selftest_")
    try:
        keys = [f"k{i}" for i in range(300)]
        perm = rng.permutation(300)
        np.save(os.path.join(tmp, "a.npy"), A)
        np.save(os.path.join(tmp, "b.npy"), A[perm])
        open(os.path.join(tmp, "a.keys.txt"), "w").write("".join(k + "\n" for k in keys))
        open(os.path.join(tmp, "b.keys.txt"), "w").write("".join(keys[i] + "\n" for i in perm))
        check(run(os.path.join(tmp, "a.npy"), os.path.join(tmp, "b.npy")) == 0, "same states in another row order, matched by key: identical")
        os.remove(os.path.join(tmp, "b.keys.txt"))
        check(run(os.path.join(tmp, "a.npy"), os.path.join(tmp, "b.npy")) == 1, "same states shuffled, no keys: different")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"selftest: {'PASS' if not fails else 'FAIL'} ({len(fails)} failed)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    x = ap.parse_args()
    if x.selftest:
        sys.exit(selftest())
    if not (x.a and x.b):
        ap.error("give two .npy files")
    sys.exit(run(x.a, x.b))


if __name__ == "__main__":
    main()
