#!/usr/bin/env python3
"""How much accuracy does a judge prompt leave unused? Probe on the final hidden state vs the judge's own score.

usage:
  probe_gap.py --tune-states T.npy --tune-rows T.csv --test-states S.npy [S2.npy ...] --test-rows S.csv [S2.csv ...]
               [--c-grid 1e-4,3e-4,...] [--cv-folds 5] [--seed 0] [--nboot 2000] [--judge-threshold 0]
               [--learning-curve [--sizes 50,100,200,400,800] [--draws 10] [--lc-folds 3] [--lc-c-grid ...]]
  probe_gap.py --selftest

Inputs
  states  .npy, float, one row per item: the model's final hidden state at the answer position
          (score_exact.py --states writes these).
  rows    CSV with a header, or JSONL, with fields key, label (1 = positive, e.g. attack; 0 = negative),
          score (the judge's own score, higher = more positive; e.g. the margin from score_exact.py).
          Optional field stratum: if present, the cross-validation folds are stratified on
          stratum x label instead of label alone.
  If PREFIX.keys.txt sits beside PREFIX.npy (as score_exact.py writes it), rows are matched to state rows
  by key; otherwise the rows file must list the items in the same order as the states.
  Several test pairs are concatenated in the order given.

What it does
  1. Fits a probe: L2 logistic regression with balanced class weights on standardised tuning states
     (scaler fitted on the tuning set). C is chosen from --c-grid by stratified k-fold cross-validation
     on the tuning set (mean fold AUC; the first C wins a tie), then the probe is refitted on all tuning items.
  2. Scores the test set with it and prints
     - judge AUC and probe AUC on the test set (negative vs positive items);
     - the lead = probe AUC - judge AUC, with a 95 % interval from a paired bootstrap over test items
       (negatives and positives resampled separately, the same resample for both scores; the probe is
       not refitted, so the interval covers test-set sampling only);
     - positives caught at the judge's own false-alarm count: the judge marks score > --judge-threshold;
       k = test negatives it marks; the probe's threshold is set on the test set so that it marks the
       same k negatives; the recall of both is printed.
     A large lead means the information is in the model's state and the prompt does not get it into the
     answer: prompt work can pay. A lead near your wording noise means the prompt already uses about
     what a linear readout can get.
  3. --learning-curve: for each n in --sizes, draws --draws label-stratified random samples of n tuning
     items (at least 6 per class), chooses C by --lc-folds-fold CV inside the sample, fits, scores the
     full test set and prints mean, sd, min and max of the lead. One random generator (--seed) serves
     all sizes in the order given, so a size's draws depend on the sizes before it.

Prints numbers only. Needs numpy and scikit-learn.
"""
import argparse, csv, json, os, sys, warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DEFAULT_GRID = "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,1"


# ---------------------------------------------------------------- loading
def read_rows(path):
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
    if head == "{":
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    else:
        rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    out = []
    for i, r in enumerate(rows):
        for f in ("key", "label", "score"):
            if r.get(f) in (None, ""):
                sys.exit(f"FATAL: {path} row {i + 1} lacks {f}")
        lab = int(float(r["label"]))
        if lab not in (0, 1):
            sys.exit(f"FATAL: {path} row {i + 1} label must be 0 or 1")
        out.append({"key": str(r["key"]), "label": lab, "score": float(r["score"]),
                    "stratum": None if r.get("stratum") in (None, "") else str(r["stratum"])})
    return out


def load_set(states_path, rows_path):
    X = np.load(states_path, mmap_mode="r")
    rows = read_rows(rows_path)
    keys_path = states_path[:-4] + ".keys.txt" if states_path.endswith(".npy") else None
    if keys_path and os.path.exists(keys_path):
        keys = [l.rstrip("\n") for l in open(keys_path, encoding="utf-8") if l.strip()]
        byk = {r["key"]: r for r in rows}
        if len(byk) != len(rows):
            sys.exit(f"FATAL: duplicate keys in {rows_path}")
        if len(keys) != X.shape[0] or set(keys) != set(byk):
            sys.exit(f"FATAL: {keys_path} and {rows_path} do not hold the same keys as the {X.shape[0]} state rows")
        rows = [byk[k] for k in keys]
    elif len(rows) != X.shape[0]:
        sys.exit(f"FATAL: {rows_path} has {len(rows)} rows, {states_path} has {X.shape[0]}")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        sys.exit(f"FATAL: {states_path} must be a finite 2-d array (unfinished run?)")
    y = np.array([r["label"] for r in rows])
    s = np.array([r["score"] for r in rows], float)
    st = [r["stratum"] for r in rows]
    st = None if all(v is None for v in st) else np.array([str(v) for v in st])
    return X, y, s, st


# ---------------------------------------------------------------- probe
def auc(y, s):
    return float(roc_auc_score(y, s))


def fit_lr(X, y, C):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, tol=1e-6)
    lr.fit(sc.transform(X), y)
    return sc, lr


def score_lr(m, X):
    sc, lr = m
    return lr.decision_function(sc.transform(X))


def choose_C(X, y, strata, grid, folds, seed):
    split = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, strata))
    res = {C: float(np.mean([auc(y[te], score_lr(fit_lr(X[tr], y[tr], C), X[te])) for tr, te in split])) for C in grid}
    return max(grid, key=lambda c: res[c]), res


def matched_thr(s, neg, k):
    """Threshold t such that exactly k negatives have s > t (if no ties at the boundary)."""
    sb = np.sort(s[neg])[::-1]
    if k <= 0:
        return sb[0] + 1e-9
    if k >= len(sb):
        return sb[-1] - 1e-9
    return 0.5 * (sb[k - 1] + sb[k])


def boot_auc(scores, i0, i1, nboot, rng):
    """Paired bootstrap of AUC: negatives i0 and positives i1 resampled separately, same weights for every score."""
    w1 = rng.multinomial(len(i1), np.ones(len(i1)) / len(i1), size=nboot).astype(float)
    w0 = rng.multinomial(len(i0), np.ones(len(i0)) / len(i0), size=nboot).astype(float)
    out = {}
    for r, s in scores.items():
        a, b = s[i1][:, None], s[i0][None, :]
        M = (a > b) + 0.5 * (a == b)
        out[r] = np.einsum("bi,bi->b", w1 @ M, w0) / (len(i1) * len(i0))
    return out


def analyse(Xt, yt, st_t, strata_t, Xs, ys, ss, grid, folds, seed, nboot, judge_thr):
    Xt, Xs = np.asarray(Xt, np.float64), np.asarray(Xs, np.float64)
    strata = np.array([f"{a}|{b}" for a, b in zip(strata_t, yt)]) if strata_t is not None else yt
    C, cv = choose_C(Xt, yt, strata, grid, folds, seed)
    model = fit_lr(Xt, yt, C)
    ps = score_lr(model, Xs)
    out = {"tune": {"n": int(len(yt)), "n_pos": int(yt.sum()), "judge_auc": auc(yt, st_t), "C": C, "cv_auc": cv},
           "test": {"n": int(len(ys)), "n_pos": int(ys.sum())}}
    ja, pa = auc(ys, ss), auc(ys, ps)
    i0, i1 = np.flatnonzero(ys == 0), np.flatnonzero(ys == 1)
    B = boot_auc({"judge": ss, "probe": ps}, i0, i1, nboot, np.random.default_rng(seed))
    d = B["probe"] - B["judge"]
    out["test"].update({"judge_auc": ja, "probe_auc": pa, "lead": pa - ja,
                        "lead_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]})
    neg = ys == 0
    jf = ss > judge_thr
    k = int(jf[neg].sum())
    pthr = matched_thr(ps, neg, k)
    pf = ps > pthr
    out["test"]["matched"] = {"judge_threshold": judge_thr, "false_alarms_judge": k,
                              "false_alarms_probe": int(pf[neg].sum()), "probe_threshold": float(pthr),
                              "caught_judge": int(jf[~neg].sum()), "caught_probe": int(pf[~neg].sum()),
                              "recall_judge": float(jf[~neg].mean()), "recall_probe": float(pf[~neg].mean())}
    return out, ps


def strat_sample(y, n, rng):
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    npos = max(6, int(round(n * len(pos) / len(y)))); nneg = max(6, n - npos)
    return np.r_[rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)]


def learning_curve(Xt, yt, Xs, ys, judge_auc, sizes, draws, folds, grid, seed):
    Xt, Xs = np.asarray(Xt, np.float64), np.asarray(Xs, np.float64)
    rng = np.random.default_rng(seed)
    res = {}
    for n in sizes:
        if n >= len(yt):
            res[n] = None
            continue
        g = []
        for r in range(draws):
            idx = strat_sample(yt, n, rng)
            C, _ = choose_C(Xt[idx], yt[idx], yt[idx], grid, folds, r)
            g.append(auc(ys, score_lr(fit_lr(Xt[idx], yt[idx], C), Xs)) - judge_auc)
        g = np.array(g)
        res[n] = {"mean": float(g.mean()), "sd": float(g.std(ddof=1)) if draws > 1 else 0.0,
                  "min": float(g.min()), "max": float(g.max()), "draws": draws}
    return res


# ---------------------------------------------------------------- printing
def report(out):
    t, s, m = out["tune"], out["test"], out["test"]["matched"]
    print(f"tuning: n {t['n']} (positives {t['n_pos']}), judge AUC {t['judge_auc']:.4f}")
    print("  C by CV: " + ", ".join(f"{c:g}:{v:.4f}" for c, v in t["cv_auc"].items()) + f"  -> C = {t['C']:g}")
    print(f"test:   n {s['n']} (positives {s['n_pos']}, negatives {s['n'] - s['n_pos']})")
    print(f"  judge AUC {s['judge_auc']:.4f}")
    print(f"  probe AUC {s['probe_auc']:.4f}")
    print(f"  lead (probe - judge) {s['lead']:+.4f}  95% CI [{s['lead_ci95'][0]:+.4f}, {s['lead_ci95'][1]:+.4f}]")
    npos = s["n_pos"]
    print(f"  at the judge's false-alarm count (judge score > {m['judge_threshold']:g}: "
          f"{m['false_alarms_judge']} of {s['n'] - npos} negatives marked):")
    print(f"    judge catches {m['caught_judge']} of {npos} positives (recall {m['recall_judge']:.3f})")
    print(f"    probe catches {m['caught_probe']} of {npos} positives (recall {m['recall_probe']:.3f}), "
          f"marking {m['false_alarms_probe']} negatives")


def report_lc(lc):
    print("learning curve: lead = probe AUC (fitted on n tuning items) - judge AUC, on the full test set")
    print("  n      mean     sd       min      max")
    for n, v in lc.items():
        if v is None:
            print(f"  {n:<6} skipped (not below the tuning-set size)")
        else:
            print(f"  {n:<6} {v['mean']:+.4f}  {v['sd']:.4f}   {v['min']:+.4f}  {v['max']:+.4f}")


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile, shutil
    fails = []

    def check(cond, what):
        print(("ok   " if cond else "FAIL ") + what, flush=True)
        if not cond:
            fails.append(what)

    rng = np.random.default_rng(1)

    def make(n, d=150, judge_noise=2.0):
        """States carry the label through a 3-d latent; the judge sees one latent axis plus noise."""
        y = (rng.random(n) < 0.4).astype(int)
        z = rng.normal(0, 1, (n, 3)) + 1.2 * y[:, None]
        A = np.random.default_rng(7).normal(0, 1, (3, d))
        X = z @ A + rng.normal(0, 3.0, (n, d))
        judge = z[:, 0] + rng.normal(0, judge_noise, n) - 0.6
        oracle = z.sum(1)                      # what a perfect linear readout could reach
        return X.astype(np.float32), y, judge, oracle

    grid = [float(c) for c in DEFAULT_GRID.split(",")]
    Xt, yt, jt, ot = make(600)
    Xs, ys, js, os_ = make(800)

    # tiny AUC and threshold checks
    check(abs(auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) - 0.75) < 1e-12, "AUC of a hand example = 0.75")
    s = rng.normal(0, 1, 50); neg = np.r_[np.ones(30, bool), np.zeros(20, bool)]
    check(all(int((s[neg] > matched_thr(s, neg, k)).sum()) == k for k in range(0, 31)),
          "matched threshold marks exactly k negatives, k = 0..30")

    out, ps = analyse(Xt, yt, jt, None, Xs, ys, js, grid, 5, 0, 2000, 0.0)
    t = out["test"]
    check(t["probe_auc"] > t["judge_auc"] + 0.05 and t["lead_ci95"][0] > 0,
          f"weak judge: probe AUC {t['probe_auc']:.3f} > judge {t['judge_auc']:.3f}, CI excludes 0")
    check(t["lead_ci95"][0] <= t["lead"] <= t["lead_ci95"][1], "bootstrap interval contains the point lead")
    m = t["matched"]
    check(m["false_alarms_probe"] == m["false_alarms_judge"] == int((js[ys == 0] > 0).sum()),
          "probe matched to the judge's false-alarm count")
    check(m["recall_probe"] >= m["recall_judge"], "at equal false alarms the stronger probe catches at least as many")

    out2, _ = analyse(Xt, yt, ot, None, Xs, ys, os_, grid, 5, 0, 2000, 0.0)
    check(abs(out2["test"]["lead"]) < 0.03 and out2["test"]["lead_ci95"][0] < 0.01,
          f"judge that already uses the signal: lead {out2['test']['lead']:+.3f}, no significant lead")

    # file formats and key alignment
    tmp = tempfile.mkdtemp(prefix="probe_gap_selftest_")
    try:
        np.save(os.path.join(tmp, "tune.npy"), Xt)
        np.save(os.path.join(tmp, "test.npy"), Xs)
        keys_t = [f"t{i}" for i in range(len(yt))]
        with open(os.path.join(tmp, "tune.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["key", "label", "score"])
            for k, l, v in zip(keys_t, yt, jt):
                w.writerow([k, l, repr(float(v))])
        perm = rng.permutation(len(ys))
        keys_s = [f"s{i}" for i in range(len(ys))]
        with open(os.path.join(tmp, "test.jsonl"), "w") as f:
            for i in perm:                      # rows shuffled; keys file restores state order
                f.write(json.dumps({"key": keys_s[i], "label": int(ys[i]), "score": float(js[i])}) + "\n")
        with open(os.path.join(tmp, "test.keys.txt"), "w") as f:
            f.write("".join(k + "\n" for k in keys_s))
        A = load_set(os.path.join(tmp, "tune.npy"), os.path.join(tmp, "tune.csv"))
        B = load_set(os.path.join(tmp, "test.npy"), os.path.join(tmp, "test.jsonl"))
        check(np.array_equal(A[1], yt) and np.array_equal(A[2], jt), "CSV rows read in state order")
        check(np.array_equal(B[1], ys) and np.array_equal(B[2], js), "shuffled JSONL rows realigned by the keys file")
        out3, _ = analyse(A[0], A[1], A[2], A[3], B[0], B[1], B[2], grid, 5, 0, 2000, 0.0)
        check(out3["test"]["lead"] == t["lead"] and out3["test"]["lead_ci95"] == t["lead_ci95"],
              "results from files equal results from arrays")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    lc = learning_curve(Xt, yt, Xs, ys, t["judge_auc"], [30, 100, 400, 600], 5, 3, grid, 0)
    check(lc[600] is None and all(lc[n] is not None for n in (30, 100, 400)), "learning curve skips n >= tuning size")
    check(lc[400]["mean"] >= lc[30]["mean"] - 0.01 and lc[400]["mean"] <= t["lead"] + 0.02,
          f"learning curve: lead at n=400 {lc[400]['mean']:+.3f} >= n=30 {lc[30]['mean']:+.3f}, not above full fit")
    report(out)
    report_lc(lc)
    print(f"selftest: {'PASS' if not fails else 'FAIL'} ({len(fails)} failed)")
    return 1 if fails else 0


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tune-states")
    ap.add_argument("--tune-rows")
    ap.add_argument("--test-states", nargs="+")
    ap.add_argument("--test-rows", nargs="+")
    ap.add_argument("--c-grid", default=DEFAULT_GRID, help="comma-separated C values for the probe")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="CV split and bootstrap seed")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--judge-threshold", type=float, default=0.0, help="judge marks an item when score > this")
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--sizes", default="50,100,200,400,800")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--lc-folds", type=int, default=3)
    ap.add_argument("--lc-c-grid", help="C grid for the learning curve (default: --c-grid)")
    ap.add_argument("--json", help="also write all numbers to this JSON file")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not (a.tune_states and a.tune_rows and a.test_states and a.test_rows):
        ap.error("--tune-states, --tune-rows, --test-states and --test-rows are required")
    if len(a.test_states) != len(a.test_rows):
        ap.error("give one --test-rows file per --test-states file")
    grid = [float(c) for c in a.c_grid.split(",")]
    Xt, yt, st_t, strata_t = load_set(a.tune_states, a.tune_rows)
    parts = [load_set(p, r) for p, r in zip(a.test_states, a.test_rows)]
    Xs = np.vstack([p[0] for p in parts]); ys = np.concatenate([p[1] for p in parts])
    ss = np.concatenate([p[2] for p in parts])
    if Xs.shape[1] != Xt.shape[1]:
        sys.exit(f"FATAL: tuning states have {Xt.shape[1]} columns, test states {Xs.shape[1]}")
    out, _ = analyse(Xt, yt, st_t, strata_t, Xs, ys, ss, grid, a.cv_folds, a.seed, a.nboot, a.judge_threshold)
    report(out)
    if a.learning_curve:
        sizes = [int(x) for x in a.sizes.split(",")]
        lgrid = [float(c) for c in a.lc_c_grid.split(",")] if a.lc_c_grid else grid
        out["learning_curve"] = learning_curve(Xt, yt, Xs, ys, out["test"]["judge_auc"], sizes, a.draws,
                                               a.lc_folds, lgrid, a.seed)
        report_lc(out["learning_curve"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
