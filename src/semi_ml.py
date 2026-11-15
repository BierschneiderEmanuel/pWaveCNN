"""Semi-supervised training of the candidate scorer with UNLABELLED ECG (after ECGMatch).

The labelled TRAIN split is left as it is. Unlabelled recordings add pseudo-labelled rows:

1. Teacher. The current model (default models/pwave_ml.pkl) scores every candidate peak
   of every unlabelled beat.
2. Neighbour vote. Each candidate's probability is replaced by the mean probability of its
   K nearest unlabelled candidates in the standardised feature space (itself included),
   which smooths out single wrong predictions. With more than --bank-size candidates, the
   neighbours come from a fixed random subset of that size.
3. Agreement weight. A row counts with weight |2q - 1|: 1 when the neighbours agree,
   0 when they split 50/50. Rows below --min-agree are dropped.
4. One P per beat. A candidate becomes a positive only if it has the highest vote in its
   beat and the vote is >= 0.5. All other candidates of the beat become negatives.
5. Strong-view consistency (--aug-copies). The pseudo P positions also label the
   candidates of augmented copies of the unlabelled signal (src/augment.py, "strong"), so
   the student must find the same P wave in a noisier signal.
6. Student. The logistic regression is refit on the labelled + pseudo rows. The unlabelled
   pool weighs lambda_u times one labelled dataset. The student then becomes the next
   teacher (--rounds).

lambda_u, the round and the threshold are chosen on VAL. The result goes to
models/pwave_ml_semi.pkl and models/pwave_ml_semi.json. The test split is never read.

Input (--unlabeled DIR, searched recursively):
  *.json  BUT PDB layout. Uses the "ECG" channel if present, else the first. Labels are ignored.
  *.hea   WFDB record. Channel --lead (default 0).
  *.npy   1-D array, sampling rate --fs.
  *.csv   one value per line, sampling rate --fs.
Every signal is resampled to 128 Hz, the rate of the labelled data. Units should be mV.

Patient leakage: unlabelled data must not come from VAL or TEST patients. Records whose
name matches a VAL/TEST record in splits.json are refused. --groups FILE (JSON
{"file name": "patient group"}) checks against the split groups as well, e.g.
"mitdb:231" for any excerpt of MIT-BIH record 231.

Usage:
  python src/semi_ml.py --unlabeled /path/to/ecg [--fs 250] [--lead 0] [--groups g.json]
                        [--lambdas 0.25 0.5 1.0] [--rounds 2] [--k 10] [--min-agree 0.4]
                        [--aug-copies 1] [--max-hours 0]
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import sys

import numpy as np
from scipy.signal import resample_poly
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                                                       # noqa: E402
from augment import augment                                         # noqa: E402
from bench import DATASETS                                          # noqa: E402
from elgendi import elgendi_rpeaks                                  # noqa: E402
from pwave_ml import MLDetector, extract, label                     # noqa: E402
from train_ml import MODELS, PRM, THRS, evaluate, featurize, stack  # noqa: E402

HERE = splits.HERE
FS = 128
CACHE = os.path.join(HERE, ".cache_unlabeled")


# ---------------------------------------------------------------- input loading
def _read(path, fs, lead):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        d = json.load(open(path))
        out = []
        if not isinstance(d, dict) or "recording" not in d:     # for example dataset_unlabeled/manifest.json
            return out
        for snip in d["recording"]:
            ch = next((c for c in snip["ch"] if c["id"] == "ECG"), snip["ch"][0])
            out.append((np.asarray(ch["y"], float), float(ch.get("sample_rate", fs or FS))))
        return out
    if ext == ".hea":
        import wfdb
        rec = wfdb.rdrecord(path[:-4], channels=[lead])
        return [(rec.p_signal[:, 0].astype(float), float(rec.fs))]
    if ext in (".npy", ".csv"):
        if not fs:
            sys.exit(f"{path}: --fs is required for .npy and .csv input")
        x = np.load(path) if ext == ".npy" else np.loadtxt(path, delimiter=",")
        return [(np.asarray(x, float).ravel(), float(fs))]
    return []


def _to_fs(x, fs):
    if fs == FS:
        return x
    g = np.gcd(int(round(fs)), FS)
    return resample_poly(x, FS // g, int(round(fs)) // g)


def check_leakage(files, groups_file):
    held = {k.split("/", 1)[1]: v for k, v in json.load(open(splits.SPLIT_FILE)).items() if v["split"] != "train"}
    held_groups = {v["group"] for v in held.values()}
    # the source record ids, e.g. "231" for mitdb:231; QTDB excerpts map back to their source
    # database: sel16272 -> NSRDB 16272, sele0107 -> European ST-T e0107, sel30 -> SDDB 30
    held_ids = {re.sub(r"^sel(e?)", lambda m: m.group(1), g.split(":", 1)[1]) for g in held_groups}
    groups = json.load(open(groups_file)) if groups_file else {}
    bad = []
    for f in files:
        name = os.path.basename(f)
        rec = name.split("_")[0].split(".")[0]
        if rec in held:
            bad.append(f"{name}: record {rec} is in the {held[rec]['split']} split")
        elif rec in held_ids:
            bad.append(f"{name}: record {rec} shares its number with a val/test patient")
        g = groups.get(name)
        if g in held_groups:
            bad.append(f"{name}: patient group {g} is in val/test")
    if bad:
        sys.exit("refusing unlabelled data from val/test patients:\n  " + "\n  ".join(bad))
    if not groups_file:
        print("note: no --groups file; only record names were checked against val/test", flush=True)


def find_files(folder):
    files = sorted(f for ext in ("json", "hea", "npy", "csv")
                   for f in glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    if not files:
        sys.exit(f"no .json/.hea/.npy/.csv files under {folder}")
    return files


def load_unlabeled(files, fs, lead, max_hours):
    """Returns a list of dicts (name, signal at 128 Hz, rpeaks), cached per file."""
    # a fixed pseudo-random order, so that --max-hours samples across every source, not just the first folder
    files = sorted(files, key=lambda f: hashlib.sha1(os.path.basename(f).encode()).hexdigest())
    os.makedirs(CACHE, exist_ok=True)
    out, seconds = [], 0.0
    for f in files:
        key = hashlib.sha1(f"{os.path.abspath(f)}|{os.path.getmtime(f)}|{fs}|{lead}".encode()).hexdigest()[:16]
        path = os.path.join(CACHE, key + ".pkl")
        if os.path.exists(path):
            recs = pickle.load(open(path, "rb"))
        else:
            recs = []
            for i, (x, f_fs) in enumerate(_read(f, fs, lead)):
                x = _to_fs(x, f_fs).astype(np.float32)
                recs.append(dict(name=f"{os.path.basename(f)}#{i}", signal=x,
                                 rpeaks=elgendi_rpeaks(x.astype(float), FS)))
            pickle.dump(recs, open(path, "wb"), protocol=4)
        out += recs
        seconds += sum(len(r["signal"]) / FS for r in recs)
        if max_hours and seconds >= max_hours * 3600:
            break
    print(f"unlabelled: {len(out)} signals, {seconds/3600:.1f} h from {len(files)} files", flush=True)
    return out


# ---------------------------------------------------------------- pseudo-labelling
def unlabeled_features(recs):
    out = []
    for r in recs:
        X, meta, ctx = extract(r["signal"].astype(float), FS, PRM, r["rpeaks"])
        if len(meta):
            out.append(dict(rec=r, X=X, meta=meta))
    return out


def pseudo_label(model, feats, k, min_agree, bank_size=200000):
    """Converts the neighbour-vote soft labels into hard one-P-per-beat labels and agreement weights."""
    X = np.concatenate([f["X"] for f in feats])
    p = model.predict_proba(X)[:, 1]
    Z = model[0].transform(X)                                   # the StandardScaler in the pipeline
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9        # on unit vectors, cosine is Euclidean
    # neighbour bank: all candidates, or a fixed random subset of size bank_size (tree search)
    bank = np.arange(len(Z)) if len(Z) <= bank_size else \
        np.sort(np.random.default_rng(0).choice(len(Z), bank_size, replace=False))
    nn = NearestNeighbors(n_neighbors=min(k, len(bank)), algorithm="ball_tree").fit(Z[bank])
    q = np.empty(len(Z))
    for s in range(0, len(Z), 20000):                           # processing in chunks keeps memory bounded
        _, idx = nn.kneighbors(Z[s:s + 20000])
        q[s:s + 20000] = p[bank[idx]].mean(axis=1)
    agree = np.abs(2 * q - 1)
    y = np.zeros(len(q), int)
    beat_conf, pseudo_p = [], []
    o = 0
    for f in feats:
        n = len(f["meta"])
        qf, best = q[o:o + n], {}
        for j, (bi, idx, _) in enumerate(f["meta"]):
            if bi not in best or qf[j] > qf[best[bi]]:
                best[bi] = j
        pos = [j for j in best.values() if qf[j] >= 0.5]
        y[o + np.asarray(pos, int)] = 1
        conf = {f["meta"][j][2]: agree[o + j] for j in best.values()}   # R sample -> per-beat confidence
        beat_conf.append(conf)
        pseudo_p.append(np.array([f["meta"][j][1] for j in pos]) / FS)
        o += n
    keep = agree >= min_agree
    print(f"  pseudo-labels: {keep.sum()} of {len(q)} candidates kept, "
          f"{y[keep].sum()} positives, mean agreement {agree.mean():.2f}", flush=True)
    return X[keep], y[keep], agree[keep], beat_conf, pseudo_p


def augmented_rows(feats, beat_conf, pseudo_p, copies, min_agree, rnd):
    """Strongly augmented copies labelled with the pseudo P positions of the clean signal."""
    Xs, ys, ws = [], [], []
    for c in range(copies):
        for f, conf, pp in zip(feats, beat_conf, pseudo_p):
            r = f["rec"]
            seed = int(hashlib.sha1(f"{r['name']}/{rnd}/{c}".encode()).hexdigest()[:8], 16)
            x = augment(r["signal"].astype(float), FS, r["rpeaks"], np.random.default_rng(seed), "strong")
            X, meta, _ = extract(x, FS, PRM, elgendi_rpeaks(x, FS))
            if not len(meta):
                continue
            r_clean = np.array(sorted(conf))
            w = np.zeros(len(meta))
            for j, (_, _, rr) in enumerate(meta):            # the weight of the matching clean beat
                if len(r_clean):
                    i = np.argmin(np.abs(r_clean - rr))
                    if abs(r_clean[i] - rr) <= 0.08 * FS:
                        w[j] = conf[r_clean[i]]
            keep = w >= min_agree
            Xs.append(X[keep]); ys.append(label(meta, FS, pp)[keep]); ws.append(w[keep])
    if not Xs:
        return np.zeros((0, 22)), np.zeros(0, int), np.zeros(0)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


# ---------------------------------------------------------------- model training
def fit_student(Xtr, ytr, wtr, Xu, yu, au, lam):
    # the unlabelled pool = lam x the weight of one labelled dataset, spread by agreement
    per_dataset = wtr.sum() / len(DATASETS)
    wu = au * (lam * per_dataset / max(au.sum(), 1e-9))
    m = MODELS["logreg"]()
    m.fit(np.concatenate([Xtr, Xu]), np.concatenate([ytr, yu]),
          logisticregression__sample_weight=np.concatenate([wtr, wu]))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unlabeled", required=True)
    ap.add_argument("--fs", type=float, default=None, help="sampling rate of .npy/.csv input")
    ap.add_argument("--lead", type=int, default=0, help="WFDB channel")
    ap.add_argument("--groups", default=None, help='JSON {"file name": "patient group"}')
    ap.add_argument("--teacher", default=os.path.join(HERE, "models", "pwave_ml.pkl"))
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-agree", type=float, default=0.4)
    ap.add_argument("--bank-size", type=int, default=200000, help="max candidates searched for neighbours")
    ap.add_argument("--aug-copies", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=0, help="cap on unlabelled hours (0 = all)")
    a = ap.parse_args()

    files = find_files(a.unlabeled)
    check_leakage(files, a.groups)
    recs = load_unlabeled(files, a.fs, a.lead, a.max_hours)
    ufeats = unlabeled_features(recs)
    ftr, fva = featurize("train"), featurize("val")
    Xtr, ytr, wtr = stack(ftr)

    teacher = MLDetector.load(a.teacher)
    base = evaluate(fva, teacher.model, THRS)
    base_thr = max(THRS, key=lambda t: base[t]["macro_f1"])
    print(f"teacher: val={base[base_thr]['macro_f1']*100:.2f} (thr={base_thr})", flush=True)
    best = dict(val=base[base_thr]["macro_f1"], model=teacher.model, thr=base_thr, lam=0.0, round=0)
    log = []
    for lam in a.lambdas:
        model = teacher.model
        for rnd in range(1, a.rounds + 1):
            print(f"lambda_u={lam} round {rnd}", flush=True)
            Xu, yu, au, beat_conf, pseudo_p = pseudo_label(model, ufeats, a.k, a.min_agree, a.bank_size)
            if a.aug_copies:
                Xa, ya, aa = augmented_rows(ufeats, beat_conf, pseudo_p, a.aug_copies, a.min_agree, rnd)
                Xu, yu, au = np.concatenate([Xu, Xa]), np.concatenate([yu, ya]), np.concatenate([au, aa])
            model = fit_student(Xtr, ytr, wtr, Xu, yu, au, lam)
            r = evaluate(fva, model, THRS)
            thr = max(THRS, key=lambda t: r[t]["macro_f1"])
            v = r[thr]
            log.append(dict(lam=lam, round=rnd, thr=thr, val=v["macro_f1"], rows=int(len(yu)),
                            per={d: v[d]["f1"] for d in DATASETS}))
            print(f"  val={v['macro_f1']*100:.2f} (thr={thr})  " +
                  " ".join(f"{d}={v[d]['f1']*100:.1f}" for d in DATASETS), flush=True)
            if v["macro_f1"] > best["val"]:
                best = dict(val=v["macro_f1"], model=model, thr=thr, lam=lam, round=rnd)
    MLDetector(best["model"], best["thr"]).save(os.path.join(HERE, "models", "pwave_ml_semi.pkl"))
    json.dump(dict(teacher=os.path.relpath(a.teacher, HERE), unlabeled=os.path.abspath(a.unlabeled),
                   lambda_u=best["lam"], round=best["round"], thr=best["thr"], val_macro=best["val"],
                   val_teacher=base[base_thr]["macro_f1"], k=a.k, min_agree=a.min_agree,
                   aug_copies=a.aug_copies, log=log),
              open(os.path.join(HERE, "models", "pwave_ml_semi.json"), "w"), indent=1)
    print(f"chosen lambda_u={best['lam']} round={best['round']} thr={best['thr']}: val={best['val']*100:.2f} "
          f"(teacher {base[base_thr]['macro_f1']*100:.2f}); round 0 = teacher kept")


if __name__ == "__main__":
    main()
