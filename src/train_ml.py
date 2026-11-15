"""Trains the learned candidate scorer on TRAIN, and selects the model + threshold on VAL.

Never touches the test split. Saves the chosen model to models/pwave_ml.pkl.

With --augment, the logistic regression is also trained on augmented copies of the TRAIN
signals (src/augment.py). The strength, the number of copies, the weight of the augmented
rows and the threshold are chosen on VAL, and the result goes to models/pwave_ml_aug.pkl.
"""
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                                            # noqa: E402
from augment import augment                              # noqa: E402
from bench import DATASETS, aggregate, score_item        # noqa: E402
from elgendi import elgendi_rpeaks                       # noqa: E402
from pwave import PParams                                # noqa: E402
from pwave_ml import FEATURES, MLDetector, decide, extract, label  # noqa: E402

HERE = splits.HERE
PRM = PParams()


def featurize_item(it, signal=None, rpeaks=None):
    """The candidate features and labels of one item; `signal` replaces the stored ECG (augmentation)."""
    fs = it["fs"]
    X, meta, ctx = extract(it["signal"].astype(float) if signal is None else signal, fs, PRM,
                           it["rpeaks"] if rpeaks is None else rpeaks)
    y = label(meta, fs, it["p"])
    # rows that can be used for training: beats inside the labelled region
    lo, hi = it["range"]
    use = np.array([lo <= r / fs <= hi for _, _, r in meta], bool)
    if it["label_scope"] == "beats":
        use &= np.array([np.any(np.abs(it["qrs"] - r / fs) <= 0.08) for _, _, r in meta], bool)
    return dict(item=it, X=X, y=y, use=use, meta=meta,
                ctx=None if ctx is None else dict(xp=ctx["xp"], pol=ctx["pol"]))


def featurize(split):
    path = os.path.join(HERE, f".cache_feat_{split}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = [featurize_item(it) for it in splits.load(split)]
    pickle.dump(out, open(path, "wb"), protocol=4)
    return out


def featurize_aug(strength, copy):
    """Features of one augmented copy of every TRAIN item (with the R peaks re-detected on the copy)."""
    path = os.path.join(HERE, f".cache_feat_train_aug_{strength}_{copy}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = []
    for it in splits.load("train"):
        seed = int(hashlib.sha1(f"{it['dataset']}/{it['record']}/{it['lead']}/{strength}/{copy}".encode()).hexdigest()[:8], 16)
        x = augment(it["signal"].astype(float), it["fs"], it["rpeaks"], np.random.default_rng(seed), strength)
        out.append(featurize_item(it, x, elgendi_rpeaks(x, it["fs"])))
    pickle.dump(out, open(path, "wb"), protocol=4)
    return out


def stack(feats, aug_weight=None):
    """Rows, labels and sample weights. Augmented items (flag `aug`) are given `aug_weight` times the weight."""
    X = np.concatenate([f["X"][f["use"]] for f in feats if len(f["meta"])])
    y = np.concatenate([f["y"][f["use"]] for f in feats if len(f["meta"])])
    ds = np.concatenate([[f["item"]["dataset"]] * int(f["use"].sum()) for f in feats if len(f["meta"])])
    w = np.ones(len(y))
    for d in DATASETS:  # equal total weight across datasets (the macro objective)
        m = ds == d
        if m.any():
            w[m] = len(y) / (len(DATASETS) * m.sum())
    if aug_weight is not None:
        w *= np.concatenate([[aug_weight if f.get("aug") else 1.0] * int(f["use"].sum())
                             for f in feats if len(f["meta"])])
    return X, y, w


def evaluate(feats, model, thrs):
    probs = [model.predict_proba(f["X"])[:, 1] if len(f["meta"]) else np.array([]) for f in feats]
    res = {}
    for thr in thrs:
        per = {d: [] for d in DATASETS}
        for f, p in zip(feats, probs):
            det = decide(p, f["meta"], f["ctx"], thr) if len(f["meta"]) else np.array([])
            per[f["item"]["dataset"]].append(
                score_item(f["item"], PRM, detector=lambda *a, _d=det, **k: _d))
        r = {d: aggregate(rows) for d, rows in per.items() if rows}
        r["macro_f1"] = float(np.mean([r[d]["f1"] for d in DATASETS if d in r]))
        res[thr] = r
    return res


MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "hgb_small": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1,
                                                         l2_regularization=1.0, random_state=0),
    "hgb_mid": lambda: HistGradientBoostingClassifier(max_depth=5, max_iter=300, learning_rate=0.05,
                                                       min_samples_leaf=50, l2_regularization=1.0,
                                                       random_state=0),
    "hgb_large": lambda: HistGradientBoostingClassifier(max_depth=8, max_iter=400, learning_rate=0.05,
                                                         min_samples_leaf=100, l2_regularization=1.0,
                                                         random_state=0),
}
THRS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

AUG_GRID = [(strength, copies, weight) for strength in ("weak", "strong")
            for copies in (1, 2) for weight in (0.5, 1.0)]


def train_augmented(ftr, fva):
    rows = []
    base = MODELS["logreg"]()
    Xtr, ytr, wtr = stack(ftr)
    base.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
    rows.append(("none", 0, 0.0, base, evaluate(fva, base, THRS)))
    for strength, copies, weight in AUG_GRID:
        faug = [dict(f, aug=True) for c in range(copies) for f in featurize_aug(strength, c)]
        X, y, w = stack(ftr + faug, aug_weight=weight)
        m = MODELS["logreg"]()
        m.fit(X, y, logisticregression__sample_weight=w)
        rows.append((strength, copies, weight, m, evaluate(fva, m, THRS)))
        best_thr = max(THRS, key=lambda t: rows[-1][4][t]["macro_f1"])
        r = rows[-1][4][best_thr]
        print(f"aug={strength:6s} copies={copies} weight={weight:.1f} rows={len(y)}  thr={best_thr:.1f}  "
              f"val={r['macro_f1']*100:.2f}  " + " ".join(f"{d}={r[d]['f1']*100:.1f}" for d in DATASETS), flush=True)
    strength, copies, weight, m, rva, thr = max(
        ((*row, t) for row in rows for t in THRS), key=lambda x: x[4][x[5]]["macro_f1"])
    rtr = evaluate(ftr, m, [thr])
    MLDetector(m, thr).save(os.path.join(HERE, "models", "pwave_ml_aug.pkl"))
    json.dump(dict(model="logreg", aug=strength, copies=copies, aug_weight=weight, thr=thr,
                   train_macro=rtr[thr]["macro_f1"], val_macro=rva[thr]["macro_f1"],
                   val={d: rva[thr][d]["f1"] for d in DATASETS},
                   val_no_aug=max(rows[0][4][t]["macro_f1"] for t in THRS)),
              open(os.path.join(HERE, "models", "pwave_ml_aug.json"), "w"), indent=1)
    print(f"chosen aug={strength} copies={copies} weight={weight} thr={thr}: "
          f"train={rtr[thr]['macro_f1']*100:.2f} val={rva[thr]['macro_f1']*100:.2f}")


if __name__ == "__main__" and "--augment" in sys.argv:
    train_augmented(featurize("train"), featurize("val"))
elif __name__ == "__main__":
    t0 = time.time()
    ftr, fva = featurize("train"), featurize("val")
    Xtr, ytr, wtr = stack(ftr)
    print(f"features: {len(FEATURES)}, train candidates {len(ytr)} (pos {ytr.sum()}), "
          f"{time.time()-t0:.0f}s", flush=True)
    results = {}
    for name, mk in MODELS.items():
        m = mk()
        if name == "logreg":
            m.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
        else:
            m.fit(Xtr, ytr, sample_weight=wtr)
        rtr = evaluate(ftr, m, THRS)
        rva = evaluate(fva, m, THRS)
        for thr in THRS:
            print(f"{name:10s} thr={thr:.1f}  train={rtr[thr]['macro_f1']*100:.2f}  val={rva[thr]['macro_f1']*100:.2f}  "
                  + " ".join(f"{d}={rva[thr][d]['f1']*100:.1f}" for d in DATASETS), flush=True)
        results[name] = (m, rtr, rva)
    best = max(((n, t) for n in results for t in THRS), key=lambda nt: results[nt[0]][2][nt[1]]["macro_f1"])
    name, thr = best
    m, rtr, rva = results[name]
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    MLDetector(m, thr).save(os.path.join(HERE, "models", "pwave_ml.pkl"))
    json.dump(dict(model=name, thr=thr, train_macro=rtr[thr]["macro_f1"], val_macro=rva[thr]["macro_f1"],
                   val={d: rva[thr][d]["f1"] for d in DATASETS}),
              open(os.path.join(HERE, "models", "pwave_ml.json"), "w"), indent=1)
    print(f"chosen {name} thr={thr}: train={rtr[thr]['macro_f1']*100:.2f} val={rva[thr]['macro_f1']*100:.2f}")
