"""CNN features for the logistic regression, trained with unlabelled ECG (after ECGMatch).

The logistic regression only sees the 22 hand-crafted features, so unlabelled data cannot teach
it anything new (src/semi_ml.py). Here a small 1-D CNN looks at the raw signal around each
candidate peak and learns its own features, with and without unlabelled ECG. The CNN's
output enters the logistic regression as two extra features:

  cnn_logit       CNN logit that the candidate is the P peak
  cnn_logit_rel   the same minus the best logit of the beat (0 for the beat's best candidate)

CNN input: 80 samples (625 ms at 128 Hz) of the ECG centred on the candidate, with the
window median removed, divided by the record's median R amplitude, and clipped to +-10.

Semi-supervised training (FixMatch with an EMA teacher, following ECGMatch):
  - labelled TRAIN candidates: weakly augmented window, BCE with equal weight per dataset
  - unlabelled candidates: the EMA teacher scores a weakly augmented window; where it is
    confident (p > tau or p < 1 - tau) the student must give the same answer on a
    strongly augmented window (loss weight lambda_u). Starts after the first epoch.
  - augmentations as in src/augment.py, on the window: noise, baseline trend, down/up-
    sampling, gain. None of them moves the candidate.
The EMA teacher is the model that is kept.

Contrastive pre-training ("contrast", SimCLR): before any label is seen, the CNN learns from
the unlabelled windows alone. Two strongly augmented views of the same window must map to
nearby embeddings and away from the other windows of the batch (NT-Xent loss, temperature
--temp, and a small projection head that is thrown away). The pre-trained CNN is then
fine-tuned on the labelled TRAIN candidates, like "sup". The windows stay centred on the
candidate, so the views differ in noise, trend, gain and sampling, never in position.

Stacking without leakage: the CNN features of the TRAIN rows are out-of-fold (patient-group
folds), so the logistic regression does not learn to trust an overfit CNN. The CNN used
on VAL, TEST and at run time is trained on all of TRAIN.

Variants compared on VAL: baseline (22 features), "sup" (CNN without unlabelled data),
"semi" (CNN with unlabelled data, FixMatch), "contrast" (contrastive pre-training on the
unlabelled data, then fine-tuning), "noisy" (noisy student / co-training, below).

Noisy student ("noisy"): the teacher averages two different views of each unlabelled
candidate, the 22-feature logistic regression and the "sup" CNN (both fold-specific, so the
out-of-fold scores stay clean). For each unlabelled beat, the candidate with the highest
teacher probability becomes the P peak if that probability is >= --hi, and every other
candidate a negative. Beats whose best probability lies between --lo and --hi are dropped.
A new CNN is trained on the labelled rows (weak augmentation) plus these pseudo-labelled
windows (strong augmentation, weight --lambda-p). The student thus also learns from the
hand-crafted features, which the CNN alone never sees.

ECGMatch ("ecgmatch"): FixMatch as above, except the pseudo-label uses ECGMatch's neighbour
vote. The EMA teacher's embeddings and probabilities of the last --bank unlabelled windows
form a memory bank. For each new window, q is the mean teacher probability of its --knn
nearest bank entries (cosine). A window counts only if its own teacher prediction and q agree
(both above or both below 0.5), with weight |2q - 1|, ECGMatch's agreement weight. ECGMatch's
label-correlation loss needs several labels per sample and is left out (there is one label here).

Every CNN is an ensemble of --seeds networks with their logits averaged. The best one goes to models/pwave_ml_cnn.pkl and
models/pwave_ml_cnn.json, which MLDetector.load reads like the other models.
The test split is never read.

Usage:
  python src/cnn_feat.py --unlabeled data/unlabelled [--max-hours 50] [--epochs 15]
                         [--lambda-u 1.0] [--tau 0.95] [--folds 3]
                         [--variants sup semi ecgmatch contrast noisy] [--pretrain-epochs 10] [--temp 0.1]
                         [--seeds 3] [--width 1] [--knn 10] [--bank 8192] [--lambda-p 1.0] [--hi 0.8] [--lo 0.2] [--tag ""] [--no-save]
"""
import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                                                       # noqa: E402
from augment import r_amplitude                                     # noqa: E402
from bench import DATASETS                                          # noqa: E402
from pwave import PParams                                           # noqa: E402
from pwave_ml import FEATURES, decide, extract                      # noqa: E402
from train_ml import MODELS, PRM, THRS, evaluate, featurize, stack  # noqa: E402

HERE = splits.HERE
FS = 128
HALF = 40                                   # the window spans candidate -40 .. +39 samples
CNN_FEATURES = ["cnn_logit", "cnn_logit_rel"]
torch.set_num_threads(max(1, os.cpu_count() or 1))


# ---------------------------------------------------------------- windowing
def windows(x, idx, r_amp):
    """[n, 2*HALF] float32 windows of x, centred on the sample indices idx."""
    xp = np.pad(np.asarray(x, float), HALF, mode="edge")
    W = sliding_window_view(xp, 2 * HALF)[np.asarray(idx, int)].copy()
    W -= np.median(W, axis=1, keepdims=True)
    return np.clip(W / max(r_amp, 1e-6), -10, 10).astype(np.float32)


def item_windows(f):
    it = f["item"]
    x = it["signal"].astype(float)
    return windows(x, [m[1] for m in f["meta"]], r_amplitude(x, it["rpeaks"]))


def unlabeled_data(folder, max_hours, need_x=True):
    """Windows (float16, to save memory), the 22 features and the beat ids of every unlabelled
    candidate (cached). need_x=False skips the features (returns None); only the noisy student uses them."""
    from semi_ml import check_leakage, find_files, load_unlabeled
    files = find_files(folder)
    check_leakage(files, None)
    key = hashlib.sha1(f"{os.path.abspath(folder)}|{len(files)}|{max_hours}|{HALF}".encode()).hexdigest()[:12]
    path = os.path.join(HERE, ".cache_unlabeled", f"unl_{key}.npz")
    if os.path.exists(path):
        d = np.load(path)
        if d["W"].dtype != np.float16:          # an old float32 cache: convert it once
            np.savez(path + ".tmp.npz", W=d["W"].astype(np.float16), X=d["X"], beat=d["beat"])
            os.replace(path + ".tmp.npz", path)
            d = np.load(path)
        return d["W"], d["X"] if need_x else None, d["beat"]
    Ws, Xs, beats = [], [], []
    for n, r in enumerate(load_unlabeled(files, None, 0, max_hours)):
        x = r["signal"].astype(float)
        X, meta, _ = extract(x, FS, PRM, r["rpeaks"])
        if meta:
            Ws.append(windows(x, [m[1] for m in meta], r_amplitude(x, r["rpeaks"])))
            Xs.append(X.astype(np.float32))
            beats.append(n * 100000 + np.array([m[0] for m in meta], np.int64))
    W, X, beat = np.concatenate(Ws).astype(np.float16), np.concatenate(Xs), np.concatenate(beats)
    np.savez(path, W=W, X=X, beat=beat)
    return W, X if need_x else None, beat


# ---------------------------------------------------------------- augmentation
AUG = {
    "weak": dict(noise=(0.01, 0.03), trend=(0.0, 0.1), resample=(1.2, 1.6), scale=(0.8, 1.25)),
    "strong": dict(noise=(0.03, 0.08), trend=(0.1, 0.3), resample=(1.6, 2.5), scale=(0.5, 2.0)),
}


def augment_windows(W, strength, g):
    """Applies each transform on a random half of the batch. W: [B, L] tensor (float16 or float32)."""
    cfg, (B, L) = AUG[strength], W.shape
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, 1, generator=g)
    on = torch.rand(B, 4, generator=g) < 0.5
    W = W.clone() if W.dtype == torch.float32 else W.float()
    m = on[:, 0]
    if m.any():                             # down/up-sampling, with one factor per batch
        f = float(u(*cfg["resample"])[0, 0])
        # anti-aliasing needs the 2-D (bilinear) form, so the window is treated as a 1 x L image
        low = F.interpolate(W[m][:, None, None], size=(1, max(2, round(L / f))), mode="bilinear",
                            align_corners=False, antialias=True)
        W[m] = F.interpolate(low, size=(1, L), mode="bilinear", align_corners=False)[:, 0, 0]
    W = torch.where(on[:, 1:2], W * u(*cfg["scale"]), W)
    ramp = torch.linspace(-1, 1, L)[None]
    sign = torch.where(torch.rand(B, 1, generator=g) < 0.5, -1.0, 1.0)
    W = torch.where(on[:, 2:3], W + sign * u(*cfg["trend"]) * ramp, W)
    W = torch.where(on[:, 3:4], W + u(*cfg["noise"]) * torch.randn(B, L, generator=g), W)
    return W


# ---------------------------------------------------------------- model
class Net(nn.Module):
    """`width` multiplies every channel count and the hidden layer (at 1: about 25k weights)."""

    def __init__(self, width=1):
        super().__init__()
        c = 16 * width
        self.hidden = 32 * width
        self.conv = nn.Sequential(
            nn.Conv1d(1, c, 7, padding=3), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Conv1d(c, 2 * c, 5, stride=2, padding=2), nn.BatchNorm1d(2 * c), nn.ReLU(),
            nn.Conv1d(2 * c, 2 * c, 5, stride=2, padding=2), nn.BatchNorm1d(2 * c), nn.ReLU(),
            nn.Conv1d(2 * c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU())
        # flatten rather than pool: where a shape sits relative to the centre matters
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(c * HALF // 2, self.hidden),
                                  nn.ReLU(), nn.Linear(self.hidden, 1))

    def forward(self, x):
        return self.head(self.conv(x[:, None])).squeeze(1)

    def embed(self, x):
        """The representation before the classifier (flatten + first linear layer)."""
        return self.head[2](self.head[0](self.conv(x[:, None])))


class Ensemble(nn.Module):
    """The mean logit over several networks."""

    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        return torch.stack([n(x) for n in self.nets]).mean(0)


def ema_update(teacher, student, decay):
    with torch.no_grad():
        for t, s in zip(teacher.state_dict().values(), student.state_dict().values()):
            if t.dtype.is_floating_point:
                t.mul_(decay).add_(s, alpha=1 - decay)
            else:
                t.copy_(s)


def pretrain_contrastive(Wu, epochs, seed, temp=0.1, bs=256, width=1):
    """SimCLR on unlabelled windows. Returns the CNN's state_dict (the projection head dropped)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Wu = torch.from_numpy(Wu)
    net = Net(width)
    proj = nn.Sequential(nn.ReLU(), nn.Linear(net.hidden, 32), nn.ReLU(), nn.Linear(32, 16))
    steps = len(Wu) // bs
    opt = torch.optim.AdamW(list(net.parameters()) + list(proj.parameters()), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * steps)
    target = torch.cat([torch.arange(bs, 2 * bs), torch.arange(bs)])   # view i <-> view i + bs
    eye = torch.eye(2 * bs, dtype=torch.bool)
    for ep in range(epochs):
        t0, perm, tot, acc = time.time(), torch.randperm(len(Wu), generator=g), 0.0, 0.0
        net.train()
        for s in range(steps):
            x = Wu[perm[s * bs:(s + 1) * bs]]
            z = F.normalize(proj(net.embed(torch.cat([augment_windows(x, "strong", g),
                                                      augment_windows(x, "strong", g)]))), dim=1)
            sim = (z @ z.T / temp).masked_fill(eye, float("-inf"))
            loss = F.cross_entropy(sim, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            acc += float((sim.argmax(1) == target).float().mean())
        print(f"    contrastive epoch {ep + 1}/{epochs} loss={tot / steps:.4f} "
              f"positive found={acc / steps * 100:.0f}% {time.time() - t0:.0f}s", flush=True)
    return copy.deepcopy(net.state_dict())


def train_cnn(Wl, yl, wl, Wu, lam_u, epochs, tau, seed, bs=256, ema=0.99, log="", init=None,
              Wp=None, yp=None, lam_p=0.0, width=1, knn=0, bank=8192):
    """Returns the EMA teacher. With Wu=None or lam_u=0 it trains on the labelled rows only.
    init: state_dict to start from (contrastive pre-training).
    Wp, yp: pseudo-labelled windows (noisy student), strongly augmented, weight lam_p.
    knn > 0: ECGMatch neighbour vote over a bank of the last `bank` teacher embeddings."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Wl, yl, wl = torch.from_numpy(Wl), torch.from_numpy(yl.astype(np.float32)), \
        torch.from_numpy((wl / wl.mean()).astype(np.float32))
    Wu = torch.from_numpy(Wu) if (Wu is not None and lam_u > 0) else None
    if Wp is not None:
        Wp, yp = torch.from_numpy(Wp), torch.from_numpy(yp.astype(np.float32))
    net = Net(width)
    if init is not None:
        net.load_state_dict(init)
    teacher = copy.deepcopy(net).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    steps = len(yl) // bs
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * steps)
    bank_z, bank_p = torch.zeros(0, net.hidden), torch.zeros(0)
    for ep in range(epochs):
        t0, perm, tot, used = time.time(), torch.randperm(len(yl), generator=g), 0.0, []
        net.train()
        for s in range(steps):
            b = perm[s * bs:(s + 1) * bs]
            xs = [augment_windows(Wl[b], "weak", g)]
            fixmatch = Wu is not None and ep >= 1
            if fixmatch:
                ub = torch.randint(len(Wu), (bs,), generator=g)
                with torch.no_grad():
                    if knn:
                        h = teacher.embed(augment_windows(Wu[ub], "weak", g))
                        pt = torch.sigmoid(teacher.head[4](teacher.head[3](h)).squeeze(1))
                        z = F.normalize(h, dim=1)
                        if len(bank_p) >= knn:
                            q = bank_p[(z @ bank_z.T).topk(knn, dim=1).indices].mean(1)
                            mask = ((pt > 0.5) == (q > 0.5)).float() * (2 * q - 1).abs()
                        else:
                            q, mask = pt, torch.zeros(bs)
                        bank_z, bank_p = torch.cat([bank_z, z])[-bank:], torch.cat([bank_p, pt])[-bank:]
                        pt = q
                    else:
                        pt = torch.sigmoid(teacher(augment_windows(Wu[ub], "weak", g)))
                        mask = ((pt > tau) | (pt < 1 - tau)).float()
                xs.append(augment_windows(Wu[ub], "strong", g))
                used.append(float(mask.mean()))
            if Wp is not None:
                pb = torch.randint(len(Wp), (bs,), generator=g)
                xs.append(augment_windows(Wp[pb], "strong", g))
            out = net(torch.cat(xs))
            loss = (F.binary_cross_entropy_with_logits(out[:bs], yl[b], reduction="none") * wl[b]).mean()
            k = bs
            if fixmatch:
                loss = loss + lam_u * (F.binary_cross_entropy_with_logits(
                    out[k:k + bs], (pt > 0.5).float(), reduction="none") * mask).mean()
                k += bs
            if Wp is not None:
                loss = loss + lam_p * F.binary_cross_entropy_with_logits(out[k:k + bs], yp[pb])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ema_update(teacher, net, ema)
            tot += float(loss.detach())
        print(f"    {log} epoch {ep + 1}/{epochs} loss={tot / steps:.4f}" +
              (f" unlabelled weight={np.mean(used) * 100:.0f}%" if used else "") +
              f" {time.time() - t0:.0f}s", flush=True)
    return teacher


def predict(net, W, bs=4096):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(W[s:s + bs]).float()).numpy() for s in range(0, len(W), bs)]) \
            if len(W) else np.zeros(0, np.float32)


def cnn_columns(logit, meta):
    """[n, 2]: the logit and the logit minus the best logit of the same beat."""
    logit = np.clip(np.asarray(logit, float), -10, 10)
    best = {}
    for z, (bi, _, _) in zip(logit, meta):
        best[bi] = max(best.get(bi, -np.inf), z)
    return np.c_[logit, logit - np.array([best[m[0]] for m in meta])]


def with_cnn(feats, logits):
    return [dict(f, X=np.c_[f["X"], cnn_columns(z, f["meta"])]) if len(f["meta"]) else
            dict(f, X=np.zeros((0, len(FEATURES) + 2))) for f, z in zip(feats, logits)]


class CNNOnly:
    """The CNN used alone as a scorer (column len(FEATURES) holds its logit)."""

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X[:, len(FEATURES)]))
        return np.c_[1 - p, p]


# ---------------------------------------------------------------- detector
class CNNDetector:
    """A callable, like detect_pwaves / MLDetector: rule front end + CNN + logistic regression."""

    def __init__(self, model, net, thr, width=1):
        self.model, self.net, self.thr, self.width = model, net.eval(), thr, width

    def __call__(self, ecg, fs, prm=PParams(), rpeaks=None):
        if fs != FS:
            raise ValueError(f"CNN features need {FS} Hz input, got {fs}")
        x = np.asarray(ecg, float)
        X, meta, ctx = extract(x, fs, prm, rpeaks)
        if len(meta) == 0:
            return np.array([])
        z = predict(self.net, windows(x, [m[1] for m in meta], r_amplitude(x, ctx["rpeaks"])))
        X = np.c_[X, cnn_columns(z, meta)]
        return decide(self.model.predict_proba(X)[:, 1], meta, ctx, self.thr)

    def save(self, path):
        pickle.dump(dict(model=self.model, thr=self.thr, features=FEATURES + CNN_FEATURES,
                         cnn={k: v.numpy() for k, v in self.net.state_dict().items()},
                         cnn_n=len(self.net.nets) if isinstance(self.net, Ensemble) else 0,
                         cnn_width=self.width), open(path, "wb"))

    @staticmethod
    def from_dict(d):
        wd = d.get("cnn_width", 1)
        net = Ensemble([Net(wd) for _ in range(d["cnn_n"])]) if d.get("cnn_n") else Net(wd)
        net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in d["cnn"].items()})
        return CNNDetector(d["model"], net, d["thr"], wd)


# ---------------------------------------------------------------- main
def fold_of(group, folds):
    return int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) % folds


def labelled_rows(feats, Ws):
    _, y, w = stack(feats)
    W = np.concatenate([W[f["use"]] for f, W in zip(feats, Ws) if len(f["meta"])])
    return W, y, w


def best_thr(res):
    t = max(THRS, key=lambda t: res[t]["macro_f1"])
    return t, res[t]


def line(name, thr, r):
    return f"{name:22s} thr={thr:.1f}  val={r['macro_f1'] * 100:.2f}  " + \
        " ".join(f"{d}={r[d]['f1'] * 100:.1f}" for d in DATASETS)


def beat_pseudo_labels(p, beat, hi, lo):
    """One P per beat: returns (kept candidate indices, 0/1 labels). Uncertain beats are dropped."""
    _, inv = np.unique(beat, return_inverse=True)
    best = np.full(inv.max() + 1, -1.0)
    np.maximum.at(best, inv, p)
    bp = best[inv]
    keep = np.where((bp >= hi) | (bp <= lo))[0]
    y = ((p == bp) & (bp >= hi)).astype(np.float32)
    return keep, y[keep]


def train_ensemble(n, seed, log, **kw):
    nets = [train_cnn(seed=seed + 1000 * j, log=f"{log} net {j + 1}/{n}", **kw) for j in range(n)]
    return Ensemble(nets).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unlabeled", default=os.path.join(HERE, "data", "unlabelled"))
    ap.add_argument("--max-hours", type=float, default=50)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lambda-u", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.95, help="teacher confidence for a pseudo-label")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=["sup", "semi", "ecgmatch", "contrast", "noisy"],
                    choices=["sup", "semi", "ecgmatch", "contrast", "noisy"])
    ap.add_argument("--knn", type=int, default=10, help="neighbours in the ECGMatch vote")
    ap.add_argument("--bank", type=int, default=8192, help="ECGMatch memory bank size")
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.1, help="NT-Xent temperature")
    ap.add_argument("--seeds", type=int, default=3, help="networks per CNN ensemble")
    ap.add_argument("--width", type=int, default=1, help="CNN channel multiplier (2: about 4x the weights)")
    ap.add_argument("--lambda-p", type=float, default=1.0, help="weight of the noisy-student pseudo-labels")
    ap.add_argument("--hi", type=float, default=0.8, help="teacher probability for a pseudo P")
    ap.add_argument("--lo", type=float, default=0.2, help="below this the whole beat is a negative")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--no-save", action="store_true", help="only write results_cnn_val<tag>.json")
    a = ap.parse_args()
    if "noisy" in a.variants and "sup" not in a.variants:
        a.variants.insert(0, "sup")                      # the noisy-student teacher requires the sup CNNs

    ftr, fva = featurize("train"), featurize("val")
    Wtr, Wva = [item_windows(f) for f in ftr], [item_windows(f) for f in fva]
    Wu, Xu, beat_u = unlabeled_data(a.unlabeled, a.max_hours, need_x="noisy" in a.variants) \
        if {"semi", "ecgmatch", "contrast", "noisy"} & set(a.variants) else (None, None, None)
    print(f"windows: train {sum(len(w) for w in Wtr)}, val {sum(len(w) for w in Wva)}, "
          f"unlabelled {0 if Wu is None else len(Wu)}", flush=True)

    Xtr, ytr, wtr = stack(ftr)
    base = MODELS["logreg"]()
    base.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
    thr, r = best_thr(evaluate(fva, base, THRS))
    print(line("baseline (22 features)", thr, r), flush=True)
    rows = [dict(variant="baseline", thr=thr, val=r["macro_f1"], per={d: r[d]["f1"] for d in DATASETS},
                 model=base, net=None)]

    folds = np.array([fold_of(f["item"]["group"], a.folds) for f in ftr])
    parts = [[i for i in range(len(ftr)) if folds[i] != k] for k in range(a.folds)] + [list(range(len(ftr)))]
    sup_nets = None
    for variant in a.variants:
        lam = a.lambda_u if variant in ("semi", "ecgmatch") else 0.0
        init = None
        if variant == "contrast":
            print("  contrastive pre-training on the unlabelled windows", flush=True)
            init = pretrain_contrastive(Wu, a.pretrain_epochs, a.seed + 200, a.temp, width=a.width)
        print(f"variant {variant}", flush=True)
        nets = []
        for k, inner in enumerate(parts):                # folds 0..K-1, then the whole of train
            W, y, w = labelled_rows([ftr[i] for i in inner], [Wtr[i] for i in inner])
            extra = {}
            if variant == "noisy":
                Xi, yi, wi = stack([ftr[i] for i in inner])
                lr = MODELS["logreg"]()
                lr.fit(Xi, yi, logisticregression__sample_weight=wi)
                pt = 0.5 * (lr.predict_proba(Xu.astype(float))[:, 1] + 1 / (1 + np.exp(-predict(sup_nets[k], Wu))))
                keep, yp = beat_pseudo_labels(pt, beat_u, a.hi, a.lo)
                print(f"  pseudo-labels: {len(keep)} of {len(pt)} candidates, {int(yp.sum())} P", flush=True)
                extra = dict(Wp=Wu[keep], yp=yp, lam_p=a.lambda_p)
            log = f"fold {k + 1}/{a.folds}" if k < a.folds else "all train"
            nets.append(train_ensemble(a.seeds, a.seed + (k if k < a.folds else 100), log,
                                       Wl=W, yl=y, wl=w, Wu=Wu, lam_u=lam, epochs=a.epochs, tau=a.tau,
                                       init=init, width=a.width, bank=a.bank,
                                       knn=a.knn if variant == "ecgmatch" else 0, **extra))
        if variant == "sup":
            sup_nets = nets
        oof = [None] * len(ftr)
        for k in range(a.folds):
            for i in np.where(folds == k)[0]:
                oof[i] = predict(nets[k], Wtr[i])
        net = nets[-1]
        ftr_c = with_cnn(ftr, oof)
        fva_c = with_cnn(fva, [predict(net, W) for W in Wva])
        thr, r = best_thr(evaluate(fva_c, CNNOnly(), THRS))
        print(line(f"CNN alone ({variant})", thr, r), flush=True)
        X, y, w = stack(ftr_c)
        m = MODELS["logreg"]()
        m.fit(X, y, logisticregression__sample_weight=w)
        thr, r = best_thr(evaluate(fva_c, m, THRS))
        print(line(f"LR + CNN ({variant})", thr, r), flush=True)
        rows.append(dict(variant=variant, thr=thr, val=r["macro_f1"], per={d: r[d]["f1"] for d in DATASETS},
                         model=m, net=net, train=evaluate(ftr_c, m, [thr])[thr]["macro_f1"]))

    best = max(rows, key=lambda x: x["val"])
    summary = dict(chosen=best["variant"], thr=best["thr"], val_macro=best["val"], val=best["per"],
                   variants={x["variant"]: dict(thr=x["thr"], val_macro=x["val"], val=x["per"],
                                                train_macro_oof=x.get("train")) for x in rows},
                   epochs=a.epochs, seeds=a.seeds, width=a.width, knn=a.knn, bank=a.bank, lambda_u=a.lambda_u, tau=a.tau,
                   pretrain_epochs=a.pretrain_epochs, temp=a.temp, lambda_p=a.lambda_p, hi=a.hi, lo=a.lo,
                   folds=a.folds, max_hours=a.max_hours, unlabelled_windows=0 if Wu is None else len(Wu))
    json.dump(summary, open(os.path.join(HERE, f"results_cnn_val{a.tag}.json"), "w"), indent=1)
    if a.no_save:
        print(f"best {best['variant']} thr={best['thr']}: val={best['val'] * 100:.2f} (not saved)")
    elif best["net"] is None:
        print("no CNN variant beats the baseline on val; nothing saved")
    else:
        out = os.path.join(HERE, "models", f"pwave_ml_cnn{a.tag}")
        CNNDetector(best["model"], best["net"], best["thr"], a.width).save(out + ".pkl")
        json.dump(summary, open(out + ".json", "w"), indent=1)
        print(f"chosen {best['variant']} thr={best['thr']}: val={best['val'] * 100:.2f} -> {out}.pkl")


if __name__ == "__main__":
    main()
