"""Classical (non-neural) P-wave peak detector.

Pipeline, per ECG lead
----------------------
1. Pre-processing: a 0.5-40 Hz band-pass (removes baseline wander / HF noise) plus a
   P-band version (0.5-12 Hz) used for P localisation.
2. R peaks are detected with the Elgendi detector (see elgendi.py).
3. For each beat the QRS onset is estimated by walking back from R until the slope
   drops below a fraction of the maximal QRS slope. The P search window is
   [R - 300 ms, QRS onset - guard].
4. A lead-level P template (median of all beats, aligned on R) gives the
   dominant P polarity and the expected latency.
5. In every beat the candidate peaks (local extrema of the dominant polarity)
   are scored by prominence weighted with a Gaussian latency prior.
6. Noise floor: a robust (MAD) estimate of the high-frequency residual
   (0.5-40 Hz signal minus P-band signal) and of the low-frequency activity
   of the TP segment, smoothed over neighbouring beats. A P wave is reported
   only if its prominence exceeds k_noise * noise floor, an absolute minimum
   amplitude and a minimum shape correlation with the template.
"""
import json
import os
from dataclasses import dataclass, fields, make_dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from elgendi import elgendi_rpeaks


@dataclass
class PParams:
    search: float = 0.300        # the search window to the left of R [s]
    p_lp: float = 15.0           # low-pass for the P band [Hz]
    onset_frac: float = 0.10     # the energy fraction that defines the QRS onset
    onset_max: float = 0.200     # maximum search for the QRS onset before R [s]
    guard: float = 0.008         # the gap between the P wave and the QRS onset [s]
    lat_sigma: float = 0.030     # standard deviation of the latency prior [s]
    k_noise: float = 1.0         # the prominence must exceed k * noise floor
    min_amp: float = 0.0        # absolute minimum prominence, in signal units
    min_corr: float = 0.7        # minimum correlation with the lead template
    noise_beats: int = 8         # the beats used to smooth the noise floor
    neg_bias: float = 3.0        # a negative P polarity must be this much stronger
    lf_weight: float = 0.5       # the weight of the low-frequency (in-window) activity in the noise floor
    p_half: float = 0.05         # half-width of the P wave left out of the LF noise estimate [s]
    # --- rules that guard against specific failure modes (LBBB, small P, ectopy, AF) ---
    min_amp_rel: float = 0.015     # minimum prominence relative to the median R amplitude of the lead
    amp_cap: float = 0.02         # upper cap for the relative amplitude threshold, in signal units
    qrs_corr_min: float = 0.7   # beats whose QRS correlates less with the dominant QRS get no P (ectopic)
    consistency: bool = True    # a second pass enforcing beat-to-beat PR latency consistency
    nb_beats: int = 4            # the neighbours on each side used for the local PR latency
    nb_min: int = 2              # the minimum accepted neighbours needed to use the local latency
    nb_sigma: float = 0.025      # standard deviation of the latency prior around the local latency [s]
    nb_tol: float = 0.04         # maximum |latency - local latency| for acceptance [s]
    k_isolated: float = 99.0     # beats lacking enough accepted neighbours need prom >= k_isolated*noise
    relax: float = 0.6           # the gate factor for candidates consistent with neighbours (<1 relaxes)
    interp: bool = True         # parabolic sub-sample interpolation of the peak


# The rule parameters tuned on the train split of this project (src/tune2.py) become the defaults.
# If tuned_params.json is absent, the values above are used (they are tune2's starting point).
TUNED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tuned_params.json")
if os.path.exists(TUNED) and not os.environ.get("PWAVE_UNTUNED"):
    _t = json.load(open(TUNED))["chosen"]["params"]
    PParams = make_dataclass("PParams", [(f.name, f.type, _t.get(f.name, f.default)) for f in fields(PParams)])


def _bp(x, fs, lo, hi, order=2):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def _qrs_energy(ecg, fs):
    """The smoothed QRS energy envelope (8-20 Hz band, 40 ms moving average)."""
    b, a = butter(3, [8.0 / (fs / 2), 20.0 / (fs / 2)], btype="band")
    e = filtfilt(b, a, ecg) ** 2
    n = max(1, int(round(0.04 * fs)))
    return np.convolve(e, np.ones(n) / n, mode="same")


def _qrs_onset(env, r, fs, frac, max_back):
    """Walks back from R until the QRS energy envelope drops below
    frac * (the envelope maximum around R)."""
    lo = max(0, r - int(round(max_back * fs)))
    hi = min(len(env), r + int(round(0.05 * fs)) + 1)
    peak = env[lo:hi].max()
    i = int(np.argmax(env[lo:hi])) + lo
    i = min(i, r)
    while i > lo and env[i] > frac * peak:
        i -= 1
    return i


def _qrs_offset(env, r, fs, frac, max_fwd):
    hi = min(len(env) - 1, r + int(round(max_fwd * fs)))
    lo = max(0, r - int(round(0.05 * fs)))
    peak = env[lo:hi + 1].max()
    i = max(r, lo + int(np.argmax(env[lo:hi + 1])))
    while i < hi and env[i] > frac * peak:
        i += 1
    return i


def detect_pwaves(ecg, fs, prm: PParams = PParams(), rpeaks=None, return_debug=False,
                  return_context=False):
    ecg = np.asarray(ecg, dtype=float)
    xf = _bp(ecg, fs, 0.5, 40.0)
    env = _qrs_energy(ecg, fs)
    if rpeaks is None:
        rpeaks = elgendi_rpeaks(ecg, fs)
    # QRS blanking: replace every QRS (onset..offset) with a straight line, so the
    # P-band low-pass does not smear QRS energy into the PR segment.
    xb = xf.copy()
    for r in rpeaks:
        on = _qrs_onset(env, r, fs, prm.onset_frac, prm.onset_max)
        off = _qrs_offset(env, r, fs, prm.onset_frac, 0.2)
        if off > on + 1:
            xb[on:off + 1] = np.linspace(xf[on], xf[off], off - on + 1)
    b, a = butter(2, prm.p_lp / (fs / 2), btype="low")
    xp = filtfilt(b, a, xb)
    hf = xf - _bp(ecg, fs, 0.5, prm.p_lp)
    n_s = int(round(prm.search * fs))
    guard = int(round(prm.guard * fs))
    ctx = int(round(0.10 * fs))  # extra context, used for prominence only

    beats = []
    for k, r in enumerate(rpeaks):
        if r - n_s - ctx < 0:
            continue
        on = _qrs_onset(env, r, fs, prm.onset_frac, prm.onset_max)
        w_end = min(on - guard, r - 1)
        w_start = r - n_s
        # do not reach back into the previous QRS
        if k > 0:
            w_start = max(w_start, rpeaks[k - 1] + int(round(0.12 * fs)))
        if w_end - w_start < 3:
            continue
        beats.append((r, w_start, w_end))
    if not beats:
        if return_context:
            return None
        return (np.array([]), {}) if return_debug else np.array([])

    # ---------- lead-level template (aligned on R) ----------
    L = n_s + ctx
    segs = np.array([xp[r - L:r] for r, _, _ in beats])
    segs = segs - np.median(segs, axis=1, keepdims=True)
    tmpl = np.median(segs, axis=0)
    t_end = int(np.median([r - we for r, _, we in beats]))   # the typical window end (from R)
    sl = slice(L - n_s, L - t_end)                            # the template search region
    best = None
    edge = 2
    for pol in (1.0, -1.0):
        pk, pr = find_peaks(pol * tmpl, prominence=0)
        for i, prom in zip(pk, pr["prominences"]):
            eff = prom if pol > 0 else prom / prm.neg_bias
            if sl.start <= i < sl.stop - edge and (best is None or eff > best[0]):
                best = (eff, pol, i)
    if best is None:
        pol, lat_t = 1.0, L - int(0.16 * fs)
    else:
        _, pol, lat_t = best
    lat_t = L - lat_t  # the number of samples before R

    # the template snippet around the expected P (used for shape correlation)
    half = int(round(0.06 * fs))
    tc = L - lat_t
    t_snip = tmpl[max(0, tc - half):tc + half + 1]

    # ---------- per-beat noise floor ----------
    hf_noise, lf_noise = [], []
    for r, ws, we in beats:
        h = hf[ws:we]
        hf_noise.append(1.4826 * np.median(np.abs(h - np.median(h))))
    hf_noise = np.array(hf_noise)
    nb = prm.noise_beats
    noise = np.array([np.median(hf_noise[max(0, i - nb):i + nb + 1]) for i in range(len(beats))])

    # ---------- ectopy: correlation of the QRS shape with the dominant QRS ----------
    qh = int(round(0.06 * fs))
    qsegs = np.array([xf[r - qh:r + qh + 1] if r - qh >= 0 and r + qh + 1 <= len(xf)
                      else np.zeros(2 * qh + 1) for r, _, _ in beats])
    qt = np.median(qsegs, axis=0)
    qcorr = np.array([np.corrcoef(q, qt)[0, 1] if np.std(q) > 0 else 0.0 for q in qsegs])
    r_amp = float(np.median(np.abs(xf[[r for r, _, _ in beats]])))
    amp_min = max(prm.min_amp, min(prm.min_amp_rel * r_amp, prm.amp_cap))

    def candidates(bi):
        """All candidate peaks of beat bi: a list of (idx, prom, corr, nf)."""
        r, ws, we = beats[bi]
        seg = pol * xp[ws - ctx:we + 1]
        pk, pr = find_peaks(seg, prominence=0)
        out_c = []
        for i, prom in zip(pk, pr["prominences"]):
            if i < ctx or i > len(seg) - 1 - 2:
                continue
            idx = ws - ctx + i
            sn = xp[idx - half:idx + half + 1]
            if len(sn) == len(t_snip) and np.std(sn) > 0 and np.std(t_snip) > 0:
                corr = float(np.corrcoef(sn, t_snip)[0, 1])
            else:
                corr = 0.0
            ph = int(round(prm.p_half * fs))
            rest = np.r_[xp[ws:max(ws, idx - ph)], xp[min(we, idx + ph):we + 1]]
            lf = 1.4826 * np.median(np.abs(rest - np.median(rest))) if len(rest) >= 4 else 0.0
            nf = np.hypot(noise[bi], prm.lf_weight * lf)
            out_c.append((idx, prom, corr, nf))
        return out_c

    def passes(prom, corr, nf, f=1.0):
        return (prom >= f * prm.k_noise * nf) and (prom >= f * amp_min) and (corr >= prm.min_corr)

    def pick(cands, r, center, sigma):
        best = None
        for c in cands:
            lat = r - c[0]
            w = np.exp(-0.5 * ((lat - center) / (sigma * fs)) ** 2)
            if best is None or c[1] * w > best[0]:
                best = (c[1] * w, c)
        return None if best is None else best[1]

    # ---------- pass 1: the global template latency prior ----------
    cand_all, first = [], []
    for bi, (r, ws, we) in enumerate(beats):
        cands = candidates(bi)
        cand_all.append(cands)
        if prm.qrs_corr_min > -1 and qcorr[bi] < prm.qrs_corr_min:
            first.append(None); continue
        c = pick(cands, r, lat_t, prm.lat_sigma)
        first.append(c if (c is not None and passes(c[1], c[2], c[3])) else None)

    # ---------- pass 2: local beat-to-beat latency consistency ----------
    out, dbg = [], []
    lat1 = np.array([np.nan if c is None else beats[k][0] - c[0] for k, c in enumerate(first)])
    if return_context:  # the internal state for the learned candidate scorer (pwave_ml.py)
        tmpl_prom = 0.0 if best is None else float(best[0])
        return dict(fs=fs, beats=beats, cand_all=cand_all, lat1=lat1, qcorr=qcorr, r_amp=r_amp,
                    noise=noise, lat_t=lat_t, pol=pol, xp=xp, xf=xf, rpeaks=np.asarray(rpeaks),
                    tmpl_prom=tmpl_prom, amp_min=amp_min)
    for bi, (r, ws, we) in enumerate(beats):
        ectopic = prm.qrs_corr_min > -1 and qcorr[bi] < prm.qrs_corr_min
        if not prm.consistency:
            c = None if ectopic else pick(cand_all[bi], r, lat_t, prm.lat_sigma)
            if c is None:
                dbg.append((r, None, 0, noise[bi], 0)); continue
            dbg.append((r, c[0], c[1], c[3], c[2]))
            if passes(c[1], c[2], c[3]):
                out.append(c[0])
            continue
        if ectopic:
            dbg.append((r, None, 0, noise[bi], 0)); continue
        lo, hi = max(0, bi - prm.nb_beats), min(len(beats), bi + prm.nb_beats + 1)
        nb = np.r_[lat1[lo:bi], lat1[bi + 1:hi]]
        nb = nb[~np.isnan(nb)]
        if len(nb) >= prm.nb_min:
            center = float(np.median(nb))
            c = pick(cand_all[bi], r, center, prm.nb_sigma)
            ok = c is not None and abs((r - c[0]) - center) <= prm.nb_tol * fs and passes(c[1], c[2], c[3], prm.relax)
        else:
            c = pick(cand_all[bi], r, lat_t, prm.lat_sigma)
            ok = c is not None and passes(c[1], c[2], c[3], prm.k_isolated / max(prm.k_noise, 1e-9)
                                          if prm.k_noise > 0 else 1.0)
        if c is None:
            dbg.append((r, None, 0, noise[bi], 0)); continue
        dbg.append((r, c[0], c[1], c[3], c[2]))
        if ok:
            out.append(c[0])

    if prm.interp and out:
        ref = []
        for idx in out:
            if 0 < idx < len(xp) - 1:
                y0, y1, y2 = pol * xp[idx - 1], pol * xp[idx], pol * xp[idx + 1]
                den = y0 - 2 * y1 + y2
                ref.append(idx + (0.5 * (y0 - y2) / den if den < 0 else 0.0))
            else:
                ref.append(float(idx))
        out_f = np.array(sorted(set(ref)))
        if return_debug:
            return out_f, dict(beats=beats, dbg=dbg, pol=pol, lat_t=lat_t, tmpl=tmpl, rpeaks=rpeaks, xp=xp, xf=xf)
        return out_f
    out = np.array(sorted(set(out)), dtype=int)
    if return_debug:
        return out, dict(beats=beats, dbg=dbg, pol=pol, lat_t=lat_t, tmpl=tmpl, rpeaks=rpeaks, xp=xp, xf=xf)
    return out
