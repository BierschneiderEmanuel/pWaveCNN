"""Signal augmentations used to train the candidate scorer (after ECGMatch's ECGAugment).

None of them shift the time axis, so the reference P and QRS times stay valid.
Time reversal and lead shuffling from ECGMatch are left out: reversal puts the P wave
behind the QRS, and there is only one lead.

- noise:    white Gaussian noise, std relative to the R amplitude
- wander:   baseline wander, one or two sinusoids at 0.05-0.5 Hz
- resample: anti-aliased downsampling by a random factor, then linear interpolation
            back to the original rate (loses the fine detail a low-rate device would)
- scale:    amplitude gain
"""
import numpy as np
from scipy.signal import butter, filtfilt

# the (low, high) ranges, drawn uniformly per copy
STRENGTH = {
    "weak": dict(noise=(0.01, 0.03), wander=(0.05, 0.15), resample=(1.2, 1.6), scale=(0.8, 1.25)),
    "strong": dict(noise=(0.03, 0.08), wander=(0.15, 0.4), resample=(1.6, 2.5), scale=(0.5, 2.0)),
}


def r_amplitude(x, rpeaks):
    """The median absolute R amplitude, after removing the median baseline."""
    if len(rpeaks) == 0:
        return float(np.std(x)) or 1.0
    return float(np.median(np.abs(x[rpeaks] - np.median(x)))) or 1.0


def add_noise(x, fs, rng, rel, r_amp):
    return x + rng.normal(0.0, rel * r_amp, len(x))


def add_wander(x, fs, rng, rel, r_amp):
    t = np.arange(len(x)) / fs
    for _ in range(rng.integers(1, 3)):
        x = x + rel * r_amp * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t + rng.uniform(0, 2 * np.pi))
    return x


def resample(x, fs, factor):
    """Downsamples by `factor` (anti-aliased), then interpolates back to len(x) samples."""
    b, a = butter(4, 0.9 / factor)          # the cutoff is at 90 % of the new Nyquist rate
    xl = filtfilt(b, a, x)
    t = np.arange(len(x))
    t_low = np.arange(0, len(x) - 1, factor)
    return np.interp(t, t_low, np.interp(t_low, t, xl))


def augment(x, fs, rpeaks, rng, strength="weak"):
    """One augmented copy: each transform is applied with probability 0.5, with at least one always applied."""
    cfg = STRENGTH[strength]
    r_amp = r_amplitude(x, rpeaks)
    on = rng.random(4) < 0.5
    if not on.any():
        on[rng.integers(4)] = True
    order = rng.permutation(4)
    x = np.asarray(x, float).copy()
    for k in order:
        if not on[k]:
            continue
        if k == 0:
            x = add_noise(x, fs, rng, rng.uniform(*cfg["noise"]), r_amp)
        elif k == 1:
            x = add_wander(x, fs, rng, rng.uniform(*cfg["wander"]), r_amp)
        elif k == 2:
            x = resample(x, fs, rng.uniform(*cfg["resample"]))
        else:
            x = x * rng.uniform(*cfg["scale"])
    return x
