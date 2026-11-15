"""Downloads unlabelled ECG (no P-wave labels) into dataset_unlabeled/ for src/semi_ml.py.

None of these databases shares patients with the labelled BUT PDB / PWAVE / LUDB / QTDB
splits (see DATASET.md). Any labels that come with the files (beats, rhythm, quality)
are ignored by semi_ml.py.

  icentia11k/  Icentia11k, 1 lead chest patch, 250 Hz. One 70-min segment (the middle
               one) for each of --icentia-patients patients, spread over all 11,000.
  cinc2017/    PhysioNet/CinC Challenge 2017 training set, 8,528 AliveCor recordings,
               1 lead, 300 Hz, 9-60 s.
  butqdb/      BUT QDB, wearable 1 lead, 1000 Hz, 18 recordings of 24 h or more. Only
               --butqdb-hours one-hour excerpts per recording are streamed, spread over
               the day, converted from uV to mV and resampled to 128 Hz (WFDB).

Everything is WFDB (.hea + .dat/.mat). Re-running the script skips files that already exist.
Files come from PhysioNet's AWS open-data mirror: physionet.org itself served about
10 KB/s from this machine, the mirror about 2 MB/s.
Usage: python src/download_unlabeled.py [--icentia-patients 100] [--butqdb-hours 3]
                                        [--only icentia11k cinc2017 butqdb]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(HERE, "dataset_unlabeled")
PN = "https://physionet-open.s3.amazonaws.com/"


def fetch(url, byte_range=None, tries=5):
    """Fetches the whole file, or the bytes [start, end) when byte_range=(start, end). Retries on network errors."""
    for t in range(tries):
        try:
            req = urllib.request.Request(url)
            if byte_range:
                req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1] - 1}")
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except OSError as e:
            if t == tries - 1:
                raise
            print(f"  retry {url}: {e}", flush=True)
            time.sleep(5 * (t + 1))


def save(path, data):
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def download(url, path, chunk=8 << 20):
    """Downloads a large file to disk in chunks, resuming any partial .part file."""
    if os.path.exists(path):
        return
    tmp = path + ".part"
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
               .headers["Content-Length"])
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    with open(tmp, "ab") as fh:
        while done < size:
            fh.write(fetch(url, (done, min(done + chunk, size))))
            done = fh.tell()
    os.replace(tmp, path)


def icentia(n_patients):
    out = os.path.join(OUT, "icentia11k")
    os.makedirs(out, exist_ok=True)
    got = []
    for pid in np.linspace(0, 10999, n_patients).astype(int):
        folder = f"p{pid // 1000:02d}/p{pid:05d}"
        try:
            segs = fetch(f"{PN}icentia11k-continuous-ecg/1.0/{folder}/RECORDS").decode().split()
        except Exception as e:                          # a handful of patient ids do not exist
            print(f"  icentia p{pid:05d}: {e}", flush=True)
            continue
        seg = segs[len(segs) // 2]
        for ext in (".hea", ".dat"):
            path = os.path.join(out, seg + ext)
            if not os.path.exists(path):
                save(path, fetch(f"{PN}icentia11k-continuous-ecg/1.0/{folder}/{seg}{ext}"))
        got.append(seg)
        if len(got) % 10 == 0:
            print(f"  icentia {len(got)} segments", flush=True)
    return dict(records=len(got), hours=round(len(got) * 1048577 / 250 / 3600, 1),
                source="https://physionet.org/content/icentia11k-continuous-ecg/1.0/",
                note="middle segment of each patient, 250 Hz, mV")


def cinc2017():
    out = os.path.join(OUT, "cinc2017")
    os.makedirs(out, exist_ok=True)
    if len([f for f in os.listdir(out) if f.endswith(".hea")]) < 8528:
        zpath = os.path.join(OUT, "training2017.zip")
        download(f"{PN}challenge-2017/1.0.0/training2017.zip", zpath)
        z = zipfile.ZipFile(zpath)
        for name in z.namelist():
            if name.endswith((".hea", ".mat")):
                save(os.path.join(out, os.path.basename(name)), z.read(name))
        os.remove(zpath)
    heads = [f for f in os.listdir(out) if f.endswith(".hea")]
    secs = sum(int(open(os.path.join(out, h)).readline().split()[3]) / 300 for h in heads)
    return dict(records=len(heads), hours=round(secs / 3600, 1),
                source="https://physionet.org/content/challenge-2017/1.0.0/",
                note="AliveCor single lead, 300 Hz, mV; REFERENCE labels not copied")


def butqdb(hours_per_record):
    import wfdb
    out = os.path.join(OUT, "butqdb")
    os.makedirs(out, exist_ok=True)
    recs = [r for r in fetch(f"{PN}butqdb/1.0.0/RECORDS").decode().split() if r.endswith("_ECG")]
    n = 0
    for rec in recs:
        folder, name = rec.split("/")
        head = fetch(f"{PN}butqdb/1.0.0/{rec}.hea").decode().split("\n")
        length, fs = int(head[0].split()[3]), int(head[0].split()[2])
        fmt, gain_spec, units = head[1].split()[1], *head[1].split()[2].split("/")
        assert fmt == "16" and fs == 1000, head        # single channel, 16-bit little-endian
        gain, baseline = float(gain_spec.split("(")[0]), int(gain_spec.split("(")[1].rstrip(")"))
        hour = 3600 * fs
        starts = np.linspace(hour, length - 2 * hour, hours_per_record).astype(int)   # skip the first/last hour
        for k, s in enumerate(starts):
            stem = f"butqdb_{folder}_h{k:02d}"
            if not all(os.path.exists(os.path.join(out, stem + e)) for e in (".hea", ".dat")):
                raw = fetch(f"{PN}butqdb/1.0.0/{rec}.dat", (2 * int(s), 2 * (int(s) + hour)))
                d = np.frombuffer(raw, "<i2").astype(float)
                x = (d - baseline) / gain / (1000.0 if units == "uV" else 1.0)            # convert to mV
                x = resample_poly(np.nan_to_num(x), 16, 125)                             # resample 1000 to 128 Hz
                x = np.clip(x, -160, 160)       # clip electrode-off artifacts; fits 16 bit at 5 uV steps
                wfdb.wrsamp(stem, fs=128, units=["mV"], sig_name=["ECG"], p_signal=x[:, None],
                            fmt=["16"], adc_gain=[200.0], baseline=[0], write_dir=out,
                            comments=[f"BUT QDB {name}, samples {s}-{s + hour} at 1000 Hz"])
            n += 1
        print(f"  butqdb {folder}: {len(starts)} h", flush=True)
    return dict(records=n, hours=float(n), source="https://physionet.org/content/butqdb/1.0.0/",
                note=f"{hours_per_record} one-hour excerpts per recording, uV->mV, resampled to 128 Hz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--icentia-patients", type=int, default=100)
    ap.add_argument("--butqdb-hours", type=int, default=3)
    ap.add_argument("--only", nargs="+", default=["icentia11k", "cinc2017", "butqdb"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    jobs = {"icentia11k": lambda: icentia(a.icentia_patients), "cinc2017": cinc2017,
            "butqdb": lambda: butqdb(a.butqdb_hours)}
    for name in a.only:
        print(name, flush=True)
        manifest[name] = jobs[name]()
        print(f"  {name}: {manifest[name]['records']} records, {manifest[name]['hours']} h", flush=True)
        json.dump(manifest, open(man_path, "w"), indent=1)
    sys.exit(0)
