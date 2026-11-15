"""Inventory of every ECG file in this project (data/labelled, data/unlabelled); see DATA.md.

Writes all_ecg_files.csv: one row per file with the dataset, path, patient group, number of
leads, native sampling rate, duration, P / QRS / AF label counts and whether the file is in
the non-12-lead pool (all of them here).

Usage: python src/inventory.py [--out all_ecg_files.csv]
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import HERE, _but_sources, group_of  # noqa: E402

LAB = os.path.join(HERE, "data", "labelled")
LABELLED = {"butpdb": (LAB, "*_but-pdb_*_annotations.json"),
            "pwave": (LAB, "*_pwave_annotations.json"),
            "qtdb": (LAB, "*_qtdb_annotations.json")}
UNLABELLED = ["icentia11k", "cinc2017", "butqdb"]
COLS = ["dataset", "path", "record", "group", "labelled", "leads", "native_fs", "duration_s",
        "P", "QRS", "AF_intervals", "contains_af", "label_scope", "pool"]


def json_row(ds, path, but_src):
    d = json.load(open(path))
    rec = os.path.basename(path).split(f"_{'but-pdb' if ds == 'butpdb' else ds}_")[0]
    c, scope = Counter(), set()
    fs = dur = None
    for sn in d["recording"]:
        ch = {c_["id"]: c_ for c_ in sn["ch"]}
        orig = ch.get("initial") or sn["ch"][0]
        if fs is None and len(orig["x"]) > 1:
            fs = round(1 / (orig["x"][1] - orig["x"][0]))
            dur = orig["x"][-1] - orig["x"][0]
        for ann in orig.get("annotations", []):
            scope.add(ann.get("label_scope", "all"))
            for lab in ann["labels"]:
                c[lab["name"]] += 1
                if lab["name"] == "AF" and lab["decision"]["verdict"] == 1:
                    c["af_on"] += 1
    leads = len(d["recording"])
    return dict(dataset=ds, path=os.path.relpath(path, HERE), record=rec, group=group_of(ds, rec, but_src),
                labelled="yes", leads=leads, native_fs=fs, duration_s=round(dur, 1),
                P=c["P-Wave"], QRS=c["QRS"], AF_intervals=c["AF"], contains_af=bool(c["af_on"]),
                label_scope="/".join(sorted(scope)), pool="no (12-lead)" if leads >= 12 else "yes")


def wfdb_row(ds, path):
    import wfdb
    h = wfdb.rdheader(path[:-4])
    rec = os.path.basename(path)[:-4]
    if ds == "icentia11k":
        group = "icentia11k:" + rec.split("_")[0]
    elif ds == "butqdb":
        group = "butqdb:" + rec.split("_")[1][:3]          # subject = the first 3 digits of the record id
    else:
        group = "cinc2017:" + rec                          # patients are not individually identified in CinC 2017
    return dict(dataset=ds, path=os.path.relpath(path, HERE), record=rec, group=group, labelled="no",
                leads=h.n_sig, native_fs=h.fs, duration_s=round(h.sig_len / h.fs, 1), P="", QRS="",
                AF_intervals="", contains_af="", label_scope="", pool="yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "all_ecg_files.csv"))
    a = ap.parse_args()
    but_src = _but_sources()
    rows = []
    for ds, (folder, pat) in LABELLED.items():
        rows += [json_row(ds, p, but_src) for p in sorted(glob.glob(os.path.join(folder, pat)))]
    for ds in UNLABELLED:
        rows += [wfdb_row(ds, p) for p in sorted(glob.glob(os.path.join(HERE, "data", "unlabelled", ds, "*.hea")))]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"{'dataset':11s} {'files':>6s} {'groups':>6s} {'leads':>6s} {'fs':>9s} {'hours':>7s} "
          f"{'lead-h':>7s} {'P':>7s} {'QRS':>7s} {'AF files':>8s} pool")
    for ds in list(LABELLED) + UNLABELLED:
        r = [x for x in rows if x["dataset"] == ds]
        h = sum(x["duration_s"] for x in r) / 3600
        lh = sum(x["duration_s"] * x["leads"] for x in r) / 3600
        num = lambda k: sum(x[k] for x in r if x[k] != "")
        print(f"{ds:11s} {len(r):6d} {len({x['group'] for x in r}):6d} "
              f"{'/'.join(map(str, sorted({x['leads'] for x in r}))):>6s} "
              f"{'/'.join(map(str, sorted({x['native_fs'] for x in r}))):>9s} {h:7.1f} {lh:7.1f} "
              f"{num('P') if r[0]['labelled'] == 'yes' else '-':>7} {num('QRS') if r[0]['labelled'] == 'yes' else '-':>7} "
              f"{sum(x['contains_af'] is True for x in r):8d} {r[0]['pool']}")
    pool = [x for x in rows if x["pool"] == "yes"]
    print(f"pool: {len(pool)} files, {len({x['group'] for x in pool})} groups, "
          f"{sum(x['duration_s'] for x in pool) / 3600:.1f} h -> {a.out}")
    # the groups that appear in more than one dataset
    by = {}
    for x in pool:
        by.setdefault(x["group"], set()).add(x["dataset"])
    multi = {g: sorted(v) for g, v in by.items() if len(v) > 1}
    print(f"groups in more than one dataset: {len(multi)}")
    for combo, n in Counter(tuple(v) for v in multi.values()).most_common():
        print("   ", "+".join(combo), n)


if __name__ == "__main__":
    main()
