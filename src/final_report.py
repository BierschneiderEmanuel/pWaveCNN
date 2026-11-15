"""One-off evaluation on the held-out TEST split (plus train/val for reference).

Compares the rule detector, the learned detector with the 22-feature logistic regression
(models/pwave_ml.pkl) and the default learned detector, which adds the CNN ensemble
(pwave_ml.DEFAULT_MODEL). Nothing is tuned here.

Usage: python src/final_report.py [--model models/pwave_ml_aug.pkl]
With --model, that scorer is added as a third method, and the results are written to
results_final_<model name>.json instead of results_final.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                       # noqa: E402
from bench import DATASETS, run      # noqa: E402
from pwave import PParams           # noqa: E402
from pwave_ml import DEFAULT_MODEL, LR_MODEL, MLDetector  # noqa: E402

if __name__ == "__main__":
    methods = {
        "rule detector": (PParams(), None),
        "learned detector": (PParams(), MLDetector.load(LR_MODEL)),
        "learned detector + CNN": (PParams(), MLDetector.load(DEFAULT_MODEL)),
    }
    extra = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
    tag = ""
    if extra:
        tag = os.path.splitext(os.path.basename(extra))[0].replace("pwave_ml", "").strip("_")
        methods[f"learned detector ({tag})"] = (PParams(), MLDetector.load(extra))
        tag = "_" + tag
    out = {}
    for sp in ("train", "val", "test"):
        items = splits.load(sp)
        for name, (prm, det) in methods.items():
            r = run(items, prm) if det is None else run(items, prm, detector=det)
            out[(sp, name)] = r
            print(f"[{sp:5s}] {name:32s} macro F1={r['macro_f1']*100:5.2f}  pooled F1={r['pooled_f1']*100:5.2f}  " +
                  "  ".join(f"{d}: Se={r[d]['se']*100:4.1f} PPV={r[d]['ppv']*100:4.1f} F1={r[d]['f1']*100:4.1f}"
                            for d in DATASETS), flush=True)
    json.dump({f"{sp}|{n}": {d: {k: float(v) for k, v in r[d].items()} for d in DATASETS} | {"macro_f1": r["macro_f1"], "pooled_f1": r["pooled_f1"]}
               for (sp, n), r in out.items()}, open(os.path.join(splits.HERE, f"results_final{tag}.json"), "w"), indent=1)
