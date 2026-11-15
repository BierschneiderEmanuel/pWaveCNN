#!/usr/bin/env bash
# Whole mixed experiment from scratch. Each step only uses the splits it is allowed to:
# tuning and fitting use train, choices use validation, test is scored at the end only.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
rm -f .cache_bench.pkl .cache_feat_*.pkl splits.json tuned_params.json
$PY -u src/splits.py                                   > logs/01_splits.log  2>&1   # mixed split + signal cache
$PY -u src/tune2.py                                    > logs/02_tune.log    2>&1   # rule parameters, train
rm -f .cache_feat_*.pkl                                                         # features use the tuned rules
$PY -u src/train_ml.py                                 > logs/03_train_ml.log 2>&1  # LR / boosted trees, chosen on val
$PY -u src/cnn_feat.py --variants sup semi --seeds 3   > logs/04_cnn.log     2>&1   # CNN features, chosen on val
$PY -u src/final_report.py                             > results_final.txt   2>&1   # train / val / test, test once
$PY -u src/bootstrap_ci.py val test --final            > logs/06_bootstrap.log 2>&1
$PY -u src/rhythm_eval.py val test --final             > logs/07_rhythm.log  2>&1
echo done
