# pwave-cnn: P-wave detection on a mixed pool of non-12-lead ECG

This is a new experiment run from scratch. Every labelled and unlabelled **non-12-lead** ECG
file that is available is pooled together. A patient-wise split is made over that mixed pool,
and the full detector is rebuilt on it:

1. the rule parameters are tuned;
2. the logistic regression (LR) is fitted;
3. the CNN candidate features are trained (supervised, and with FixMatch on unlabelled ECG);
4. the test split is scored exactly once at the end.

Nothing is carried over from the earlier project (`pWaveEcgMatch`, now deleted) except the
code and the converted data files: no split, no tuned parameters, no trained model. LUDB
(12-lead) is not used.

## Data

| Source | Folder | Files | Leads | Labels | Hours |
|---|---|---|---|---|---|
| BUT PDB | `data/labelled/*_but-pdb_buotesd_annotations.json` | 50 | 2 | P + QRS + AF | 1.7 |
| PWAVE (MIT-BIH) | `data/labelled/*_pwave_annotations.json` | 180 | 2 | P + QRS + AF | 6.0 |
| QT Database | `data/labelled/*_qtdb_annotations.json` | 105 | 2 | P + QRS, selected beats only | 2.2 |
| Icentia11k | `data/unlabelled/icentia11k/` | 100 | 1 | none | 116.5 |
| PhysioNet/CinC 2017 | `data/unlabelled/cinc2017/` | 8,528 | 1 | none | 77.0 |
| BUT QDB | `data/unlabelled/butqdb/` | 54 | 1 | none | 54.0 |

All 335 labelled files are mixed into one folder. A file's source is read only from its
name, and only in order to report results per source. `raw/but-pdb/README.txt` maps the
BUT PDB records back to their source patients. Full details on every dataset are in
`DATA.md`.

## Split

`src/splits.py` assigns each **patient group** to train / validation / test with probability
60 / 20 / 20 %. The assignment is derived from a hash of `mixed-v1:<group>`. The groups from
all sources are pooled together, with no stratification by source. A patient group holds all
files belonging to one source patient. BUT PDB, PWAVE and QTDB share 19 MIT-BIH patients,
and the files of those patients always fall into the same split. The salt `mixed-v1` was
fixed before any result was seen; the only thing checked afterwards was that every source
appears in every split.

| Source | Train: files (groups), P labels | Validation | Test |
|---|---|---|---|
| BUT PDB | 32 (28), 7,916 | 11 (10), 1,812 | 7 (7), 1,130 |
| PWAVE | 120 (8), 31,244 | 45 (3), 9,842 | 15 (1), 3,006 |
| QTDB | 69 (69), 4,122 | 16 (16), 1,030 | 20 (20), 1,236 |
| **Patient groups** | **92** | **25** | **25** |
| P labels | 43,282 (70.6 %) | 12,684 (20.7 %) | 5,372 (8.8 %) |

Because the split is random by patient, it is uneven in label count. The test split holds
only 8.8 % of the P labels, and its PWAVE part is a single patient.

## Metric

A detection matches a reference P peak within ±50 ms, with one-to-one matching. Two summary
numbers are reported:

- **Macro F1:** the mean of the per-source F1 over BUT PDB, PWAVE and QTDB. It is used for
  every model choice, so that the large PWAVE set does not dominate.
- **Pooled F1:** all files counted as a single dataset. It is reported alongside.

Confidence intervals come from a patient bootstrap (`src/bootstrap_ci.py`, B = 2000).

## Run

```bash
./run_all.sh     # about 3 h on one CPU core; logs in logs/
```

| Step | Script | Split used |
|---|---|---|
| Mixed split and signal cache | `src/splits.py` | none |
| Rule tuning (coordinate descent from the code defaults) | `src/tune2.py` → `tuned_params.json` | train; validation picks the round |
| Candidate scorer: LR or boosted trees, and threshold | `src/train_ml.py` → `models/pwave_ml.pkl` | train; validation picks |
| CNN features: supervised vs FixMatch (50 h unlabelled), 3-network ensembles | `src/cnn_feat.py` → `models/pwave_ml_cnn.pkl` | train; validation picks |
| Final report | `src/final_report.py` → `results_final.txt` | train, validation, **test** |
| Bootstrap intervals | `src/bootstrap_ci.py val test --final` → `results_ci_val_test.txt` | validation, **test** |
| Results inside and outside AF | `src/rhythm_eval.py val test --final` → `results_rhythm_val_test.txt` | validation, **test** |

`src/pwave.py` reads `tuned_params.json` when it is present and uses those rule parameters
as its defaults. `tune2.py` itself always starts from the code defaults. `.venv` is the
project's Python environment (`requirements.txt` plus PyTorch).

## Results

All numbers are in %, from a single run of `run_all.sh` (`results_final.txt`,
`results_ci_val_test.txt`, `results_rhythm_val_test.txt`, and `logs/`). The test split was
scored only once, after all choices had been made on validation.

**Choices made on validation:**
- **Rule tuning:** train macro F1 rose from 87.19 (code defaults) to 87.64. Round 0 was kept.
- **Candidate scorer:** LR at threshold 0.6 (val 89.21). The boosted trees fit train better
  (up to 94.5) but scored lower on validation (at most 87.0).
- **CNN features:** supervised and FixMatch scored almost the same, so the supervised
  ensemble was kept.

| CNN variant (3-network ensemble) | CNN alone | LR + CNN |
|---|---|---|
| none (LR, 22 features) | | 89.21 |
| supervised | 87.21 | **91.57** |
| FixMatch, 50 h unlabelled | 87.25 | 91.36 |

**Final detectors, macro F1 with patient-bootstrap 95 % intervals:**

| Detector | Train | Validation | Test | Test pooled F1 |
|---|---|---|---|---|
| Rules (tuned) | 87.64 | 86.4 [77.8, 92.3] | 82.9 [76.2, 87.1] | 82.60 |
| Rules + LR | 88.64 | 89.2 [81.0, 94.2] | 90.6 [86.4, 93.4] | 91.08 |
| Rules + LR + CNN | 92.75 | 91.6 [84.4, 95.8] | 91.4 [87.8, 94.0] | 91.80 |

**Test F1 per source** (Se / PPV are in `results_final.txt`):

| Detector | BUT PDB | PWAVE | QTDB |
|---|---|---|---|
| Rules | 79.2 | 81.9 | 87.5 |
| Rules + LR | 90.0 | 92.1 | 89.6 |
| Rules + LR + CNN | 91.7 | 92.6 | 90.1 |

**Paired differences in macro F1** (patient bootstrap, B = 2000):

| Comparison | Validation | Test |
|---|---|---|
| LR − rules | +2.79 [+1.06, +5.58] | +7.71 [+5.43, +11.88] |
| LR + CNN − LR | +2.35 [+1.10, +4.08] | +0.87 [−0.09, +2.00] |

**In AF** (`results_rhythm_val_test.txt`): AF beats have almost no reference P waves, so
false detections per 100 AF beats is the useful number. On validation (6 patients, 1,172 AF
beats) it falls from 20.9 (rules) to 12.3 (LR) to 6.7 (LR + CNN). On test (3 BUT PDB
patients, 760 AF beats) it is 18.8, 8.3 and 7.8. Outside AF, LR + CNN reaches F1 93.4 on
test, with 1.2 false detections per 100 beats.

**Reading the results:**
1. The learned scorer is the largest step, and its gain over the rules is clear on both
   splits.
2. The CNN features add a clear +2.35 on validation. On test the gain is smaller (+0.87)
   and its interval touches zero. This matches the pattern in the earlier project (val
   +2.54, test +1.59).
3. FixMatch with unlabelled ECG did not beat supervised training on the mixed pool either.
4. The test split is small: 8.8 % of the P labels. Its PWAVE part is a single patient, so
   the PWAVE test interval has zero width and says nothing about other patients. The test
   numbers are therefore less reliable than the validation numbers.
5. The numbers are not directly comparable with the earlier project. The split, the sources
   per split, and the tuned rule parameters are all different.
