# Data of this project

All non-12-lead ECG available for the mixed experiment (see `README.md`). Every file is
listed in **`all_ecg_files.csv`**, one row per file. `src/inventory.py` regenerates it.

## Summary

| Dataset | Files | Patient groups | Leads per file | Native rate | Hours (per lead) | Lead-hours | P labels | QRS labels | Files with AF | In pool |
|---|---|---|---|---|---|---|---|---|---|---|
| BUT PDB | 50 | 45 | 2 | 360 Hz (01–38), 128 Hz (39–50) | 1.7 | 3.3 | 10,858 | 15,276 | 11 | yes |
| PWAVE (MIT-BIH) | 180 | 12 | 2 | 360 Hz | 6.0 | 12.0 | 44,092 | 49,886 | 8 | yes |
| QT Database | 105 | 105 | 2 | 250 Hz | 2.2 | 4.4 | 6,388 | 7,056 | 1 | yes |
| Icentia11k | 100 | 100 | 1 | 250 Hz | 116.5 | 116.5 | none | none | unknown | yes |
| PhysioNet/CinC 2017 | 8,528 | 8,528 (one per file) | 1 | 300 Hz | 77.0 | 77.0 | none | none | unknown | yes |
| BUT QDB | 54 | 15 | 1 | 1000 Hz, stored at 128 Hz | 54.0 | 54.0 | none | none | unknown | yes |
| **Pool total** | **9,017** | **8,785** | | | **257.4** | **267.2** | **61,338** | **72,218** | | |

Counts come from `src/inventory.py`. P and QRS labels are counted per lead, so a P wave
annotated in both leads counts twice. "Files with AF" means the file holds at least one AF
rhythm interval. For the unlabelled sets, rhythm labels were not copied, so AF is unknown.
Icentia11k and CinC 2017 are known to contain AF.

## The datasets

### Labelled (P and QRS peak positions)

All are JSON files in the project format (`recording` → one per lead → `ch`).
Channel `initial` holds the signal at the native rate, and channel `ECG` holds the same
signal at 128 Hz. Labels: `P-Wave` = P peak, `QRS` = QRS position, `AF` = AF rhythm interval.

- **BUT PDB**: `data/labelled/*_but-pdb_buotesd_annotations.json` (50 files).
  - Two-minute, two-lead Holter excerpts.
  - Sources: records 01–38 from the MIT-BIH Arrhythmia Database (360 Hz); 39–43 from the
    MIT-BIH Supraventricular Arrhythmia Database; 44–50 from the Long-Term AF Database. The
    files store records 39–50 at 128 Hz.
  - Rich in pathology: AF, flutter, AV block, bundle-branch block, ectopy.
  - Source: <https://physionet.org/content/butpdb/>
- **PWAVE**: `data/labelled/*_pwave_annotations.json` (180 files).
  - 12 MIT-BIH Arrhythmia records of 30 min, each cut into 15 two-minute files.
  - Records: 100, 101, 103, 106, 117, 119, 122, 207, 214, 222, 223, 231.
  - Source: <https://physionet.org/content/pwave/>
- **QT Database**: `data/labelled/*_qtdb_annotations.json` (105 files).
  - Two leads, cut to the annotated section (q1c) ±2 s.
  - Only selected beats are labelled (`label_scope = beats`): detections outside annotated
    beats cannot be scored.
  - Sources: 15 records from MIT-BIH Arrhythmia, 13 from MIT-BIH Supraventricular, and 77
    from other databases (European ST-T, MIT-BIH Normal Sinus Rhythm, MIT-BIH Long-Term,
    Sudden Cardiac Death Holter, MIT-BIH ST Change).
  - Source: <https://physionet.org/content/qtdb/>

### Unlabelled (signal only)

WFDB files (`.hea` + `.dat`/`.mat`) in `data/unlabelled/`. They were downloaded by
`src/download_unlabeled.py`.

- **Icentia11k**: `data/unlabelled/icentia11k/` (100 files).
  - CardioSTAT chest patch, modified lead I, outpatients.
  - The middle 70-minute segment of 100 patients spread over ids 0–10999.
  - Source: <https://physionet.org/content/icentia11k-continuous-ecg/1.0/>
- **PhysioNet/CinC Challenge 2017**: `data/unlabelled/cinc2017/` (8,528 files).
  - AliveCor handheld, 9–61 s per recording.
  - The rhythm labels (normal, AF, other, noisy) exist in the original download but were not
    copied.
  - Patients are not identified, so every file is its own group.
  - Source: <https://physionet.org/content/challenge-2017/1.0.0/>
- **BUT QDB**: `data/unlabelled/butqdb/` (54 files).
  - Bittium Faros wearable during everyday activity: 18 recordings of 15 subjects, 3 one-hour
    excerpts each.
  - Converted to mV, resampled from 1000 to 128 Hz and clipped at ±160 mV.
  - Source: <https://physionet.org/content/butqdb/1.0.0/>

### Not used

- **LUDB**: 12-lead, excluded by design. It is not on this machine any more (the earlier project folder was deleted).
- **Original downloads**: `~/pWaveDetection/raw/` holds the PhysioNet files from which the
  labelled JSON files were built. Same recordings, not more data.

## Properties of the data

How the split handles them is in `README.md`.

1. **The same patient appears in several datasets.** BUT PDB, PWAVE and QTDB all contain
   excerpts of MIT-BIH patients. The CSV column `group` names the source patient (for example
   `mitdb:231`). 19 patient groups occur in more than one dataset:
   - butpdb + qtdb: 8
   - butpdb + pwave: 6
   - pwave + qtdb: 4
   - all three: 1

   PWAVE's 180 files come from only 12 patients, and BUT QDB's 54 from 15 subjects. If files
   are mixed and split at file level, the same patient can end up in both training and test.
   Split by `group` to avoid that.
2. **Only 335 of the 9,017 pool files have labels** (BUT PDB, PWAVE, QTDB). They make up
   about 10 of the 257 hours. All P-wave evaluation has to use these files.
3. **Two leads in the labelled files, one in the unlabelled files.** Each lead of a labelled
   file has its own labels. Both leads come from the same heartbeat, so they are not
   independent.
4. **Different native rates** (128, 250, 300, 360, 1000 Hz). The labelled JSON files already
   contain a 128 Hz channel. `src/semi_ml.py` and `src/cnn_feat.py` resample the unlabelled
   WFDB files to 128 Hz when reading them.
5. **QTDB labels cover only selected beats** (`label_scope = beats`). Use them only for
   detections that belong to an annotated beat.

## More non-12-lead data that could be downloaded

Not on this machine:

| Dataset | Leads, rate | Size | Notes |
|---|---|---|---|
| [SHDB-AF](https://physionet.org/content/shdb-af/) | 2 leads (Holter) | 128 recordings of 24 h | Paroxysmal AF and flutter |
| Full [Icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) | 1 lead, 250 Hz | 11,000 patients, up to 2 weeks each | Only 100 segments downloaded |
| Full [BUT QDB](https://physionet.org/content/butqdb/1.0.0/) | 1 lead, 1000 Hz | 18 recordings of at least 24 h | Only 3 h per recording downloaded |
| Full source databases of the labelled sets (mitdb, svdb, ltafdb, edb, nsrdb, ltdb, sddb, stdb) | 2 leads | hours to days per record | Unlabelled outside the annotated excerpts |

INCART, PTB-XL, Chapman-Shaoxing and MIMIC-IV-ECG are 12-lead and therefore excluded.
