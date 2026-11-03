# OpenADMET CYP method code: local release package

This package contains **code only**. No challenge data, blinded predictions, fitted models, API credentials, or secrets are included. Run it in a separate output directory supplied by the user. Nothing here uploads data or contacts Hugging Face.

## What is reproduced

| Component | Code | Scope |
| --- | --- | --- |
| Fixed molecular features | `features.py`, `alerts.py`, `expanded_alerts.py` | Morgan/RDKit regression features; Morgan/Avalon/path/RDKit/ErG/alert TDI features. |
| Public binary auxiliary features and regression | `train_regression.py`, `infer_regression.py` | Scaffold OOF LightGBM member with nested, binary-only ChEMBL/PubChem probabilities; trains and predicts from user-supplied inputs. |
| Interval-censored local metric and frozen stress | `proxy.py` | Local assumed ST-RAE denominator and fixed CYP2D6 shift; **not the official scorer**. |
| Aggregate-score shift scenarios and fixed postprocessing | `shift_inference.py`, `recalibrate.py` | Conditional inverse model and the exact frozen `reg_v32` mild coefficients. |
| TDI screening and positive-only decision | `train_tdi.py`, `infer_tdi.py`, `tdi_rules.py` | Nested official-screen probabilities, two LightGBM members, and guarded positive-only rule. |
| Frozen historical configuration snapshots | `regression_plan.json`, `tdi_policy.json` | Audit the original member/search settings; compact trainers have their own explicit parameters. |

The exact historical `reg_v32` entry is a multi-version ensemble built through `reg_v1`–`reg_v30` and then recalibrated. The compact regression trainer here reproduces a **standalone auxiliary member**, not those historical fitted weights or their selected blend. Exact historical reproduction requires the source version chain and artifacts identified in the project reports. Likewise, the exact `tdi_v22` rule requires the historical `tdi_v14` nested baseline scores/calls and its stricter 12-identity exclusion data. The standalone TDI trainer uses canonical molecules and Murcko scaffolds; it is a runnable method reference, **not a claim to regenerate the frozen `tdi_v22` file bit-for-bit**. The fixed recalibration transform itself is exact when applied to the historical `reg_v30` prediction table.

## Environment

Python 3.11 or later; tested against the package versions in `requirements.txt`. RDKit descriptor order and dimensions can vary across releases; use the pinned environment for comparable features. For Parquet auxiliary data, install `pyarrow`; CSV is also supported.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Inputs

Obtain official CYP challenge CSVs under their published license and pass paths explicitly. Do not copy them into this package. Regression training needs `TRAIN_inhibition.csv` with the four `CYP*_pIC50_direct_inhibition` columns and their `_conf_low/_conf_high` columns; TDI training needs `TRAIN_TDI.csv`; both need `Molecule_Name,SMILES`. Screening uses the official single-concentration train CSV with `SMILES,enzyme,log2fc_estimate`. The blinded structure CSV needs `Molecule_Name,SMILES` only.

An optional curated external binary file must have `source,target,label,canonical_smiles,murcko_scaffold,parent_inchikey`, with `label` exactly 0/1 and source ChEMBL or PubChem_AID1851. The trainer deliberately reads **only** these six columns. Obtain and audit this file separately; no external continuous IC50, AC50, or pChEMBL column is consumed.

## Reproduce the compact regression path

```bash
python train_regression.py --train /path/TRAIN_inhibition.csv --blind /path/TEST-BLINDED.csv --external-binary /path/external_binary.csv --out /tmp/cyp-reg
python proxy.py --oof /tmp/cyp-reg/regression_oof.csv --out /tmp/cyp-reg/proxy.json
python infer_regression.py --models /tmp/cyp-reg --structures /path/TEST-BLINDED.csv --out /tmp/cyp-reg/inferred.csv
```

Omit `--external-binary` for the structure-only ablation. Compare the two runs with the **same** folds and metric definition; do not compare either proxy number directly to a Live score.

For historical `reg_v30` predictions and **published aggregate** metrics, run `shift_inference.py --predictions /path/reg_v30_predictions.csv --live-metrics /path/live_metrics.json --out /tmp/shift_scenarios.json`. The JSON is keyed by endpoint and contains `mae`, `r2`, and `spearman`. It reports both possible mean-shift directions and three residual-tail assumptions. The originally selected mild transform can be applied with:

```bash
python recalibrate.py --predictions /path/reg_v30_predictions.csv --out /tmp/reg_v32_mild.csv
```

The recalibration consumes no labels. Its coefficients are frozen in source; running the scenario solver is for audit and **does not silently retune** them.

## Reproduce the compact TDI path

```bash
python train_tdi.py --tdi-train /path/TRAIN_TDI.csv --screen-train /path/single-concentration-TRAIN.csv --blind /path/TEST-BLINDED.csv --out /tmp/cyp-tdi
python infer_tdi.py --models /tmp/cyp-tdi/tdi_models.joblib --structures /path/TEST-BLINDED.csv --out /tmp/cyp-tdi/inferred_scores.csv
```

The model output is **scores**, not final decisions. For the guarded rule, provide an inner calibration CSV and outer/deployment CSV with `Molecule_Name,SMILES,base_score,base_call,meta_score,structure_score`. Inner must also have `target` (0/1); outer may carry the frozen `CYP3A4_is_TDI` calls. The inner scores must be generated with the evaluation fold and each row's own fold excluded from auxiliary and TDI fitting. A conventional OOF table alone does **not** satisfy this nested requirement. The rule searches a fixed grid on inner labels, preserves all base positives, caps additions/call rate, and falls back when guards fail:

```bash
python tdi_rules.py --inner /path/nested_inner.csv --outer /path/frozen_outer.csv --out /tmp/cyp-tdi/tdi_calls.csv --audit /tmp/cyp-tdi/rule_audit.json
```

## Validation boundaries

The local interval metric assumes the denominator is the loss of the evaluation-set mean constant predictor. The official backend was unavailable, so `proxy.py` does not certify official ST-RAE. The fixed stress shift is hypothetical. The scenario solver infers conditional possibilities from aggregate metrics, not hidden labels or confidence intervals. Published Live results and historical local results belong in the accompanying report, with their original `reg_v*/REPORT.md` and `tdi_v*/REPORT.md` provenance.
