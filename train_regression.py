"""Standalone scaffold-OOF reproduction of the public-binary feature experiment.

This trains a compact member, not the historical multi-version ensemble.
No external continuous assay column is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import build

NAMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
ENDPOINTS = [f"{n}_pIC50_direct_inhibition" for n in NAMES]
SOURCES = {"ChEMBL": 0, "PubChem_AID1851": 1}


def folds(scaffold: np.ndarray) -> np.ndarray:
    if len(set(scaffold)) < 5:
        raise ValueError("At least five Murcko groups are required")
    out = np.empty(len(scaffold), dtype=np.int8)
    for h, (_, val) in enumerate(GroupKFold(5).split(scaffold, groups=scaffold)):
        out[val] = h
    return out


def regressor():
    return lgb.LGBMRegressor(
        n_estimators=180, learning_rate=.04, num_leaves=15,
        min_child_samples=30, reg_lambda=12, colsample_bytree=.9,
        random_state=42, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )


def classifier():
    return lgb.LGBMClassifier(
        n_estimators=75, learning_rate=.055, num_leaves=15,
        min_child_samples=80, reg_lambda=15, colsample_bytree=.9,
        random_state=42, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )


def external_data(path: Path):
    # Deliberately read only six safe binary columns.
    cols = ["source", "target", "label", "canonical_smiles", "murcko_scaffold", "parent_inchikey"]
    if path.suffix == ".parquet":
        ext = pd.read_parquet(path, columns=cols)
    else:
        ext = pd.read_csv(path, usecols=cols)
    ext = ext[ext.target.isin(NAMES) & ext.source.isin(SOURCES) & ext.label.notna()].copy()
    if not set(ext.label.unique()).issubset({0, 1}) or ext.label.nunique() != 2:
        raise ValueError("External activity must be binary 0/1")
    x, canonical, scaffold = build(ext.canonical_smiles, "regression")
    x = np.column_stack((
        x, ext.target.map({n: i for i, n in enumerate(NAMES)}).to_numpy(np.float32),
        ext.source.map(SOURCES).to_numpy(np.float32),
    ))
    return x, ext.label.to_numpy(int), canonical, scaffold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path, help="Official TRAIN_inhibition CSV")
    ap.add_argument("--blind", required=True, type=Path, help="Structure-only blinded CSV")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--external-binary", type=Path, help="Curated binary-only ChEMBL/PubChem table")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train)
    blind = pd.read_csv(args.blind)
    if not {"Molecule_Name", "SMILES"}.issubset(train) or not {"Molecule_Name", "SMILES"}.issubset(blind):
        raise ValueError("Missing molecule identifiers")
    both = pd.concat([train[["SMILES"]], blind[["SMILES"]]], ignore_index=True)
    x, canonical, scaffold = build(both.SMILES, "regression")
    n = len(train)
    ext = external_data(args.external_binary) if args.external_binary else None
    if ext:
        # The curated historical source was also screened against all official
        # and blinded canonical structures before any nested fold fitting.
        safe = ~np.isin(ext[2], canonical)
        ext = tuple(part[safe] for part in ext)
    external_models = {}

    def aux_model(excluded_c, excluded_s):
        xe, ye, ce, se = ext
        keep = ~np.isin(ce, list(excluded_c)) & ~np.isin(se, list(excluded_s))
        if len(np.unique(ye[keep])) != 2:
            raise ValueError("External exclusion removed a class")
        model = classifier().fit(xe[keep], ye[keep])
        return model

    def aux_predict(model, query, endpoint_id):
        p = []
        for source_id in (0, 1):
            xx = np.column_stack((query,
                                  np.full(len(query), endpoint_id, np.float32),
                                  np.full(len(query), source_id, np.float32)))
            p.append(model.predict_proba(xx)[:, 1])
        return np.column_stack(p).astype(np.float32)

    oof, test = [], blind[["Molecule_Name", "SMILES"]].copy()
    for endpoint_id, ep in enumerate(ENDPOINTS):
        valid = train[ep].notna().to_numpy()
        rows = np.flatnonzero(valid)
        y = train.loc[valid, ep].to_numpy(float)
        xf, cf, sf = x[rows], canonical[rows], scaffold[rows]
        ff = folds(sf)
        pred = np.full(len(rows), np.nan)
        for h in range(5):
            fit = ff != h
            held_c, held_s = set(cf[~fit]), set(sf[~fit])
            if ext:
                meta_fit = np.empty((fit.sum(), 2), np.float32)
                fit_positions = np.flatnonzero(fit)
                for j in range(5):
                    if j == h:
                        continue
                    own = ff[fit_positions] == j
                    if not own.any():
                        continue
                    banned_c = held_c | set(cf[ff == j])
                    banned_s = held_s | set(sf[ff == j])
                    m = aux_model(banned_c, banned_s)
                    meta_fit[own] = aux_predict(m, xf[fit_positions[own]], endpoint_id)
                held_aux = aux_model(held_c, held_s)
                xfit = np.column_stack((xf[fit], meta_fit))
                xval = np.column_stack((xf[~fit], aux_predict(held_aux, xf[~fit], endpoint_id)))
            else:
                xfit, xval = xf[fit], xf[~fit]
            m = regressor().fit(xfit, y[fit])
            pred[~fit] = m.predict(xval)
        if ext:
            meta_all = np.empty((len(rows), 2), np.float32)
            for j in range(5):
                own = ff == j
                m = aux_model(set(cf[own]), set(sf[own]))
                meta_all[own] = aux_predict(m, xf[own], endpoint_id)
            # Deployment classifier excludes blinded chemistry, including scaffolds.
            deploy_aux = aux_model(set(canonical[n:]), set(scaffold[n:]))
            external_models[ep] = deploy_aux
            xall = np.column_stack((xf, meta_all))
            xtest = np.column_stack((x[n:], aux_predict(deploy_aux, x[n:], endpoint_id)))
        else:
            xall, xtest = xf, x[n:]
        final = regressor().fit(xall, y)
        joblib.dump(final, args.out / f"{ep}.joblib")
        test[ep] = final.predict(xtest)
        low, high = ep + "_conf_low", ep + "_conf_high"
        oof.append(pd.DataFrame({
            "endpoint": ep, "Molecule_Name": train.Molecule_Name.iloc[rows].to_numpy(),
            "SMILES": train.SMILES.iloc[rows].to_numpy(), "fold": ff,
            "target": y, "conf_low": train[low].iloc[rows].to_numpy(),
            "conf_high": train[high].iloc[rows].to_numpy(), "prediction": pred,
        }))
    if ext:
        joblib.dump(external_models, args.out / "external_binary_models.joblib")
    pd.concat(oof, ignore_index=True).to_csv(args.out / "regression_oof.csv", index=False)
    test.to_csv(args.out / "regression_test.csv", index=False)
    (args.out / "manifest.json").write_text(json.dumps({
        "kind": "standalone_member", "external_binary": bool(ext),
        "continuous_external_labels_read": False, "five_scaffold_folds": True,
        "note": "This does not reconstruct the historical multi-version ensemble.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
