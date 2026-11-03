"""Standalone CYP2D6 TDI members with nested official-screen probabilities.

The historical v14 base scores/calls are separate inputs to tdi_rules.py.
"""
import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import build


def screen_model():
    return lgb.LGBMClassifier(
        n_estimators=140, learning_rate=.05, num_leaves=15,
        min_child_samples=30, random_state=42, n_jobs=2,
        verbosity=-1, deterministic=True, force_col_wise=True)


def tdi_model():
    return lgb.LGBMClassifier(
        n_estimators=260, learning_rate=.035, num_leaves=15,
        min_child_samples=24, max_bin=63, colsample_bytree=.8,
        reg_lambda=5, class_weight={0: 1, 1: 4},
        random_state=42, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdi-train", type=Path, required=True)
    ap.add_argument("--screen-train", type=Path, required=True)
    ap.add_argument("--blind", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tdi = pd.read_csv(args.tdi_train)
    tdi = tdi[tdi.CYP2D6_is_TDI.notna()].reset_index(drop=True)
    screen = pd.read_csv(args.screen_train, usecols=["SMILES", "enzyme", "log2fc_estimate"])
    screen = screen[(screen.enzyme == "CYP2D6") & screen.log2fc_estimate.notna()].reset_index(drop=True)
    blind = pd.read_csv(args.blind)
    all_x, all_c, all_s = build(pd.concat([tdi.SMILES, blind.SMILES]), "tdi")
    n = len(tdi)
    x, xt = all_x[:n], all_x[n:]
    canon, scaff = all_c[:n], all_s[:n]
    sx, sc, ss = build(screen.SMILES, "tdi")
    sx = sx[:, :2580]  # Historical screen classifiers used fixed structure features only.
    sy = np.column_stack((screen.log2fc_estimate.to_numpy() < -1,
                          screen.log2fc_estimate.to_numpy() < -2)).astype(int)
    if len(set(scaff)) < 5:
        raise ValueError("Five scaffold groups required")
    fold = np.empty(n, np.int8)
    for h, (_, val) in enumerate(GroupKFold(5).split(x, groups=scaff)):
        fold[val] = h
    labels = tdi.CYP2D6_is_TDI.astype(int).to_numpy()

    def fit_screen(banned_c, banned_s):
        keep = ~np.isin(sc, list(banned_c)) & ~np.isin(ss, list(banned_s))
        if not keep.any():
            raise ValueError("No screen rows after exclusion")
        models = []
        for k in range(2):
            if len(np.unique(sy[keep, k])) != 2:
                raise ValueError("Screen class missing after exclusion")
            models.append(screen_model().fit(sx[keep], sy[keep, k]))
        return models

    def screen_pred(models, query):
        p = np.column_stack([m.predict_proba(query[:, :2580])[:, 1] for m in models])
        return np.column_stack((p, p[:, 1] - p[:, 0])).astype(np.float32)

    oof_structure = np.full(n, np.nan)
    oof_meta = np.full(n, np.nan)
    for h in range(5):
        fit, held = fold != h, fold == h
        banned_c, banned_s = set(canon[held]), set(scaff[held])
        meta_fit = np.empty((fit.sum(), 3), np.float32)
        fit_idx = np.flatnonzero(fit)
        for j in range(5):
            if j == h:
                continue
            own = fold[fit_idx] == j
            if own.any():
                models = fit_screen(banned_c | set(canon[fold == j]),
                                    banned_s | set(scaff[fold == j]))
                meta_fit[own] = screen_pred(models, x[fit_idx[own]])
        held_models = fit_screen(banned_c, banned_s)
        meta_held = screen_pred(held_models, x[held])
        structure = tdi_model().fit(x[fit], labels[fit])
        meta = tdi_model().fit(np.column_stack((x[fit], meta_fit)), labels[fit])
        oof_structure[held] = structure.predict_proba(x[held])[:, 1]
        oof_meta[held] = meta.predict_proba(np.column_stack((x[held], meta_held)))[:, 1]
    # Full-data deployment, with screen models for each training fold excluding that fold.
    meta_all = np.empty((n, 3), np.float32)
    for j in range(5):
        own = fold == j
        models = fit_screen(set(canon[own]), set(scaff[own]))
        meta_all[own] = screen_pred(models, x[own])
    deploy_screen = fit_screen(set(all_c[n:]), set(all_s[n:]))
    deploy_meta = screen_pred(deploy_screen, xt)
    structure = tdi_model().fit(x, labels)
    meta = tdi_model().fit(np.column_stack((x, meta_all)), labels)
    joblib.dump({"screen": deploy_screen, "structure": structure, "meta": meta},
                args.out / "tdi_models.joblib")
    pd.DataFrame({
        "Molecule_Name": tdi.Molecule_Name, "SMILES": tdi.SMILES, "fold": fold,
        "target": labels, "structure_score": oof_structure, "meta_score": oof_meta,
    }).to_csv(args.out / "tdi_oof_scores.csv", index=False)
    pd.DataFrame({
        "Molecule_Name": blind.Molecule_Name, "SMILES": blind.SMILES,
        "structure_score": structure.predict_proba(xt)[:, 1],
        "meta_score": meta.predict_proba(np.column_stack((xt, deploy_meta)))[:, 1],
    }).to_csv(args.out / "tdi_test_scores.csv", index=False)
    (args.out / "tdi_manifest.json").write_text(json.dumps({
        "kind": "standalone_tdi_members", "uses_external_samples": False,
        "screen_labels": ["log2fc_estimate < -1", "log2fc_estimate < -2"],
        "nested_screen_exclusion": True,
        "note": "Historical v22 also used stricter 12-identity exclusions and nested v14 base calls.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
