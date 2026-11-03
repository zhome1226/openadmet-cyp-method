"""Frozen positive-only TDI rule with inner-label and call-rate guards.

Input scores must already be nested out of fold. For exact historical v22
evaluation, use its stored inner/outer member scores and v14 base calls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import matthews_corrcoef, recall_score

GRID = (.8, .85, .9, .925, .95, .975, .99)
FLOORS = (0., .5, .65, .75)


def ranks(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite member score")
    return rankdata(values, method="average") / len(values)


def dual_rank(frame):
    return (ranks(frame.meta_score) + ranks(frame.structure_score)) / 2


def bool_values(column):
    values = column.astype(str).str.lower()
    if not values.isin(("true", "false", "1", "0")).all():
        raise ValueError("Call column must be boolean")
    return values.isin(("true", "1")).to_numpy()


def select(inner: pd.DataFrame, outer: pd.DataFrame, budget=.05):
    """Select using only inner labels; return frozen rule and outer calls."""
    y = inner.target.to_numpy(int)
    bi = bool_values(inner.base_call)
    bo = bool_values(outer.base_call)
    si, so = dual_rank(inner), dual_rank(outer)
    base_inner_mcc = matthews_corrcoef(y, bi)
    base_inner_recall = recall_score(y, bi)
    candidates = []
    for threshold in GRID:
        for floor in FLOORS:
            add_inner = (~bi) & (si >= threshold) & (inner.base_score.to_numpy(float) >= floor)
            add_outer = (~bo) & (so >= threshold) & (outer.base_score.to_numpy(float) >= floor)
            if not add_inner.any() or add_inner.mean() > budget + 1e-12:
                continue
            calls_inner, calls_outer = bi | add_inner, bo | add_outer
            if calls_outer.mean() > 1.5 * y.mean() + 1e-12:
                continue
            mcc = matthews_corrcoef(y, calls_inner)
            recall = recall_score(y, calls_inner)
            if recall + 1e-12 < base_inner_recall or mcc <= base_inner_mcc + 1e-12:
                continue
            candidates.append((mcc, recall, -int(add_inner.sum()), threshold, floor, calls_outer))
    if not candidates:
        return {"action": "baseline_fallback", "inner_MCC": float(base_inner_mcc),
                "inner_recall": float(base_inner_recall), "added_batch": 0}, bo
    best = max(candidates, key=lambda x: x[:5])
    calls = best[5]
    if not np.all(calls[bo]):
        raise AssertionError("Base positive was removed")
    return {
        "action": "selective_add", "inner_MCC": float(best[0]),
        "inner_recall": float(best[1]), "threshold": best[3],
        "base_score_floor": best[4], "added_batch": int((calls & ~bo).sum()),
    }, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", type=Path, required=True,
                    help="Inner-fold nested scores with target and base calls")
    ap.add_argument("--outer", type=Path, required=True,
                    help="Held-out/deployment scores with base calls")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    args = ap.parse_args()
    inner, outer = pd.read_csv(args.inner), pd.read_csv(args.outer)
    needed = {"meta_score", "structure_score", "base_score", "base_call"}
    if not needed | {"target"} <= set(inner) or not needed <= set(outer):
        raise ValueError("Missing score/label columns")
    detail, calls = select(inner, outer)
    ratio = float(calls.mean() / inner.target.mean())
    detail["deployment_prevalence_ratio"] = ratio
    detail["deployment_guard_passed"] = bool(.5 <= ratio <= 1.5)
    if not detail["deployment_guard_passed"]:
        calls = bool_values(outer.base_call)
        detail["action"] = "prevalence_guard_fallback"
    out = outer[["Molecule_Name", "SMILES"]].copy()
    out["CYP2D6_is_TDI"] = calls.astype(bool)
    if "CYP3A4_is_TDI" in outer:
        out["CYP3A4_is_TDI"] = bool_values(outer.CYP3A4_is_TDI)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    args.audit.write_text(json.dumps(detail, indent=2) + "\n")


if __name__ == "__main__":
    main()
