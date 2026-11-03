"""Inference for standalone TDI member models; decisions require tdi_rules.py."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--structures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frame = pd.read_csv(args.structures)
    x, _, _ = build(frame.SMILES, "tdi")
    models = joblib.load(args.models)
    p = np.column_stack([m.predict_proba(x[:, :2580])[:, 1] for m in models["screen"]])
    sx = np.column_stack((x, p, p[:, 1] - p[:, 0]))
    out = frame[["Molecule_Name", "SMILES"]].copy()
    out["structure_score"] = models["structure"].predict_proba(x)[:, 1]
    out["meta_score"] = models["meta"].predict_proba(sx)[:, 1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
