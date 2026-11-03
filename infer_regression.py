"""Inference for train_regression.py's fitted standalone member."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build
from train_regression import ENDPOINTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, type=Path)
    ap.add_argument("--structures", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    frame = pd.read_csv(args.structures)
    x, _, _ = build(frame.SMILES, "regression")
    out = frame[["Molecule_Name", "SMILES"]].copy()
    aux_path = args.models / "external_binary_models.joblib"
    aux = joblib.load(aux_path) if aux_path.exists() else {}
    for ei, ep in enumerate(ENDPOINTS):
        xx = x
        if ep in aux:
            model = aux[ep]
            probs = []
            for source_id in (0, 1):
                query = np.column_stack((
                    x, np.full(len(x), ei, np.float32),
                    np.full(len(x), source_id, np.float32),
                ))
                probs.append(model.predict_proba(query)[:, 1])
            xx = np.column_stack((x, *probs))
        reg = joblib.load(args.models / f"{ep}.joblib")
        out[ep] = reg.predict(xx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
