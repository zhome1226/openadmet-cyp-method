"""Apply the frozen reg_v32 mild transform to an input prediction table."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = {
    "CYP1A2_pIC50_direct_inhibition": (1.3914730515, -0.4196983000),
    "CYP2D6_pIC50_direct_inhibition": (1.4745515579, -0.6625896795),
}


def apply(frame):
    out = frame.copy()
    for ep, (scale, shift) in PARAMS.items():
        p = out[ep].to_numpy(float)
        if not np.isfinite(p).all() or scale <= 0:
            raise ValueError(ep)
        calibrated = p.mean() + scale * (p - p.mean()) + shift
        if np.any(np.diff(calibrated[np.argsort(p)]) < -1e-10):
            raise AssertionError("Rank was not preserved")
        out[ep] = calibrated
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    apply(pd.read_csv(args.predictions)).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
