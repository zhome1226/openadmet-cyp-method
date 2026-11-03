"""Local interval-censored ST-RAE assumption and fixed shift proxy."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENDPOINTS = [f"CYP{x}_pIC50_direct_inhibition" for x in ("1A2", "2C9", "2D6", "3A4")]


def distance(p, lo, hi):
    return np.maximum(np.maximum(lo - p, p - hi), 0)


def score(y, p, lo, hi, weight=None):
    y, p, lo, hi = map(lambda a: np.asarray(a, float), (y, p, lo, hi))
    if not (np.isfinite(y).all() and np.isfinite(p).all() and
            np.isfinite(lo).all() and np.isfinite(hi).all() and (lo <= hi).all()):
        raise ValueError("Nonfinite input or reversed interval")
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    ref = np.average(y, weights=w)
    denominator = np.sum(w * distance(ref, lo, hi))
    if denominator <= 0:
        raise ValueError("Zero interval-reference denominator")
    return float(np.sum(w * distance(p, lo, hi)) / denominator)


def stress(y, lo, hi, endpoint):
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    if endpoint.endswith("CYP2D6_pIC50_direct_inhibition"):
        center = y.mean()
        weight = np.where(y <= np.quantile(y, .25), 2.0, 1.0)
        return (*(center - .5 + 1.3 * (a - center) for a in (y, lo, hi)), weight)
    weight = np.where(y >= np.quantile(y, .75), 3.0, 1.0)
    return y, lo, hi, weight


def evaluate(frame):
    rows = []
    for ep in ENDPOINTS:
        part = frame[frame.endpoint == ep]
        if part.empty:
            raise ValueError(f"Missing endpoint {ep}")
        y, p, lo, hi = (part[c].to_numpy(float)
                         for c in ("target", "prediction", "conf_low", "conf_high"))
        local = score(y, p, lo, hi)
        ys, ls, hs, w = stress(y, lo, hi, ep)
        rows.append({"endpoint": ep, "local": local,
                     "stress": score(ys, p, ls, hs, w), "rows": len(part)})
    return {"endpoints": rows, "local_macro": float(np.mean([r["local"] for r in rows])),
            "stress_macro": float(np.mean([r["stress"] for r in rows])),
            "metric_note": "Local assumed denominator; not official Live ST-RAE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = evaluate(pd.read_csv(args.oof))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
