"""Conditional inversion of published Live aggregate metrics (no hidden labels).

The model family is an assumption. Both signs of the unknown mean shift are
reported; the observed metrics do not identify the sign.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import rankdata

MIXTURES = {
    "normal": ((1.0, 1.0),),
    "moderate_tails": ((.8, .75), (.2, math.sqrt(2.75))),
    "heavy_tails": ((.9, .5), (.1, math.sqrt(7.75))),
}


def positive_normal(mean, sd):
    z = mean / sd
    return sd * np.exp(-.5 * z * z) / math.sqrt(2 * math.pi) + mean * ndtr(z)


def absolute_normal(mean, sd):
    return positive_normal(mean, sd) + positive_normal(-mean, sd)


def expected_spearman(x, r, mixture):
    z = (x - x.mean()) / x.std(ddof=0)
    u = (rankdata(x, method="average") - .5) / len(x)
    delta = z[:, None] - z[None, :]
    cdf = np.zeros(len(x))
    for wk, sk in mixture:
        for wl, sl in mixture:
            cdf += wk * wl * ndtr(
                r * delta / math.sqrt((1 - r * r) * (sk * sk + sl * sl))
            ).mean(axis=1)
    return float(np.mean((u - .5) * (cdf - .5)) / math.sqrt(u.var() / 12))


def model_mae(x, b, r, d, mixture):
    a = x.std(ddof=0)
    z = (x - x.mean()) / a
    mean_error = d + (r * b - a) * z
    sd = b * math.sqrt(1 - r * r)
    return float(sum(w * absolute_normal(mean_error, sd * k).mean()
                     for w, k in mixture))


def solve(x, mae, r2, spearman, sign, mixture):
    a = float(x.std(ddof=0))
    r = brentq(lambda q: expected_spearman(x, q, mixture) - spearman,
               0.0, .999999, xtol=1e-12)

    def candidate(b):
        d2 = 2 * r * a * b - a * a - r2 * b * b
        if d2 < 0:
            return None
        d = sign * math.sqrt(d2)
        return d, model_mae(x, b, r, d, mixture) - mae

    grid = np.geomspace(max(.01, mae / math.sqrt(1 - r2) * .99999), 30., 1800)
    valid = [(b, *result) for b in grid if (result := candidate(b)) is not None]
    roots = []
    for left, right in zip(valid[:-1], valid[1:]):
        if left[2] * right[2] < 0:
            b = brentq(lambda value: candidate(value)[1], left[0], right[0], xtol=1e-12)
            roots.append((float(b), float(candidate(b)[0])))
    if len(roots) != 1:
        raise ValueError(f"Expected one solution, got {roots}")
    b, d = roots[0]
    full_scale = max(1.0, r * b / a)
    return {"pearson_model": r, "target_sd_model": b, "mean_shift_model": d,
            "conditional_scale": full_scale,
            "mild_scale": 1 + (full_scale - 1) / 2,
            "mild_shift": d / 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Uncalibrated blinded prediction CSV, no outcomes")
    ap.add_argument("--live-metrics", type=Path, required=True,
                    help="JSON: per endpoint {mae,r2,spearman}; published aggregates only")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frame = pd.read_csv(args.predictions)
    metrics = json.loads(args.live_metrics.read_text())
    result = {"warning": "Conditional scenarios, not confidence bounds or hidden-label estimates"}
    for ep, m in metrics.items():
        x = frame[ep].to_numpy(float)
        result[ep] = {
            name: {str(sign): solve(x, m["mae"], m["r2"], m["spearman"], sign, mixture)
                   for sign in (-1, 1)}
            for name, mixture in MIXTURES.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
