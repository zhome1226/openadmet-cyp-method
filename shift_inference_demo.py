"""Synthetic, file-free illustration of aggregate-metric shift inference.

Run: python shift_inference_demo.py

This uses no competition records or coefficients. The published-like aggregates
are calculated from a declared toy data-generating model; the solver sees only
those aggregates and the synthetic prediction distribution, not target rows.
"""

import json
import math

import numpy as np

from shift_inference import MIXTURES, expected_spearman, model_mae, solve


def main():
    rng = np.random.default_rng(20261103)
    predictions = np.r_[rng.normal(5.0, 0.45, 80), rng.normal(5.5, 0.25, 20)]
    prediction_mean = float(predictions.mean())
    prediction_sd = float(predictions.std(ddof=0))

    # Toy population model: y = mean(p) + d + r*b*z + b*sqrt(1-r²)*epsilon.
    # We use it to make public-like endpoint aggregates; it is not a claim
    # about the real blinded targets or the official scorer.
    target_sd, target_shift, correlation = 1.20, -0.60, 0.72
    normal = MIXTURES["normal"]
    aggregate = {
        "mae": model_mae(predictions, target_sd, correlation, target_shift, normal),
        "r2": 1 - (target_shift**2 + prediction_sd**2 + target_sd**2
                   - 2 * correlation * prediction_sd * target_sd) / target_sd**2,
        "spearman": expected_spearman(predictions, correlation, normal),
    }

    # This is the inference boundary: the solver receives aggregate metrics
    # and prediction values only. It cannot see the generating parameters.
    branches = {
        "negative": solve(predictions, **aggregate, sign=-1, mixture=normal),
        "positive": solve(predictions, **aggregate, sign=1, mixture=normal),
    }
    chosen = branches["negative"]  # A declared choice, not identified by metrics.
    scale, shift = chosen["mild_scale"], chosen["mild_shift"]
    adjusted = prediction_mean + scale * (predictions - prediction_mean) + shift

    # Synthetic labels exist only to check this demo's chosen branch. They
    # are never passed into solve(), mirroring the no-hidden-label boundary.
    residuals = rng.normal(size=len(predictions))
    synthetic_targets = (prediction_mean + target_shift
                         + correlation * target_sd
                         * (predictions - prediction_mean) / prediction_sd
                         + target_sd * math.sqrt(1 - correlation**2) * residuals)
    assert np.all(np.diff(adjusted[np.argsort(predictions)]) >= 0)
    result = {
        "data": "synthetic only; no challenge inputs",
        "public_like_aggregate_metrics": aggregate,
        "compatible_mean_shift_branches": {
            name: {"shift": value["mean_shift_model"],
                   "full_scale": value["conditional_scale"]}
            for name, value in branches.items()
        },
        "declared_negative_branch_mild_action": {"scale": scale, "shift": shift},
        "synthetic_check_only": {
            "mae_before": float(np.mean(np.abs(synthetic_targets - predictions))),
            "mae_after": float(np.mean(np.abs(synthetic_targets - adjusted))),
            "rank_preserved": True,
        },
        "interpretation": "Both shift signs fit the aggregates; direction needs an external assumption.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
