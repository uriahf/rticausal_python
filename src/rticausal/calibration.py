from __future__ import annotations

from typing import Mapping

import numpy as np
import polars as pl

from rtichoke.calibration.calibration import (
    _create_calibration_curve_list,
    _create_plotly_curve_from_calibration_curve_list,
    create_calibration_curve as _rtichoke_create_calibration_curve,
)


def create_calibration_curve(
    probs: Mapping[str, np.ndarray],
    reals: np.ndarray,
    treats: np.ndarray | None = None,
    intervention: object | None = None,
    weights: np.ndarray | None = None,
    *,
    calibration_type: str = "discrete",
    size: int = 600,
    color_values: list[str] | None = None,
):
    """Create factual or intervention-specific calibration curves.

    When ``treats`` is omitted, this delegates directly to rtichoke. In
    intervention mode, ``treats`` contains observed treatment assignments and
    ``intervention`` identifies the treatment level whose counterfactual risks
    are represented by ``probs``. Keys in ``probs`` remain model/series names.

    ``weights`` are caller-supplied identification/design weights. rticausal
    consumes them but does not fit treatment models.
    """
    kwargs = {"calibration_type": calibration_type, "size": size}
    if color_values is not None:
        kwargs["color_values"] = color_values

    if treats is None:
        if intervention is not None or weights is not None:
            raise ValueError("intervention and weights require treats.")
        return _rtichoke_create_calibration_curve(dict(probs), reals, **kwargs)

    if intervention is None:
        raise ValueError("intervention is required when treats is supplied.")
    if calibration_type != "discrete":
        raise ValueError(
            "Intervention calibration currently supports only "
            "calibration_type='discrete'."
        )

    prepared = _prepare_intervention_calibration(
        probs=dict(probs),
        reals=np.asarray(reals),
        treats=np.asarray(treats),
        intervention=intervention,
        weights=None if weights is None else np.asarray(weights),
        size=size,
        color_values=color_values,
    )
    return _create_plotly_curve_from_calibration_curve_list(
        prepared, calibration_type="discrete"
    )


def _prepare_intervention_calibration(
    *,
    probs: dict[str, np.ndarray],
    reals: np.ndarray,
    treats: np.ndarray,
    intervention: object,
    weights: np.ndarray | None,
    size: int,
    color_values: list[str] | None,
) -> dict:
    if not probs or any(not str(key) for key in probs):
        raise ValueError("probs must be a non-empty mapping with model/series names.")

    y = np.asarray(reals).reshape(-1)
    a = np.asarray(treats).reshape(-1)
    n = y.shape[0]
    if a.shape[0] != n or any(
        np.asarray(p).reshape(-1).shape[0] != n for p in probs.values()
    ):
        raise ValueError("probs, reals, and treats must describe the same observations.")

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError(
                "weights must be finite, non-negative, and the same length as reals."
            )

    treatment_labels = a.astype(str)
    intervention_label = str(intervention)
    if intervention_label not in set(np.unique(treatment_labels)):
        raise ValueError("intervention must match an observed treatment level in treats.")

    prepare_kwargs = {"size": size}
    if color_values is not None:
        prepare_kwargs["color_values"] = color_values
    prepared = _create_calibration_curve_list(dict(probs), y, **prepare_kwargs)

    selected_weight = w * (treatment_labels == intervention_label)
    adjusted_frames: list[pl.DataFrame] = []
    for series, raw_probs in probs.items():
        p = np.asarray(raw_probs, dtype=float).reshape(-1)
        if np.unique(p).size == 1:
            decile = np.ones(n, dtype=int)
        else:
            order = np.argsort(p, kind="stable")
            ranks = np.empty(n, dtype=int)
            ranks[order] = np.arange(1, n + 1)
            decile = ((ranks - 1) * 10 // n) + 1

        frame = (
            pl.DataFrame(
                {
                    "reference_group": [series] * n,
                    "decile": decile,
                    "selected_weight": selected_weight,
                    "weighted_event": selected_weight * y,
                }
            )
            .group_by(["reference_group", "decile"])
            .agg(
                pl.col("selected_weight").sum().alias("selected_weight"),
                pl.col("weighted_event").sum().alias("weighted_event"),
            )
            .with_columns(
                (pl.col("weighted_event") / pl.col("selected_weight")).alias("y")
            )
            .select("reference_group", "decile", "y")
        )
        adjusted_frames.append(frame)

    adjusted = pl.concat(adjusted_frames)
    if adjusted.filter(~pl.col("y").is_finite()).height:
        raise ValueError(
            "Each prediction bin must contain positive treatment weight for the intervention."
        )

    deciles = prepared["deciles_dat"]
    prepared["deciles_dat"] = (
        deciles.drop("y")
        .join(adjusted, on=["reference_group", "decile"], how="left")
        .with_columns(
            pl.format(
                "Predicted: {}<br>Observed: {}",
                pl.col("x").round(3),
                pl.col("y").round(3),
            ).alias("text")
        )
    )
    return prepared
