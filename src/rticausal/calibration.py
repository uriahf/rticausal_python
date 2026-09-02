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
    groups: int | None = None,
    calibration_type: str = "discrete",
    size: int = 600,
    color_values: list[str] | None = None,
):
    """Create factual or intervention-specific calibration curves.

    Factual calibration delegates directly to rtichoke. Intervention
    calibration reproduces the subgroup-coordinate semantics of
    ``ipeval::ip_score(..., metrics="calplot")``: predictions are sorted and
    split into equally sized rank groups; x is the unweighted mean prediction
    among all subjects and y is the weighted mean outcome among subjects
    observed under ``intervention``.
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
        groups=groups,
        size=size,
        color_values=color_values,
    )
    return _create_plotly_curve_from_calibration_curve_list(
        prepared, calibration_type="discrete"
    )


def _r_cut_groups(n: int, groups: int) -> np.ndarray:
    """Match R cut(seq_len(n), breaks=groups, labels=FALSE)."""
    if groups == 1:
        return np.ones(n, dtype=np.int64)

    positions = np.arange(1, n + 1, dtype=float)
    edges = np.linspace(1.0, float(n), groups + 1)
    dx = float(n - 1)
    edges[0] -= dx / 1000.0
    edges[-1] += dx / 1000.0
    return np.searchsorted(edges, positions, side="left").astype(np.int64)


def _ipeval_calplot_rows(
    *,
    probs: np.ndarray,
    reals: np.ndarray,
    pseudo_i: np.ndarray,
    weights: np.ndarray,
    groups: int = 8,
    reference_group: str = "model",
) -> pl.DataFrame:
    p = np.asarray(probs, dtype=float).reshape(-1)
    y = np.asarray(reals).reshape(-1)
    pseudo = np.asarray(pseudo_i, dtype=bool).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)

    n_breaks = min(groups, np.unique(p).size)
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    y_sorted = y[order]
    pseudo_sorted = pseudo[order]
    w_sorted = w[order]
    group = _r_cut_groups(p.size, n_breaks)

    pred_values: list[float] = []
    obs_values: list[float] = []
    for group_id in range(1, n_breaks + 1):
        in_group = group == group_id
        pred_values.append(float(np.mean(p_sorted[in_group])))

        observed = in_group & pseudo_sorted
        if not np.any(observed):
            obs_values.append(float("nan"))
            continue

        group_weights = w_sorted[observed]
        weight_sum = float(np.sum(group_weights))
        if weight_sum == 0.0:
            obs_values.append(float("nan"))
        else:
            obs_values.append(
                float(np.sum(y_sorted[observed] * group_weights) / weight_sum)
            )

    return pl.DataFrame(
        {
            "reference_group": [reference_group] * n_breaks,
            "decile": np.arange(1, n_breaks + 1),
            "x": pred_values,
            "y": obs_values,
        }
    )


def _prepare_intervention_calibration(
    *,
    probs: dict[str, np.ndarray],
    reals: np.ndarray,
    treats: np.ndarray,
    intervention: object,
    weights: np.ndarray | None,
    groups: int | None = None,
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
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("reals must be binary (0/1).")

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError(
                "weights must be finite, non-negative, and the same length as reals."
            )

    if groups is None:
        groups = 8
    if not isinstance(groups, (int, np.integer)) or groups < 1:
        raise ValueError("groups must be a positive integer.")

    treatment_labels = a.astype(str)
    intervention_label = str(intervention)
    if intervention_label not in set(np.unique(treatment_labels)):
        raise ValueError("intervention must match an observed treatment level in treats.")
    pseudo_i = treatment_labels == intervention_label

    prepare_kwargs = {"size": size}
    if color_values is not None:
        prepare_kwargs["color_values"] = color_values
    prepared = _create_calibration_curve_list(dict(probs), y, **prepare_kwargs)

    rows = [
        _ipeval_calplot_rows(
            probs=np.asarray(raw_probs),
            reals=y,
            pseudo_i=pseudo_i,
            weights=w,
            groups=int(groups),
            reference_group=series,
        )
        for series, raw_probs in probs.items()
    ]
    calibration_rows = pl.concat(rows).with_columns(
        pl.format(
            "Predicted: {}<br>Observed: {}",
            pl.col("x").round(3),
            pl.col("y").round(3),
        ).alias("text")
    )
    prepared["deciles_dat"] = calibration_rows

    finite = np.concatenate(
        [
            calibration_rows.get_column("x").to_numpy(),
            calibration_rows.get_column("y").to_numpy(),
        ]
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        limits = [0.0, 1.0]
    else:
        lower = max(0.0, float(np.min(finite)))
        upper = float(np.max(finite))
        if upper == lower:
            limits = [max(0.0, lower - 0.05), min(1.0, upper + 0.05)]
        else:
            pad = (upper - lower) * 0.05
            limits = [lower - pad, upper + pad]
    prepared["axes_ranges"] = {"xaxis": limits, "yaxis": limits}
    return prepared
