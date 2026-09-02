from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .calibration import create_calibration_curve


def create_summary_report(
    probs: Mapping[str, np.ndarray],
    reals: np.ndarray,
    treats: np.ndarray | None = None,
    intervention: object | None = None,
    weights: np.ndarray | None = None,
    *,
    output_file: str | Path | None = None,
    **kwargs,
):
    """Create a summary report containing only calibration."""
    calibration = create_calibration_curve(
        probs=probs,
        reals=reals,
        treats=treats,
        intervention=intervention,
        weights=weights,
        **kwargs,
    )
    calibration.update_layout(title="Calibration")

    if output_file is not None:
        calibration.write_html(str(output_file), include_plotlyjs=True, full_html=True)

    return calibration
