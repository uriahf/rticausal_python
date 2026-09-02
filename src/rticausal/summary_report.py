from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .calibration import create_calibration_curve


def create_summary_report(
    probs: Mapping[str, np.ndarray],
    reals: np.ndarray,
    treats: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    *,
    output_file: str | Path | None = None,
    **kwargs,
):
    """Create a summary report containing only calibration.

    The returned object is the same Plotly calibration figure used by
    ``create_calibration_curve``. When ``output_file`` is supplied, the figure
    is also written as a self-contained HTML document.
    """
    calibration = create_calibration_curve(
        probs=probs,
        reals=reals,
        treats=treats,
        weights=weights,
        **kwargs,
    )
    calibration.update_layout(title="Calibration")

    if output_file is not None:
        calibration.write_html(str(output_file), include_plotlyjs=True, full_html=True)

    return calibration
