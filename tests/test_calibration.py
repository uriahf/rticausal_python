import numpy as np

from rticausal import create_calibration_curve, create_summary_report
from rticausal.calibration import _prepare_intervention_calibration


def test_factual_calibration_delegates_to_rtichoke():
    figure = create_calibration_curve(
        probs={"model": np.array([0.1, 0.2, 0.8, 0.9])},
        reals=np.array([0, 0, 1, 1]),
    )
    assert figure.data


def test_intervention_calibration_uses_treatment_specific_outcomes():
    probs = {
        "0": np.linspace(0.05, 0.95, 20),
        "1": np.linspace(0.95, 0.05, 20),
    }
    reals = np.tile(np.array([0, 1]), 10)
    treats = np.tile(np.array([0, 1]), 10)

    prepared = _prepare_intervention_calibration(
        probs=probs,
        reals=reals,
        treats=treats,
        weights=None,
        size=600,
        color_values=None,
    )

    observed = prepared["deciles_dat"].get_column("y").to_numpy()
    assert np.all(np.isfinite(observed))
    assert np.all((observed >= 0) & (observed <= 1))


def test_summary_report_contains_only_calibration(tmp_path):
    output_file = tmp_path / "report.html"
    figure = create_summary_report(
        probs={"model": np.array([0.1, 0.2, 0.8, 0.9])},
        reals=np.array([0, 0, 1, 1]),
        output_file=output_file,
    )

    assert output_file.exists()
    assert figure.layout.title.text == "Calibration"
