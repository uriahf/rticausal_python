import numpy as np

from rticausal import create_calibration_curve, create_summary_report
from rticausal.calibration import _prepare_intervention_calibration


def test_factual_calibration_delegates_to_rtichoke():
    figure = create_calibration_curve(
        probs={"model": np.array([0.1, 0.2, 0.8, 0.9])},
        reals=np.array([0, 0, 1, 1]),
    )
    assert figure.data


def test_intervention_calibration_separates_treatment_from_model_identity():
    probs = {
        "model_a": np.linspace(0.05, 0.95, 40),
        "model_b": np.linspace(0.10, 0.90, 40),
    }
    reals = np.tile(np.array([0, 1]), 20)
    treats = np.tile(np.array([0, 0, 1, 1]), 10)

    prepared = _prepare_intervention_calibration(
        probs=probs,
        reals=reals,
        treats=treats,
        intervention=1,
        weights=None,
        size=600,
        color_values=None,
    )

    groups = set(prepared["deciles_dat"].get_column("reference_group").to_list())
    assert groups == {"model_a", "model_b"}
    observed = prepared["deciles_dat"].get_column("y").to_numpy()
    assert np.all(np.isfinite(observed))
    assert np.all((observed >= 0) & (observed <= 1))


def test_intervention_must_match_observed_treatment_level():
    try:
        _prepare_intervention_calibration(
            probs={"model": np.linspace(0.1, 0.9, 20)},
            reals=np.tile(np.array([0, 1]), 10),
            treats=np.tile(np.array([0, 1]), 10),
            intervention=2,
            weights=None,
            size=600,
            color_values=None,
        )
    except ValueError as exc:
        assert "observed treatment level" in str(exc)
    else:
        raise AssertionError("Expected invalid intervention to raise ValueError")


def test_summary_report_contains_only_calibration(tmp_path):
    output_file = tmp_path / "report.html"
    figure = create_summary_report(
        probs={"model": np.array([0.1, 0.2, 0.8, 0.9])},
        reals=np.array([0, 0, 1, 1]),
        output_file=output_file,
    )

    assert output_file.exists()
    assert figure.layout.title.text == "Calibration"
