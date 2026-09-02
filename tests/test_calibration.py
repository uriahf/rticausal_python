import numpy as np

from rticausal import create_calibration_curve, create_summary_report
from rticausal.calibration import (
    _ipeval_calplot_rows,
    _prepare_intervention_calibration,
    _r_cut_groups,
)


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
        groups=None,
        size=600,
        color_values=None,
    )

    groups = set(prepared["deciles_dat"].get_column("reference_group").to_list())
    assert groups == {"model_a", "model_b"}
    assert prepared["deciles_dat"].height == 16


def test_r_cut_groups_matches_ipeval_rank_grouping():
    np.testing.assert_array_equal(
        _r_cut_groups(16, 4),
        np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]),
    )


def test_ipeval_calplot_fixture_matches_reference_coordinates():
    probs = np.array(
        [
            0.80,
            0.10,
            0.70,
            0.20,
            0.60,
            0.30,
            0.50,
            0.40,
            0.75,
            0.15,
            0.65,
            0.25,
            0.55,
            0.35,
            0.45,
            0.05,
        ]
    )
    reals = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    treats = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    weights = np.array(
        [
            1.0,
            1.2,
            0.8,
            1.5,
            1.1,
            0.9,
            1.3,
            1.0,
            1.4,
            0.7,
            1.6,
            1.0,
            0.6,
            1.8,
            1.2,
            0.5,
        ]
    )

    rows = _ipeval_calplot_rows(
        probs=probs,
        reals=reals,
        pseudo_i=treats == 1,
        weights=weights,
        groups=4,
    )

    np.testing.assert_allclose(
        rows.get_column("x").to_numpy(), [0.125, 0.325, 0.525, 0.725]
    )

    # Hand-computed from ipeval's sorted-rank groups and weighted.mean rule.
    expected_obs = np.array(
        [
            (0.5 * 0 + 0.7 * 1 + 1.5 * 1) / (0.5 + 0.7 + 1.5),
            (1.0 * 0 + 1.8 * 0) / (1.0 + 1.8),
            (1.2 * 1 + 1.3 * 0 + 1.1 * 0) / (1.2 + 1.3 + 1.1),
            (0.8 * 1 + 1.4 * 1 + 1.0 * 1) / (0.8 + 1.4 + 1.0),
        ]
    )
    np.testing.assert_allclose(rows.get_column("y").to_numpy(), expected_obs)


def test_group_count_is_capped_by_unique_predictions_like_ipeval():
    rows = _ipeval_calplot_rows(
        probs=np.repeat(np.array([0.1, 0.5, 0.9]), 4),
        reals=np.tile(np.array([0, 1]), 6),
        pseudo_i=np.ones(12, dtype=bool),
        weights=np.ones(12),
        groups=8,
    )
    assert rows.height == 3
    np.testing.assert_allclose(rows.get_column("x").to_numpy(), [0.1, 0.5, 0.9])


def test_intervention_must_match_observed_treatment_level():
    try:
        _prepare_intervention_calibration(
            probs={"model": np.linspace(0.1, 0.9, 20)},
            reals=np.tile(np.array([0, 1]), 10),
            treats=np.tile(np.array([0, 1]), 10),
            intervention=2,
            weights=None,
            groups=None,
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
