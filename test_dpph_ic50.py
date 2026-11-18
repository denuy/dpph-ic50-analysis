"""Tests for dpph_ic50.

Most tests build synthetic absorbances from a curve with a *known* IC50 and check that the
analysis recovers it, so the expected answer never depends on the code under test.
"""

from pathlib import Path

import numpy as np
import pytest

import dpph_ic50 as dp

EXAMPLE = Path(__file__).resolve().parent.parent / "data" / "example_data.csv"
CONC = np.array([10.0, 25.0, 50.0, 100.0, 200.0])


def make_readings(ic50=45.0, hill=1.3, conc=CONC, a_control=0.8, n_reps=3,
                  noise_sd=0.0, seed=0, offset=0.0, top=100.0):
    """Absorbances for a sample whose true response is a Hill curve (0 -> ``top`` %)."""
    rng = np.random.default_rng(seed)
    pct = top / (1.0 + (ic50 / conc) ** hill)
    clean = a_control * (1.0 - pct / 100.0)
    absorb = clean[:, None] + rng.normal(0, noise_sd, (conc.size, n_reps))
    control = a_control + rng.normal(0, noise_sd, n_reps)
    return dp.Readings(conc, absorb + offset, control + offset)


# --- percent scavenging ----------------------------------------------------- #
def test_percent_scavenging_formula():
    assert dp.percent_scavenging(0.8, 0.4) == pytest.approx(50.0)
    assert dp.percent_scavenging(0.8, 0.8) == pytest.approx(0.0)
    assert dp.percent_scavenging(0.8, 0.0) == pytest.approx(100.0)
    np.testing.assert_allclose(dp.percent_scavenging(0.5, [0.4, 0.25]), [20.0, 50.0])


def test_percent_scavenging_rejects_non_positive_control():
    with pytest.raises(ValueError):
        dp.percent_scavenging(0.0, 0.1)


def test_four_pl_is_midpoint_at_ic50():
    assert dp.four_pl(30.0, 0.0, 100.0, 30.0, 1.7) == pytest.approx(50.0)
    assert dp.four_pl(30.0, 10.0, 90.0, 30.0, 0.8) == pytest.approx(50.0)  # midpoint of 10 and 90


# --- recovering a known IC50 ------------------------------------------------ #
@pytest.mark.parametrize("true_ic50, hill", [(45.0, 1.3), (20.0, 0.8), (120.0, 2.0)])
def test_recovers_known_ic50_without_noise(true_ic50, hill):
    result = dp.analyze(make_readings(ic50=true_ic50, hill=hill)).fit
    assert result.status == "ok"
    assert result.ic50 == pytest.approx(true_ic50, rel=1e-3)
    assert result.hill == pytest.approx(hill, rel=1e-3)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


def test_recovers_known_ic50_with_noisy_replicates():
    result = dp.analyze(make_readings(noise_sd=0.012, seed=2026)).fit
    assert result.status == "ok"
    assert result.ic50 == pytest.approx(45.0, rel=0.10)
    assert result.ic50_se > 0
    lo, hi = result.ic50_ci95
    assert lo < result.ic50 < hi


def test_confidence_interval_coverage_is_roughly_95_percent():
    """Over many simulated experiments, the 95 % CI should contain the true IC50 most of the time."""
    hits = 0
    runs = 100
    for seed in range(runs):
        fit = dp.analyze(make_readings(noise_sd=0.012, seed=seed)).fit
        lo, hi = fit.ic50_ci95
        hits += lo <= 45.0 <= hi
    assert 0.88 <= hits / runs <= 1.0


def test_fit_on_replicates_agrees_with_fit_on_means_when_noiseless():
    readings = make_readings()
    by_mean = dp.analyze(readings, fit_on="mean").fit
    by_rep = dp.analyze(readings, fit_on="replicates").fit
    assert by_rep.ic50 == pytest.approx(by_mean.ic50, rel=1e-4)
    assert by_rep.dof == 15 - 2  # every replicate is a point


def test_free_top_recovers_a_plateau_below_100():
    conc = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0])
    readings = make_readings(ic50=40.0, hill=1.2, top=90.0, conc=conc)
    result = dp.analyze(readings, top=None).fit
    assert result.status == "ok"
    assert result.top == pytest.approx(90.0, rel=1e-3)
    assert result.ic50 == pytest.approx(40.0, rel=1e-3)  # midpoint of the 0-90 curve


# --- blank correction ------------------------------------------------------- #
def test_blank_correction_removes_a_constant_offset():
    shifted = make_readings(offset=0.05)
    corrected = dp.analyze(shifted, blank=0.05).fit
    uncorrected = dp.analyze(shifted, blank=0.0).fit
    assert corrected.ic50 == pytest.approx(45.0, rel=1e-3)
    assert abs(uncorrected.ic50 - 45.0) > 1.0  # ignoring the blank visibly biases the answer


def test_blank_larger_than_control_is_rejected():
    with pytest.raises(ValueError):
        dp.analyze(make_readings(), blank=0.9)


# --- edge cases: 50 % never bracketed --------------------------------------- #
def test_no_concentration_reaches_50_percent():
    weak = make_readings(ic50=1000.0, hill=1.0)  # ~17 % at the top concentration
    result = dp.analyze(weak).fit
    assert result.status == "not_reached"
    assert result.ic50 is None
    assert "> 200" in result.message


def test_every_concentration_already_above_50_percent():
    potent = make_readings(ic50=1.0, hill=1.0)  # ~90 % even at the lowest concentration
    result = dp.analyze(potent).fit
    assert result.status == "below_range"
    assert result.ic50 is None
    assert "< 10" in result.message


# --- input validation ------------------------------------------------------- #
def test_too_few_concentrations_to_fit():
    readings = make_readings(conc=np.array([25.0, 100.0]))
    with pytest.raises(ValueError, match="distinct concentrations"):
        dp.analyze(readings)


def test_readings_must_be_sorted_unique_and_positive():
    with pytest.raises(ValueError):
        dp.Readings(np.array([50.0, 10.0]), np.ones((2, 3)), np.ones(3))
    with pytest.raises(ValueError):
        dp.Readings(np.array([0.0, 10.0]), np.ones((2, 3)), np.ones(3))


def test_single_replicate_has_no_sd_but_still_fits():
    result = dp.analyze(make_readings(n_reps=1))
    assert np.all(np.isnan(result.sd_percent))
    assert result.fit.ic50 == pytest.approx(45.0, rel=1e-3)


# --- CSV loading, example data and CLI -------------------------------------- #
def test_load_readings_handles_comments_unsorted_rows_and_header(tmp_path):
    path = tmp_path / "unsorted.csv"
    path.write_text(
        "# a comment\n"
        "concentration,rep1,rep2\n"
        "100,0.2,0.21\n"
        "\n"
        "0,0.8,0.79\n"
        "10,0.7,0.72\n"
    )
    readings = dp.load_readings(path)
    np.testing.assert_array_equal(readings.concentrations, [10.0, 100.0])
    np.testing.assert_allclose(readings.absorbances, [[0.7, 0.72], [0.2, 0.21]])
    np.testing.assert_allclose(readings.control, [0.8, 0.79])


def test_load_readings_requires_a_control_row(tmp_path):
    path = tmp_path / "nocontrol.csv"
    path.write_text("concentration,rep1\n10,0.7\n20,0.6\n")
    with pytest.raises(ValueError, match="control"):
        dp.load_readings(path)


def test_example_data_gives_a_sensible_ic50():
    result = dp.analyze(dp.load_readings(EXAMPLE)).fit
    assert result.status == "ok"
    assert result.ic50 == pytest.approx(45.0, rel=0.10)  # data were simulated with IC50 = 45
    assert result.r_squared > 0.99
    assert result.ic50_ci95[0] < 45.0 < result.ic50_ci95[1]


def test_cli_prints_ic50_and_writes_plot(tmp_path, capsys):
    png = tmp_path / "fit.png"
    code = dp.main([str(EXAMPLE), "--plot", str(png)])
    out = capsys.readouterr().out
    assert code == 0
    assert "IC50 =" in out and "95% CI" in out
    assert png.exists() and png.stat().st_size > 1000


def test_cli_reports_bad_input_with_exit_code_2(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("concentration,rep1\n10,0.7\n")
    assert dp.main([str(path)]) == 2
    assert "error:" in capsys.readouterr().err
