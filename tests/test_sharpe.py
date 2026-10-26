"""PSR, DSR, expected maximum Sharpe and minimum track record length.

Golden values come from the numerical example in Bailey and López de
Prado (2014). The other tests check hand computations, identities between
the formulas, and the mistakes these formulas are known for.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, stats

from luckgate import (
    SharpeStats,
    dsr,
    dsr_report,
    dsr_report_from_stats,
    expected_max_sharpe,
    min_track_record_length,
    psr,
    sharpe_ratio,
    sharpe_se,
    sharpe_stats,
    trial_sharpe_std,
)
from luckgate.sharpe import EULER_GAMMA

# The paper's example: 100 trials, variance of annualised Sharpe ratios 0.5,
# best annualised Sharpe 2.5, 1250 daily values, skew -3, raw kurtosis 10.
DAYS = 250
PAPER = {
    "sr": 2.5 / math.sqrt(DAYS),
    "n_obs": 1250,
    "n_trials": 100,
    "sr_std": math.sqrt(0.5 / DAYS),
    "skew": -3.0,
    "kurtosis": 10.0,
}


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Golden values from the DSR paper
# ---------------------------------------------------------------------------


def test_dsr_matches_the_paper_example():
    assert dsr(**PAPER) == pytest.approx(0.9004, abs=5e-5)


def test_expected_max_sharpe_of_the_paper_example():
    # 0.1132 per day; about 1.79 once annualised.
    value = expected_max_sharpe(PAPER["n_trials"], PAPER["sr_std"])
    assert value == pytest.approx(0.1132, abs=5e-5)
    assert value * math.sqrt(DAYS) == pytest.approx(1.7894, abs=5e-4)


def test_paper_example_by_hand():
    """Every step of the example written out with math only."""
    z1 = stats.norm.ppf(1 - 1 / 100)
    z2 = stats.norm.ppf(1 - 1 / (100 * math.e))
    sr0 = PAPER["sr_std"] * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    sr = PAPER["sr"]
    term = 1 - (-3.0) * sr + (10.0 - 1.0) / 4.0 * sr**2
    z = (sr - sr0) * math.sqrt(1250 - 1) / math.sqrt(term)
    assert dsr(**PAPER) == pytest.approx(normal_cdf(z), abs=1e-12)


# ---------------------------------------------------------------------------
# Regression tests for known mistakes
# ---------------------------------------------------------------------------


def test_excess_kurtosis_gives_the_same_answer_once_declared():
    excess = dict(PAPER, kurtosis=7.0)
    assert dsr(**excess, kurtosis_kind="excess") == pytest.approx(dsr(**PAPER), abs=1e-15)


def test_excess_kurtosis_passed_as_raw_changes_the_answer():
    """The bug: excess kurtosis fed to the raw-kurtosis formula. The kurtosis
    term then shrinks by 3/4 * SR**2, the standard error is too small, and
    the DSR is too high. The effect grows with the per-period Sharpe ratio."""
    args = {"sr": 0.5, "n_obs": 46, "n_trials": 10, "sr_std": 0.1, "skew": -1.0}
    right = dsr(**args, kurtosis=10.0, kurtosis_kind="raw")
    wrong = dsr(**args, kurtosis=7.0, kurtosis_kind="raw")  # excess value, labelled raw
    # Here the mix-up turns a FAIL at the 0.95 line into a PASS.
    assert right < 0.95 < wrong

    se_right = sharpe_se(0.5, n_obs=46, skew=-1.0, kurtosis=10.0)
    se_wrong = sharpe_se(0.5, n_obs=46, skew=-1.0, kurtosis=7.0)
    term_right = 1 + 0.5 + (10 - 1) / 4 * 0.25
    term_wrong = 1 + 0.5 + (7 - 1) / 4 * 0.25
    assert se_right / se_wrong == pytest.approx(math.sqrt(term_right / term_wrong))


def test_excess_kurtosis_of_a_normal_series_passed_as_raw_is_rejected():
    # Excess kurtosis 0 (a normal distribution) labelled raw is impossible.
    with pytest.raises(ValueError, match="excess"):
        sharpe_se(0.1, n_obs=100, kurtosis=0.0)
    with pytest.raises(ValueError, match="1 \\+ skew"):
        psr(0.1, n_obs=100, skew=2.0, kurtosis=4.0)


def test_annualised_sharpe_in_the_standard_error_is_wrong():
    """The other bug: using the annual Sharpe ratio in the per-period formula.
    The paper example then reads as near-certain instead of 0.9004."""
    annual = dict(PAPER, sr=2.5, sr_std=math.sqrt(0.5))
    assert dsr(**annual) > 0.9999
    assert abs(dsr(**annual) - dsr(**PAPER)) > 0.09


def test_dsr_report_never_annualises():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.05, 1.0, size=500)
    report = dsr_report(returns, n_trials=20, sr_std=0.05)
    assert report.sr == pytest.approx(returns.mean() / returns.std(ddof=1))


# ---------------------------------------------------------------------------
# Hand computations and identities
# ---------------------------------------------------------------------------


def test_sharpe_se_for_normal_data():
    sr, n = 0.2, 101
    expected = math.sqrt((1 + sr**2 / 2) / (n - 1))
    assert sharpe_se(sr, n_obs=n) == pytest.approx(expected, rel=1e-14)


def test_psr_by_hand():
    sr, n, skew, kurt = 0.12, 400, -0.5, 5.0
    term = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    expected = normal_cdf((sr - 0.02) * math.sqrt(n - 1) / math.sqrt(term))
    assert psr(sr, n_obs=n, sr_benchmark=0.02, skew=skew, kurtosis=kurt) == pytest.approx(
        expected, abs=1e-12
    )


def test_psr_is_one_half_at_the_benchmark():
    assert psr(0.3, n_obs=50, sr_benchmark=0.3, skew=-1, kurtosis=6) == pytest.approx(0.5)


def test_dsr_is_psr_against_the_expected_maximum():
    benchmark = expected_max_sharpe(PAPER["n_trials"], PAPER["sr_std"])
    expected = psr(
        PAPER["sr"],
        n_obs=PAPER["n_obs"],
        sr_benchmark=benchmark,
        skew=PAPER["skew"],
        kurtosis=PAPER["kurtosis"],
    )
    assert dsr(**PAPER) == expected


def test_one_trial_means_no_deflation():
    args = dict(PAPER, n_trials=1)
    assert dsr(**args) == psr(
        PAPER["sr"], n_obs=PAPER["n_obs"], skew=PAPER["skew"], kurtosis=PAPER["kurtosis"]
    )


@settings(max_examples=200, deadline=None)
@given(
    sr=st.floats(0.01, 1.0),
    benchmark=st.floats(-0.5, 0.5),
    skew=st.floats(-2.0, 2.0),
    extra_kurtosis=st.floats(0.0, 20.0),
    confidence=st.floats(0.6, 0.999),
)
def test_psr_at_the_min_track_record_length_equals_the_confidence(
    sr, benchmark, skew, extra_kurtosis, confidence
):
    if sr - benchmark < 0.01:
        return
    kurtosis = 1 + skew**2 + extra_kurtosis  # always a possible raw kurtosis
    if 1 - skew * sr + (kurtosis - 1) / 4 * sr**2 <= 1e-6:
        return
    length = min_track_record_length(
        sr, sr_benchmark=benchmark, skew=skew, kurtosis=kurtosis, confidence=confidence
    )
    reached = psr(sr, n_obs=length, sr_benchmark=benchmark, skew=skew, kurtosis=kurtosis)
    assert reached == pytest.approx(confidence, abs=1e-9)


def test_min_track_record_length_by_hand():
    sr, skew, kurt = 0.1, -1.0, 6.0
    term = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    expected = 1 + term * (stats.norm.ppf(0.95) / sr) ** 2
    assert min_track_record_length(sr, skew=skew, kurtosis=kurt) == pytest.approx(expected)


def test_min_track_record_length_is_infinite_without_an_edge():
    assert min_track_record_length(0.1, sr_benchmark=0.1) == math.inf
    assert min_track_record_length(-0.2) == math.inf


@settings(max_examples=100, deadline=None)
@given(sr=st.floats(-1.0, 1.0), n_obs=st.integers(2, 10_000), skew=st.floats(-3, 3))
def test_psr_is_a_probability(sr, n_obs, skew):
    kurtosis = 1 + skew**2 + 2.0
    if 1 - skew * sr + (kurtosis - 1) / 4 * sr**2 <= 1e-9:
        return
    value = psr(sr, n_obs=n_obs, skew=skew, kurtosis=kurtosis)
    assert 0.0 <= value <= 1.0


def test_psr_grows_with_the_track_record_length():
    values = [psr(0.1, n_obs=n) for n in (10, 50, 250, 1000)]
    assert values == sorted(values)


@settings(max_examples=150, deadline=None)
@given(
    sr=st.floats(-0.5, 0.5),
    n_obs=st.integers(10, 5_000),
    n_trials=st.floats(1.0, 1_000.0),
    extra_trials=st.floats(0.1, 1_000.0),
    sr_std=st.floats(0.001, 0.2),
)
def test_more_trials_or_more_spread_never_raise_the_dsr(sr, n_obs, n_trials, extra_trials, sr_std):
    """More trials, or trials that are more spread out, make luck a better
    explanation. The DSR can only go down."""
    base = dsr(sr, n_obs=n_obs, n_trials=n_trials, sr_std=sr_std)
    more_trials = dsr(sr, n_obs=n_obs, n_trials=n_trials + extra_trials, sr_std=sr_std)
    more_spread = dsr(sr, n_obs=n_obs, n_trials=n_trials, sr_std=2 * sr_std)
    assert more_trials <= base + 1e-12
    assert more_spread <= base + 1e-12


@settings(max_examples=100, deadline=None)
@given(sr=st.floats(-0.5, 0.5), benchmark=st.floats(-0.5, 0.5), gap=st.floats(0.001, 0.5))
def test_a_higher_benchmark_never_raises_the_psr(sr, benchmark, gap):
    lower = psr(sr, n_obs=250, sr_benchmark=benchmark, skew=-1.0, kurtosis=6.0)
    higher = psr(sr, n_obs=250, sr_benchmark=benchmark + gap, skew=-1.0, kurtosis=6.0)
    assert higher <= lower


@settings(max_examples=40, deadline=None)
@given(scale=st.floats(1e-3, 1e3), seed=st.integers(0, 10_000))
def test_rescaling_the_series_does_not_change_the_report(scale, seed):
    """The Sharpe ratio, skewness and kurtosis have no units, so the PSR,
    DSR and track record lengths cannot depend on the units of the series."""
    series = np.random.default_rng(seed).standard_t(5, size=200) * 0.5 + 0.1
    plain = dsr_report(series, n_trials=30, sr_std=0.07)
    scaled = dsr_report(series * scale, n_trials=30, sr_std=0.07)
    assert scaled.sr == pytest.approx(plain.sr, rel=1e-9)
    assert scaled.psr == pytest.approx(plain.psr, abs=1e-9)
    assert scaled.dsr == pytest.approx(plain.dsr, abs=1e-9)
    assert scaled.min_track_record == pytest.approx(plain.min_track_record, rel=1e-6)


# ---------------------------------------------------------------------------
# Expected maximum Sharpe ratio
# ---------------------------------------------------------------------------


def exact_expected_max_of_normals(n: int) -> float:
    """E[max of n standard normals] by numerical integration."""

    def integrand(x):
        return x * n * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1)

    value, _ = integrate.quad(integrand, -12, 12, limit=200)
    return value


@pytest.mark.parametrize("n", [10, 100, 1_000, 10_000])
def test_expected_max_is_close_to_the_exact_value(n):
    approx = expected_max_sharpe(n, 1.0)
    exact = exact_expected_max_of_normals(n)
    assert approx == pytest.approx(exact, rel=0.03)


def test_expected_max_basic_properties():
    assert expected_max_sharpe(1, 0.3) == 0.0
    assert expected_max_sharpe(1.1, 0.3) == 0.0  # clipped: the formula is negative here
    values = [expected_max_sharpe(n, 1.0) for n in (2, 5, 10, 100, 1000)]
    assert values == sorted(values)
    assert expected_max_sharpe(100, 0.2) == pytest.approx(0.2 * expected_max_sharpe(100, 1.0))
    assert expected_max_sharpe(100, 0.0) == 0.0


def test_expected_max_accepts_an_effective_number_of_trials():
    assert expected_max_sharpe(10, 1.0) < expected_max_sharpe(10.5, 1.0) < expected_max_sharpe(11)


# ---------------------------------------------------------------------------
# Statistics of a series
# ---------------------------------------------------------------------------


def test_sharpe_stats_match_scipy():
    rng = np.random.default_rng(5)
    x = rng.standard_t(5, size=300)
    measured = sharpe_stats(x)
    assert measured.sr == pytest.approx(x.mean() / x.std(ddof=1))
    assert measured.n_obs == 300
    assert measured.skew == pytest.approx(stats.skew(x))
    assert measured.kurtosis == pytest.approx(stats.kurtosis(x) + 3.0)  # raw = excess + 3


def test_trial_sharpe_std():
    rng = np.random.default_rng(6)
    scores = rng.normal(size=(30, 80))
    sharpes = [row.mean() / row.std(ddof=1) for row in scores]
    assert trial_sharpe_std(scores) == pytest.approx(np.std(sharpes, ddof=1))


def test_dsr_report_agrees_with_the_formulas():
    rng = np.random.default_rng(8)
    x = rng.normal(0.1, 1.0, size=400)
    report = dsr_report(x, n_trials=50, sr_std=0.06, confidence=0.9)
    m = sharpe_stats(x)
    shape = {"skew": m.skew, "kurtosis": m.kurtosis}
    assert report.dsr == dsr(m.sr, n_obs=m.n_obs, n_trials=50, sr_std=0.06, **shape)
    assert report.psr == psr(m.sr, n_obs=m.n_obs, **shape)
    assert report.expected_max_sharpe == expected_max_sharpe(50, 0.06)
    assert report.min_track_record == min_track_record_length(m.sr, confidence=0.9, **shape)
    assert report.min_track_record_deflated == min_track_record_length(
        m.sr, sr_benchmark=report.expected_max_sharpe, confidence=0.9, **shape
    )
    assert report.to_dict()["n_obs"] == 400


def test_dsr_report_from_stats_reproduces_the_paper():
    measured = SharpeStats(sr=PAPER["sr"], n_obs=1250, skew=-3.0, kurtosis=10.0)
    report = dsr_report_from_stats(measured, n_trials=100, sr_std=PAPER["sr_std"])
    assert report.dsr == pytest.approx(0.9004, abs=5e-5)


def test_clipping_positive_skew_only_ever_lowers_the_dsr():
    rng = np.random.default_rng(9)
    right_skewed = rng.exponential(1.0, size=300) - 0.9
    left_skewed = 0.9 - rng.exponential(1.0, size=300) + 0.2
    for series in (right_skewed, left_skewed):
        plain = dsr_report(series, n_trials=5, sr_std=0.05)
        clipped = dsr_report(series, n_trials=5, sr_std=0.05, clip_positive_skew=True)
        assert clipped.skew == plain.skew
        assert clipped.skew_used == min(plain.skew, 0.0)
        assert clipped.dsr <= plain.dsr
    assert dsr_report(right_skewed, n_trials=5, sr_std=0.05, clip_positive_skew=True).dsr < (
        dsr_report(right_skewed, n_trials=5, sr_std=0.05).dsr
    )
    assert dsr_report(left_skewed, n_trials=5, sr_std=0.05, clip_positive_skew=True).dsr == (
        dsr_report(left_skewed, n_trials=5, sr_std=0.05).dsr
    )


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: sharpe_se(0.1, n_obs=1),
        lambda: sharpe_se(float("nan"), n_obs=10),
        lambda: sharpe_se(0.1, n_obs=10, kurtosis_kind="pearson"),
        lambda: psr(0.1, n_obs=10, sr_benchmark=float("inf")),
        lambda: expected_max_sharpe(0.5, 1.0),
        lambda: expected_max_sharpe(10, -0.1),
        lambda: dsr(0.1, n_obs=100, n_trials=10, sr_std=float("nan")),
        lambda: min_track_record_length(0.1, confidence=0.5),
        lambda: min_track_record_length(0.1, confidence=1.0),
        lambda: sharpe_ratio([1.0, 1.0, 1.0]),
        lambda: sharpe_ratio([1.0]),
        lambda: sharpe_ratio([[1.0, 2.0], [3.0, 4.0]]),
        lambda: sharpe_ratio([1.0, float("nan"), 2.0]),
        lambda: trial_sharpe_std([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]),
        lambda: trial_sharpe_std([1.0, 2.0, 3.0]),
        # float(True) is 1.0, so True would quietly mean one trial or one period.
        lambda: expected_max_sharpe(True, 1.0),
        lambda: dsr(0.1, n_obs=100, n_trials=True, sr_std=0.1),
        lambda: psr(0.1, n_obs=True),
        lambda: sharpe_se(0.1, n_obs=100, skew=np.True_),
    ],
)
def test_bad_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


def test_variance_term_must_be_positive():
    # Raw kurtosis exactly 1 + skew**2 with SR = 2 / skew makes the term 0.
    with pytest.raises(ValueError, match="variance term"):
        sharpe_se(1.0, n_obs=100, skew=2.0, kurtosis=5.0)
    # The minimum track record length uses the same term, so it must refuse
    # the same inputs instead of returning a meaningless 1.
    with pytest.raises(ValueError, match="variance term"):
        min_track_record_length(1.0, skew=2.0, kurtosis=5.0)
    with pytest.raises(ValueError, match="variance term"):
        min_track_record_length(1.0, sr_benchmark=2.0, skew=2.0, kurtosis=5.0)
