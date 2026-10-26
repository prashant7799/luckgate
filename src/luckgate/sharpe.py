"""Sharpe-ratio statistics that account for luck.

All Sharpe ratios here are per period: the mean of the per-period values
divided by their standard deviation. Do not annualise them. If your data is
daily, the Sharpe ratio here is a daily Sharpe ratio, and `n_obs` is the
number of days.

The formulas come from:

- Bailey and López de Prado (2012), "The Sharpe Ratio Efficient Frontier",
  Journal of Risk 15(2): the Probabilistic Sharpe Ratio (PSR) and the
  minimum track record length.
- Bailey and López de Prado (2014), "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality", Journal of
  Portfolio Management 40(5): the Deflated Sharpe Ratio (DSR) and the
  expected maximum Sharpe ratio under N trials.

The papers write kurtosis in its raw form (3 for a normal distribution).
Every function that takes a kurtosis also takes `kurtosis_kind`, so you
must say whether you pass raw or excess kurtosis.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from luckgate._validate import (
    as_matrix,
    as_series,
    finite_number,
    n_obs_value,
    n_trials_value,
    probability,
    raw_kurtosis,
    sr_std_value,
)

#: The Euler-Mascheroni constant, used by the expected maximum formula.
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Plain statistics of one series
# ---------------------------------------------------------------------------


def sharpe_ratio(returns) -> float:
    """Per-period Sharpe ratio: mean divided by sample standard deviation (ddof=1).

    There is no risk-free rate and no annualising. Subtract a benchmark from
    `returns` first if you need one.
    """
    values = as_series(returns)
    spread = values.std(ddof=1)
    if spread == 0:
        raise ValueError("returns have zero spread, so the Sharpe ratio is undefined")
    return float(values.mean() / spread)


@dataclass(frozen=True)
class SharpeStats:
    """The four numbers the PSR and DSR formulas need from a series.

    `kurtosis` is raw kurtosis (3 for a normal distribution).
    """

    sr: float
    n_obs: int
    skew: float
    kurtosis: float


def sharpe_stats(returns) -> SharpeStats:
    """Per-period Sharpe ratio, length, skewness and raw kurtosis of a series.

    Skewness and kurtosis use the plain moment estimators (the scipy
    defaults with `bias=True`). These always satisfy kurtosis >= 1 + skew**2.
    """
    values = as_series(returns, min_length=3)
    return SharpeStats(
        sr=sharpe_ratio(values),
        n_obs=int(values.size),
        skew=float(stats.skew(values)),
        kurtosis=float(stats.kurtosis(values, fisher=False)),
    )


def trial_sharpe_std(scores) -> float:
    """Standard deviation (ddof=1) of the per-period Sharpe ratios of all trials.

    `scores` has one row per trial and one column per period. This is the
    `sr_std` input of `dsr` and `expected_max_sharpe`. It is a standard
    deviation, not a variance.
    """
    matrix = as_matrix(scores)
    spreads = matrix.std(axis=1, ddof=1)
    flat_rows = np.flatnonzero(spreads == 0)
    if flat_rows.size:
        raise ValueError(
            f"trial rows {flat_rows[:5].tolist()} have zero spread, so their Sharpe "
            "ratio is undefined"
        )
    sharpes = matrix.mean(axis=1) / spreads
    return float(sharpes.std(ddof=1))


# ---------------------------------------------------------------------------
# Formulas from the papers
# ---------------------------------------------------------------------------


def _variance_term(sr: float, skew: float, kurtosis_raw: float) -> float:
    """The non-normality term 1 - skew*SR + (kurtosis - 1)/4 * SR**2.

    With raw kurtosis this is the term in Bailey and López de Prado (2012,
    2014). With normal data (skew 0, kurtosis 3) it is 1 + SR**2 / 2.

    The term is a variance, so it must be positive. It can only be zero or
    negative for impossible inputs, most often an annualised `sr`.
    """
    term = 1.0 - skew * sr + (kurtosis_raw - 1.0) / 4.0 * sr * sr
    if term <= 0:
        raise ValueError(
            f"the variance term 1 - skew*sr + (kurtosis-1)/4*sr**2 is {term:g}; it must be "
            "positive. Check that sr is per period (not annualised) and that skew and "
            "kurtosis are right."
        )
    return term


def sharpe_se(
    sr: float,
    *,
    n_obs: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    kurtosis_kind: str = "raw",
) -> float:
    """Standard error of an estimated per-period Sharpe ratio.

    sqrt((1 - skew*SR + (kurtosis - 1)/4 * SR**2) / (n_obs - 1)), with raw
    kurtosis. Pass `kurtosis_kind="excess"` if your kurtosis is 0 for a
    normal distribution.
    """
    sr = finite_number(sr, "sr")
    n = n_obs_value(n_obs)
    skew = finite_number(skew, "skew")
    kurtosis_raw = raw_kurtosis(kurtosis, kurtosis_kind, skew)
    term = _variance_term(sr, skew, kurtosis_raw)
    return math.sqrt(term / (n - 1.0))


def psr(
    sr: float,
    *,
    n_obs: float,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    kurtosis_kind: str = "raw",
) -> float:
    """Probabilistic Sharpe Ratio: the probability that the true Sharpe ratio
    is above `sr_benchmark`, given the observed one.

    PSR = Phi((SR - SR*) / SE), where SE is `sharpe_se`. Bailey and López
    de Prado (2012).
    """
    se = sharpe_se(sr, n_obs=n_obs, skew=skew, kurtosis=kurtosis, kurtosis_kind=kurtosis_kind)
    benchmark = finite_number(sr_benchmark, "sr_benchmark")
    return float(stats.norm.cdf((float(sr) - benchmark) / se))


def expected_max_sharpe(n_trials: float, sr_std: float = 1.0) -> float:
    """Expected largest Sharpe ratio among `n_trials` trials whose true Sharpe is zero.

    sr_std * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e))), where Z is the
    inverse normal CDF and g is the Euler-Mascheroni constant. This is the
    approximation used by the DSR paper. `sr_std` is the standard deviation
    of the trials' Sharpe ratios, in the same per-period units.

    With one trial the answer is 0. Below about 1.3 trials the formula
    turns negative, so the result is clipped at 0 (the largest of one or
    more zero-mean draws cannot have a negative expected value).
    """
    n = n_trials_value(n_trials)
    spread = sr_std_value(sr_std)
    if n == 1:
        return 0.0
    first = stats.norm.ppf(1.0 - 1.0 / n)
    second = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    value = spread * ((1.0 - EULER_GAMMA) * first + EULER_GAMMA * second)
    return max(0.0, float(value))


def dsr(
    sr: float,
    *,
    n_obs: float,
    n_trials: float,
    sr_std: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    kurtosis_kind: str = "raw",
) -> float:
    """Deflated Sharpe Ratio: the PSR of the winner against the Sharpe ratio
    that the best of `n_trials` pure-noise trials would reach on average.

    DSR = PSR(SR* = expected_max_sharpe(n_trials, sr_std)). Bailey and López
    de Prado (2014).

    `sr`, `skew` and `kurtosis` describe the winner's series. `n_trials` is
    the honest number of trials, including the ones you threw away.
    `sr_std` is the standard deviation of all trials' Sharpe ratios.
    """
    benchmark = expected_max_sharpe(n_trials, sr_std)
    return psr(
        sr,
        n_obs=n_obs,
        sr_benchmark=benchmark,
        skew=skew,
        kurtosis=kurtosis,
        kurtosis_kind=kurtosis_kind,
    )


def min_track_record_length(
    sr: float,
    *,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    kurtosis_kind: str = "raw",
    confidence: float = 0.95,
) -> float:
    """Minimum track record length: how many periods are needed before the
    PSR against `sr_benchmark` reaches `confidence`.

    MinTRL = 1 + (1 - skew*SR + (kurtosis - 1)/4 * SR**2) * (Z(confidence) / (SR - SR*))**2.
    Bailey and López de Prado (2012). The answer is in periods and is not
    rounded. It is infinite when `sr` is not above `sr_benchmark`.
    """
    sr = finite_number(sr, "sr")
    benchmark = finite_number(sr_benchmark, "sr_benchmark")
    skew = finite_number(skew, "skew")
    kurtosis_raw = raw_kurtosis(kurtosis, kurtosis_kind, skew)
    level = probability(confidence, "confidence")
    if level <= 0.5:
        raise ValueError(f"confidence must be above 0.5; got {confidence!r}")
    term = _variance_term(sr, skew, kurtosis_raw)
    if sr <= benchmark:
        return math.inf
    z = stats.norm.ppf(level)
    return float(1.0 + term * (z / (sr - benchmark)) ** 2)


# ---------------------------------------------------------------------------
# Everything at once, from a series
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSRReport:
    """All the Sharpe-ratio numbers for one winner. See `dsr_report`."""

    sr: float
    n_obs: int
    skew: float
    kurtosis: float
    skew_used: float
    n_trials: float
    sr_std: float
    expected_max_sharpe: float
    sharpe_se: float
    psr: float
    dsr: float
    confidence: float
    min_track_record: float
    min_track_record_deflated: float

    def to_dict(self) -> dict:
        return asdict(self)


def dsr_report(
    returns,
    *,
    n_trials: float,
    sr_std: float,
    confidence: float = 0.95,
    clip_positive_skew: bool = False,
) -> DSRReport:
    """Compute the PSR, DSR and minimum track record lengths for one series.

    `returns` is the winner's per-period series. `n_trials` and `sr_std`
    are as in `dsr`. See `dsr_report_from_stats` for the other arguments.
    """
    return dsr_report_from_stats(
        sharpe_stats(returns),
        n_trials=n_trials,
        sr_std=sr_std,
        confidence=confidence,
        clip_positive_skew=clip_positive_skew,
    )


def dsr_report_from_stats(
    measured: SharpeStats,
    *,
    n_trials: float,
    sr_std: float,
    confidence: float = 0.95,
    clip_positive_skew: bool = False,
) -> DSRReport:
    """Like `dsr_report`, but from summary numbers instead of a series.

    `measured.kurtosis` must be raw kurtosis.

    `min_track_record` is the number of periods needed for PSR(0) to reach
    `confidence`. `min_track_record_deflated` is the number needed for the
    DSR to reach it.

    `clip_positive_skew=True` replaces a positive skew by 0 in the formulas.
    This is a cautious heuristic, not part of the papers: positive skew
    shrinks the standard error, so a few large wins can make a weak series
    look reliable. Negative skew is always kept. `skew` in the report is the
    measured value and `skew_used` is the value the formulas used.
    """
    skew_used = min(measured.skew, 0.0) if clip_positive_skew else measured.skew
    benchmark = expected_max_sharpe(n_trials, sr_std)
    shape = {"skew": skew_used, "kurtosis": measured.kurtosis, "kurtosis_kind": "raw"}
    sr = measured.sr
    n_obs = measured.n_obs
    return DSRReport(
        sr=sr,
        n_obs=n_obs,
        skew=measured.skew,
        kurtosis=measured.kurtosis,
        skew_used=skew_used,
        n_trials=float(n_trials),
        sr_std=float(sr_std),
        expected_max_sharpe=benchmark,
        sharpe_se=sharpe_se(sr, n_obs=n_obs, **shape),
        psr=psr(sr, n_obs=n_obs, sr_benchmark=0.0, **shape),
        dsr=psr(sr, n_obs=n_obs, sr_benchmark=benchmark, **shape),
        confidence=float(confidence),
        min_track_record=min_track_record_length(
            sr, sr_benchmark=0.0, confidence=confidence, **shape
        ),
        min_track_record_deflated=min_track_record_length(
            sr, sr_benchmark=benchmark, confidence=confidence, **shape
        ),
    )
