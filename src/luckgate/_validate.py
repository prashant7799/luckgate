"""Small input checks shared by the public functions.

Every public function checks its inputs here first, so that a bad input
fails early with a message that says what to fix.
"""

from __future__ import annotations

import math

import numpy as np

KURTOSIS_KINDS = ("raw", "excess")


def as_series(values, name: str = "returns", min_length: int = 2) -> np.ndarray:
    """Return `values` as a 1-D float array of finite numbers."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D (one value per period); got shape {array.shape}")
    if array.size < min_length:
        raise ValueError(f"{name} needs at least {min_length} values; got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def as_matrix(values, name: str = "scores") -> np.ndarray:
    """Return `values` as a 2-D float array: one row per trial, one column per period."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with one row per trial and one column per period; "
            f"got shape {array.shape}"
        )
    n_trials, n_periods = array.shape
    if n_trials < 2:
        raise ValueError(f"{name} needs at least 2 trials (rows); got {n_trials}")
    if n_periods < 2:
        raise ValueError(f"{name} needs at least 2 periods (columns); got {n_periods}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def finite_number(value, name: str) -> float:
    """Return `value` as a float, or raise if it is not a finite number.

    True and False are refused: float(True) is 1.0, so n_trials=True would
    quietly mean one trial.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a number, not {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number; got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return number


def n_obs_value(n_obs) -> float:
    """The number of observations must be a number greater than 1."""
    number = finite_number(n_obs, "n_obs")
    if number <= 1:
        raise ValueError(f"n_obs must be greater than 1; got {n_obs!r}")
    return number


def n_trials_value(n_trials) -> float:
    """The number of trials must be at least 1. It may be a non-integer
    'effective' number of independent trials."""
    number = finite_number(n_trials, "n_trials")
    if number < 1:
        raise ValueError(f"n_trials must be at least 1; got {n_trials!r}")
    return number


def sr_std_value(sr_std) -> float:
    number = finite_number(sr_std, "sr_std")
    if number < 0:
        raise ValueError(f"sr_std must be zero or positive; got {sr_std!r}")
    return number


def probability(value, name: str) -> float:
    number = finite_number(value, name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1; got {value!r}")
    return number


def raw_kurtosis(kurtosis, kurtosis_kind: str, skew: float) -> float:
    """Convert a kurtosis value to the raw (Pearson) form used by the papers.

    Raw kurtosis is 3 for a normal distribution. Excess kurtosis is raw
    kurtosis minus 3, so it is 0 for a normal distribution. Mixing the two
    up changes the standard error of the Sharpe ratio, so the caller must
    say which one they pass.
    """
    if kurtosis_kind not in KURTOSIS_KINDS:
        raise ValueError(f"kurtosis_kind must be 'raw' or 'excess'; got {kurtosis_kind!r}")
    value = finite_number(kurtosis, "kurtosis")
    raw = value + 3.0 if kurtosis_kind == "excess" else value

    # Any distribution has raw kurtosis >= 1 + skew**2. A smaller value is
    # impossible, and the most common cause is excess kurtosis passed as raw.
    lowest_possible = 1.0 + skew * skew
    if raw < lowest_possible - 1e-12:
        hint = " Did you pass excess kurtosis? Then set kurtosis_kind='excess'."
        raise ValueError(
            f"raw kurtosis {raw:g} is below 1 + skew**2 = {lowest_possible:g}, "
            f"which no distribution can have.{hint if kurtosis_kind == 'raw' else ''}"
        )
    return raw
