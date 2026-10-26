"""A slow, obvious version of CSCV, written straight from the paper.

It loops over every split and recomputes each trial's performance from
the raw columns. It follows the rules that luckgate documents:

- A half whose values are all equal has zero spread. Its Sharpe ratio is
  +inf, -inf or 0, following the sign of that value.
- Two scores tie when they are equal, or when both are finite and differ
  by at most 1e-12 times the size of the values behind them: the mean
  absolute value over the half, divided by the standard deviation for the
  Sharpe ratio (the larger of the two sizes).
- A tie for the best in-sample score goes to the first row. Ties out of
  sample share the average rank.

The fast version in luckgate must agree with it.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

TIE_TOLERANCE = 1e-12  # the tolerance that luckgate documents


def flat_sharpe(value: float) -> float:
    """The Sharpe ratio of a half with zero spread: +inf, -inf or 0."""
    return np.inf if value > 0 else (-np.inf if value < 0 else 0.0)


def performance(values: np.ndarray, metric: str) -> tuple[float, float]:
    """The score of one trial on one half, and the size of the values behind it."""
    mean = float(np.mean(values))
    typical = float(np.mean(np.abs(values)))
    if metric == "mean":
        return mean, typical
    if np.all(values == values[0]):
        return flat_sharpe(float(values[0])), 0.0  # zero spread: every value is the same
    spread = float(np.std(values, ddof=1))
    if spread == 0:
        return flat_sharpe(mean), 0.0  # a spread too small for floating point to hold
    return mean / spread, typical / spread


def tied(a: tuple[float, float], b: tuple[float, float]) -> bool:
    (score_a, size_a), (score_b, size_b) = a, b
    if score_a == score_b:
        return True
    if not (math.isfinite(score_a) and math.isfinite(score_b)):
        return False
    return abs(score_a - score_b) <= TIE_TOLERANCE * max(size_a, size_b)


def reference_logits(scores: np.ndarray, n_blocks: int, metric: str = "sharpe") -> np.ndarray:
    n_trials, n_periods = scores.shape
    blocks = np.array_split(np.arange(n_periods), n_blocks)
    logits = []
    for train_blocks in itertools.combinations(range(n_blocks), n_blocks // 2):
        test_blocks = [b for b in range(n_blocks) if b not in train_blocks]
        train_columns = np.concatenate([blocks[b] for b in train_blocks])
        test_columns = np.concatenate([blocks[b] for b in test_blocks])

        train = [performance(scores[n, train_columns], metric) for n in range(n_trials)]
        test = [performance(scores[n, test_columns], metric) for n in range(n_trials)]

        # The first row that ties with the best in-sample score.
        top = max(range(n_trials), key=lambda n: train[n][0])
        best = next(n for n in range(n_trials) if tied(train[n], train[top]))

        # 1 = worst. The others that tie with the winner share its rank.
        winner = test[best]
        lower = sum(
            1 for n in range(n_trials) if test[n][0] < winner[0] and not tied(test[n], winner)
        )
        tied_others = sum(1 for n in range(n_trials) if n != best and tied(test[n], winner))
        rank = 1 + lower + 0.5 * tied_others
        omega = rank / (n_trials + 1)
        logits.append(np.log(omega / (1 - omega)))
    return np.array(logits)
