"""Synthetic data for the tests. No real data is used anywhere."""

from __future__ import annotations

import numpy as np


def noise_scores(n_trials: int, n_periods: int, seed: int) -> np.ndarray:
    """A search where no trial is better than any other: standard normal scores."""
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(n_trials, n_periods))


def row_sharpes(scores: np.ndarray) -> np.ndarray:
    return scores.mean(axis=1) / scores.std(axis=1, ddof=1)


def best_row(scores: np.ndarray) -> int:
    return int(np.argmax(row_sharpes(scores)))


def lottery_series(n_periods: int, n_wins: int, win: float, loss: float, seed: int) -> np.ndarray:
    """Many small losses and a few large wins, in a random order."""
    values = np.full(n_periods, -loss)
    values[:n_wins] = win
    return np.random.default_rng(seed).permutation(values)
