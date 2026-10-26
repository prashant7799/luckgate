"""Adversarial scenarios: ways a lucky or fragile winner can sneak through.

Each scenario has its own seed and asserts the exact list of checks that
fail. Some scenarios show a weakness on purpose: a check that is fooled
by dishonest inputs, and the other check or heuristic that catches it.
All data is synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import best_row, lottery_series, noise_scores, row_sharpes

from luckgate import Checklist, GateConfig

SHARPE_ONLY = GateConfig(checks=("min_periods", "dsr"))


def failed(verdict) -> list[str]:
    return verdict.failed


def test_a_real_winner_passes():
    scores = noise_scores(200, 250, seed=101)
    scores[17] += 0.4
    verdict = Checklist().run(scores=scores, n_trials=200)
    assert verdict.inputs["winner_row"] == 17
    assert failed(verdict) == []


def test_the_best_of_pure_noise_fails():
    scores = noise_scores(200, 250, seed=102)
    verdict = Checklist().run(scores=scores, n_trials=200)
    assert failed(verdict) == ["dsr", "pbo"]


def test_a_shortlist_with_an_understated_n_fools_dsr_but_not_pbo():
    """Keep only the 10 best of 200 noise trials and claim 10 trials. The
    DSR penalty shrinks and the DSR check passes. PBO still fails, because
    the in-sample best of the shortlist keeps losing out of sample."""
    scores = noise_scores(200, 250, seed=103)
    top10 = np.argsort(row_sharpes(scores))[-10:]
    verdict = Checklist().run(scores=scores[top10], n_trials=10)
    assert failed(verdict) == ["pbo"]
    honest = Checklist().run(scores=scores, n_trials=200)
    assert failed(honest) == ["dsr", "pbo"]


def test_a_short_record_fails_min_periods():
    scores = noise_scores(200, 24, seed=104)
    scores[17] += 0.4
    verdict = Checklist(SHARPE_ONLY).run(scores=scores, n_trials=200)
    assert failed(verdict) == ["min_periods", "dsr"]


def test_a_winner_that_only_works_in_one_season_fails_segments():
    seasons = np.repeat(["winter", "spring", "summer", "autumn"], 91)
    scores = noise_scores(60, 364, seed=105)
    for season, effect in {"winter": 0.9, "spring": 0.3, "summer": -0.25, "autumn": 0.3}.items():
        scores[5, seasons == season] += effect
    config = GateConfig(checks=("min_periods", "dsr", "pbo", "segments"))
    verdict = Checklist(config).run(scores=scores, n_trials=60, segments=seasons)
    assert verdict.inputs["winner_row"] == 5
    assert failed(verdict) == ["segments"]
    assert verdict["segments"].details["summer"]["status"] == "FAIL"


def test_a_segment_with_too_few_periods_fails():
    rng = np.random.default_rng(106)
    winner = rng.normal(0.5, 1.0, size=200)
    labels = np.array(["main"] * 195 + ["rare"] * 5)
    config = GateConfig(checks=("segments",))
    verdict = Checklist(config).run(winner, n_trials=1, segments=labels)
    assert failed(verdict) == ["segments"]
    assert "only 5 periods" in verdict["segments"].message


def test_a_collapsed_sr_std_fools_dsr_unless_floored():
    """Near-duplicate trials have almost the same Sharpe ratio, so their
    spread is tiny and the DSR penalty nearly vanishes. The opt-in
    sr_std_floor heuristic puts a floor under it. Here the floor is the
    standard error of one Sharpe estimate under the null, 1 / sqrt(T - 1)."""
    scores = noise_scores(200, 250, seed=107)
    winner = scores[best_row(scores)]
    fooled = Checklist(SHARPE_ONLY).run(winner, n_trials=200, sr_std=0.001)
    assert failed(fooled) == []

    floored = GateConfig(checks=("min_periods", "dsr"), sr_std_floor=1 / np.sqrt(249))
    verdict = Checklist(floored).run(winner, n_trials=200, sr_std=0.001)
    assert failed(verdict) == ["dsr"]


def test_a_lottery_payoff_passes_unless_positive_skew_is_clipped():
    """13 large wins and 237 small losses: strong positive skew. Positive
    skew shrinks the standard error, so one pre-registered rule (n_trials=1)
    passes. The opt-in clip_positive_skew heuristic removes that help."""
    winner = lottery_series(250, n_wins=13, win=1.6, loss=0.05, seed=7)
    plain = Checklist(SHARPE_ONLY).run(winner, n_trials=1, sr_std=0.0)
    assert plain["dsr"].details["skew"] > 3
    assert failed(plain) == []

    clipped = GateConfig(checks=("min_periods", "dsr"), clip_positive_skew=True)
    verdict = Checklist(clipped).run(winner, n_trials=1, sr_std=0.0)
    assert verdict["dsr"].details["skew_used"] == 0.0
    assert failed(verdict) == ["dsr"]


# ---------------------------------------------------------------------------
# Measured rates over many seeds
# ---------------------------------------------------------------------------


def pass_rates(n_runs, n_trials, n_periods, gain, seed_offset):
    checklist = Checklist()
    passes = 0
    for seed in range(n_runs):
        scores = noise_scores(n_trials, n_periods, seed=seed_offset + seed)
        scores[0] += gain
        passes += checklist.run(scores=scores, n_trials=n_trials).passed
    return passes / n_runs


@pytest.mark.parametrize("seed_offset", [0, 10_000])
def test_false_pass_rate_on_pure_noise_is_low(seed_offset):
    assert pass_rates(40, 50, 200, gain=0.0, seed_offset=seed_offset) <= 0.05


def test_a_clear_edge_is_usually_found():
    assert pass_rates(40, 50, 200, gain=0.5, seed_offset=20_000) >= 0.9
