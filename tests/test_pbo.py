"""CSCV probability of backtest overfitting.

The fast version is checked against a slow reference written straight
from the paper, on many random tables and on tables full of ties. Then
come cases with a known answer, invariances, and input checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from reference_pbo import reference_logits
from synthetic import noise_scores

from luckgate import cscv_pbo
from luckgate.pbo import _choose_splits

# ---------------------------------------------------------------------------
# Agreement with the slow reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["sharpe", "mean"])
@pytest.mark.parametrize(
    ("n_trials", "n_periods", "n_blocks", "seed"),
    [
        (5, 40, 4, 1),
        (12, 60, 6, 2),
        (30, 97, 8, 3),  # 97 periods do not split evenly into 8 blocks
        (8, 33, 10, 4),
        (3, 16, 2, 5),
    ],
)
def test_matches_the_reference(metric, n_trials, n_periods, n_blocks, seed):
    scores = noise_scores(n_trials, n_periods, seed)
    result = cscv_pbo(scores, n_blocks=n_blocks, metric=metric)
    expected = reference_logits(scores, n_blocks, metric)
    np.testing.assert_allclose(result.logits, expected, rtol=0, atol=1e-12)
    assert result.pbo == np.mean(expected <= 0)


def fits(n_periods: int, n_blocks: int) -> bool:
    """True if the table is long enough for `n_blocks` and the Sharpe metric."""
    return n_periods >= n_blocks and (n_blocks // 2) * (n_periods // n_blocks) >= 2


@settings(max_examples=60, deadline=None)
@given(
    scores=arrays(
        np.float64,
        st.tuples(st.integers(2, 9), st.integers(8, 30)),
        elements=st.floats(-100, 100, allow_nan=False, width=32),
    ),
    n_blocks=st.sampled_from([2, 4, 6, 8]),
    metric=st.sampled_from(["sharpe", "mean"]),
)
def test_matches_the_reference_on_random_tables(scores, n_blocks, metric):
    """No offset is added to dodge ties: hypothesis often repeats values, so
    many of these tables have flat rows, flat halves and tied trials."""
    if not fits(scores.shape[1], n_blocks):
        return
    result = cscv_pbo(scores, n_blocks=n_blocks, metric=metric)
    expected = reference_logits(scores, n_blocks, metric)
    np.testing.assert_allclose(result.logits, expected, rtol=0, atol=1e-12)


@st.composite
def tables_with_ties(draw):
    """Tables built from a few values, some of which (0.1, 0.2, -0.3) do not
    add up exactly in floating point. Some rows are copied over others, in
    whole or on a stretch of periods, so many scores tie."""
    n_trials = draw(st.integers(2, 8))
    n_periods = draw(st.integers(8, 30))
    pool = draw(
        st.lists(st.sampled_from([0.0, 1.0, -1.0, 0.1, 0.2, -0.3, 2.5]), min_size=1, max_size=4)
    )
    scores = draw(arrays(np.float64, (n_trials, n_periods), elements=st.sampled_from(pool)))
    for _ in range(draw(st.integers(0, 3))):
        source = draw(st.integers(0, n_trials - 1))
        target = draw(st.integers(0, n_trials - 1))
        start = draw(st.integers(0, n_periods - 1))
        stop = draw(st.integers(start + 1, n_periods))
        scores[target, start:stop] = scores[source, start:stop]
    return scores


@settings(max_examples=60, deadline=None)
@given(
    scores=tables_with_ties(),
    n_blocks=st.sampled_from([2, 4, 6, 8]),
    metric=st.sampled_from(["sharpe", "mean"]),
)
def test_matches_the_reference_when_many_scores_tie(scores, n_blocks, metric):
    if not fits(scores.shape[1], n_blocks):
        return
    result = cscv_pbo(scores, n_blocks=n_blocks, metric=metric)
    expected = reference_logits(scores, n_blocks, metric)
    np.testing.assert_allclose(result.logits, expected, rtol=0, atol=1e-12)


def test_large_means_do_not_lose_precision():
    """Squared distances from each block's own mean keep the variance
    accurate when the mean is huge next to the spread."""
    scores = noise_scores(10, 64, seed=6) + 1e6
    result = cscv_pbo(scores, n_blocks=8)
    expected = reference_logits(scores, 8)
    np.testing.assert_allclose(result.logits, expected, rtol=0, atol=1e-9)


def test_a_level_shift_between_halves_does_not_lose_precision():
    """Small values in the first half and values near 1e6 in the second. A
    sum-of-squares formula around each trial's overall mean would lose
    every digit of the small half's spread."""
    small = 1e-3 * noise_scores(6, 32, seed=23)
    large = 1e6 + noise_scores(6, 32, seed=24)
    scores = np.hstack([small, large])
    result = cscv_pbo(scores, n_blocks=4)
    np.testing.assert_allclose(result.logits, reference_logits(scores, 4), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Ties follow the documented rules, not rounding
# ---------------------------------------------------------------------------


def test_an_all_zero_half_gives_every_trial_a_sharpe_ratio_of_zero():
    """A half of zeros has zero spread and a mean of 0, so every trial scores
    exactly 0 there, they all tie, and the first row is selected."""
    scores = np.random.default_rng(0).normal(size=(5, 40))
    scores[:, :20] = 0.0
    result = cscv_pbo(scores, n_blocks=2)
    assert result.selected[0] == 0
    assert result.is_performance[0] == 0.0
    np.testing.assert_allclose(result.logits, reference_logits(scores, 2), rtol=0, atol=1e-12)
    assert result.pbo == 0.5


@pytest.mark.parametrize("metric", ["sharpe", "mean"])
@pytest.mark.parametrize("n_blocks", [4, 6])
def test_sparse_zero_one_rows_match_the_reference(n_blocks, metric):
    """Hits and misses: rows of 0 and 1, where many trials have the same
    number of hits in a half."""
    scores = (np.random.default_rng(1).random((8, 60)) < 0.15).astype(float)
    result = cscv_pbo(scores, n_blocks=n_blocks, metric=metric)
    expected = reference_logits(scores, n_blocks, metric)
    np.testing.assert_allclose(result.logits, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_blocks", [4, 8])
def test_rows_identical_on_some_blocks_only_match_the_reference(n_blocks):
    """Rows 1 to 3 copy row 0 on the first half of the periods only. Where a
    half lies inside that stretch, the four rows must tie exactly."""
    scores = noise_scores(6, 48, seed=2)
    scores[1:4, :24] = scores[0, :24]
    result = cscv_pbo(scores, n_blocks=n_blocks)
    expected = reference_logits(scores, n_blocks)
    np.testing.assert_allclose(result.logits, expected, rtol=0, atol=1e-12)


def test_duplicated_trials_match_the_reference():
    rows = noise_scores(4, 40, seed=3)
    scores = rows[[0, 1, 1, 2, 0, 3, 2]]
    result = cscv_pbo(scores, n_blocks=4)
    np.testing.assert_allclose(result.logits, reference_logits(scores, 4), rtol=0, atol=1e-12)
    assert set(result.selected.tolist()) <= {0, 1, 3, 5}  # never the later copy of a row


@pytest.mark.parametrize("metric", ["sharpe", "mean"])
def test_means_that_are_zero_on_paper_tie(metric):
    """Every block of 3 periods holds 0.1, 0.2 and -0.3 in some order, so
    every half of every trial has a mean of exactly 0 on paper. In floating
    point the sums come out as tiny numbers that depend on the order. All
    trials must still tie: row 0 is selected and omega is 1/2."""
    rng = np.random.default_rng(4)
    scores = np.vstack(
        [np.concatenate([rng.permutation([0.1, 0.2, -0.3]) for _ in range(8)]) for _ in range(6)]
    )
    result = cscv_pbo(scores, n_blocks=8, metric=metric)
    assert np.all(result.selected == 0)
    np.testing.assert_allclose(result.relative_ranks, 0.5)
    assert result.pbo == 1.0
    np.testing.assert_allclose(result.logits, reference_logits(scores, 8, metric), atol=1e-12)


def test_a_flat_half_is_infinite_even_when_its_sum_does_not_round_exactly():
    """0.1 added up 12 times is not exactly 1.2 in floating point. The half
    is still flat, so its Sharpe ratio is +inf, not a huge finite number."""
    scores = noise_scores(4, 24, seed=7)
    scores[1, :12] = 0.1
    scores[2, :12] = 0.1
    result = cscv_pbo(scores, n_blocks=2)
    assert result.selected[0] == 1  # rows 1 and 2 tie at +inf; the first wins
    assert result.is_performance[0] == np.inf


# ---------------------------------------------------------------------------
# Known answers
# ---------------------------------------------------------------------------


def test_number_of_splits():
    # The PBO paper prints "12,780" for S = 16. That is a typo: C(16, 8) is 12,870.
    result = cscv_pbo(noise_scores(4, 64, seed=7), n_blocks=16)
    assert result.n_splits == math.comb(16, 8) == 12870
    assert result.logits.shape == (12870,)


def test_rows_that_sum_to_zero_always_overfit():
    """If every row sums to zero and the halves are the same size, the best
    in-sample mean is always the worst out-of-sample mean. PBO is 1."""
    scores = noise_scores(20, 64, seed=8)
    scores -= scores.mean(axis=1, keepdims=True)
    result = cscv_pbo(scores, n_blocks=8, metric="mean")
    assert result.pbo == 1.0
    np.testing.assert_allclose(result.relative_ranks, 1 / 21)  # rank 1 of 20


def test_one_dominant_trial_never_overfits():
    scores = noise_scores(50, 128, seed=9)
    scores[7] += 5.0
    result = cscv_pbo(scores)
    assert result.pbo == 0.0
    assert np.all(result.selected == 7)
    np.testing.assert_allclose(result.relative_ranks, 50 / 51)  # rank 50 of 50
    assert result.prob_oos_loss == 0.0


def test_pure_noise_gives_about_one_half():
    values = [cscv_pbo(noise_scores(40, 96, seed=seed)).pbo for seed in range(20)]
    assert 0.35 < np.mean(values) < 0.65


def test_ties_share_the_average_rank():
    """Identical trials tie everywhere: the winner gets rank (N + 1) / 2, so
    omega is exactly 1/2, the logit is 0, and that counts as overfit."""
    row = noise_scores(1, 32, seed=10)[0]
    scores = np.vstack([row, row, row])
    result = cscv_pbo(scores, n_blocks=4)
    np.testing.assert_allclose(result.relative_ranks, 0.5)
    np.testing.assert_allclose(result.logits, 0.0, atol=1e-15)
    assert result.pbo == 1.0


def test_a_tie_for_the_best_in_sample_score_picks_the_first_row():
    """Rows 1 and 3 are identical and better than the rest. Row 1 is picked
    in every split. Out of sample the two share the top two ranks, so the
    rank is 4.5 of 5 and omega is 4.5 / 6 = 0.75."""
    scores = noise_scores(5, 32, seed=22)
    scores[1] += 5.0
    scores[3] = scores[1]
    result = cscv_pbo(scores, n_blocks=4)
    assert np.all(result.selected == 1)
    np.testing.assert_allclose(result.relative_ranks, 0.75)
    assert result.pbo == 0.0


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    log_scales=arrays(np.float64, 12, elements=st.floats(-3, 3)),
)
def test_rescaling_each_trial_on_its_own_does_not_change_the_sharpe_result(seed, log_scales):
    """A Sharpe ratio has no units. Multiplying one trial's scores by a
    positive number changes none of its Sharpe ratios, so no rank moves."""
    scores = noise_scores(12, 64, seed)
    scaled = scores * (10.0**log_scales)[:, None]
    plain = cscv_pbo(scores, n_blocks=8)
    rescaled = cscv_pbo(scaled, n_blocks=8)
    np.testing.assert_array_equal(rescaled.selected, plain.selected)
    np.testing.assert_allclose(rescaled.logits, plain.logits, atol=1e-12)


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n_trials=st.integers(2, 15),
    n_blocks=st.sampled_from([2, 4, 6, 8]),
    metric=st.sampled_from(["sharpe", "mean"]),
)
def test_every_result_is_in_range(seed, n_trials, n_blocks, metric):
    scores = noise_scores(n_trials, 48, seed)
    result = cscv_pbo(scores, n_blocks=n_blocks, metric=metric)
    assert 0.0 <= result.pbo <= 1.0
    assert result.pbo == np.mean(result.logits <= 0)
    assert np.all(result.relative_ranks >= 1 / (n_trials + 1) - 1e-15)
    assert np.all(result.relative_ranks <= n_trials / (n_trials + 1) + 1e-15)
    assert np.all((result.selected >= 0) & (result.selected < n_trials))
    assert np.all(np.isfinite(result.logits))


def test_flat_halves_get_an_infinite_sharpe_ratio():
    scores = noise_scores(4, 16, seed=11)
    scores[2] = 3.0  # constant and positive: the best trial in every half
    result = cscv_pbo(scores, n_blocks=4)
    assert np.all(result.selected == 2)
    assert np.all(np.isinf(result.is_performance))
    assert result.pbo == 0.0


# ---------------------------------------------------------------------------
# Invariances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["sharpe", "mean"])
def test_positive_scaling_does_not_change_the_result(metric):
    scores = noise_scores(25, 80, seed=12)
    plain = cscv_pbo(scores, n_blocks=8, metric=metric)
    scaled = cscv_pbo(scores * 37.5, n_blocks=8, metric=metric)
    np.testing.assert_allclose(plain.logits, scaled.logits, atol=1e-12)


def test_shifting_every_value_does_not_change_the_mean_metric():
    scores = noise_scores(25, 80, seed=13)
    plain = cscv_pbo(scores, n_blocks=8, metric="mean")
    shifted = cscv_pbo(scores + 4.0, n_blocks=8, metric="mean")
    np.testing.assert_allclose(plain.logits, shifted.logits, atol=1e-12)


def test_reordering_the_trials_does_not_change_the_result():
    scores = noise_scores(25, 80, seed=14)
    order = np.random.default_rng(0).permutation(25)
    plain = cscv_pbo(scores, n_blocks=8)
    shuffled = cscv_pbo(scores[order], n_blocks=8)
    np.testing.assert_allclose(plain.logits, shuffled.logits, atol=1e-12)
    np.testing.assert_array_equal(order[shuffled.selected], plain.selected)


def test_chunk_size_does_not_change_the_result():
    scores = noise_scores(15, 64, seed=15)
    whole = cscv_pbo(scores, n_blocks=8, chunk_size=10_000)
    pieces = cscv_pbo(scores, n_blocks=8, chunk_size=7)
    np.testing.assert_array_equal(whole.logits, pieces.logits)


# ---------------------------------------------------------------------------
# Sampled splits
# ---------------------------------------------------------------------------


def test_sampled_splits_are_distinct_and_repeatable():
    scores = noise_scores(30, 160, seed=16)
    first = cscv_pbo(scores, max_splits=500, seed=3)
    again = cscv_pbo(scores, max_splits=500, seed=3)
    other = cscv_pbo(scores, max_splits=500, seed=4)
    assert first.n_splits == 500
    np.testing.assert_array_equal(first.logits, again.logits)
    assert not np.array_equal(first.logits, other.logits)
    full = cscv_pbo(scores)
    assert abs(first.pbo - full.pbo) < 0.1


def test_sampling_almost_every_split_still_gives_distinct_splits():
    # 6 blocks give 20 splits. Asking for 19 makes repeats likely, and each
    # repeat must be thrown away.
    splits = _choose_splits(6, max_splits=19, seed=0)
    assert splits.shape == (19, 3)
    assert len({tuple(row) for row in splits.tolist()}) == 19
    assert all(list(row) == sorted(row) for row in splits.tolist())


def test_max_splits_above_the_total_uses_every_split():
    scores = noise_scores(5, 40, seed=17)
    result = cscv_pbo(scores, n_blocks=6, max_splits=1_000)
    assert result.n_splits == math.comb(6, 3)


def test_huge_split_counts_ask_for_sampling():
    scores = noise_scores(3, 64, seed=18)
    with pytest.raises(ValueError, match="max_splits"):
        cscv_pbo(scores, n_blocks=32)
    assert cscv_pbo(scores, n_blocks=32, max_splits=50).n_splits == 50


# ---------------------------------------------------------------------------
# Summary numbers and input checks
# ---------------------------------------------------------------------------


def test_summary_numbers():
    scores = noise_scores(10, 64, seed=19)
    result = cscv_pbo(scores, n_blocks=8)
    summary = result.to_dict()
    assert summary["pbo"] == result.pbo
    assert summary["n_trials"] == 10 and summary["n_periods"] == 64
    assert "logits" not in summary
    assert len(result.to_dict(include_arrays=True)["logits"]) == result.n_splits
    slope = np.polyfit(result.is_performance, result.oos_performance, 1)[0]
    assert result.degradation_slope == pytest.approx(slope)
    assert result.prob_oos_loss == np.mean(result.oos_performance < 0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_blocks": 5}, "even"),
        ({"n_blocks": 0}, "even"),
        ({"n_blocks": 2.5}, "whole"),
        ({"n_blocks": 64}, "cannot fill"),
        ({"metric": "median"}, "metric"),
        ({"max_splits": 0}, "max_splits"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cscv_pbo(noise_scores(4, 32, seed=20), **kwargs)


@pytest.mark.parametrize(
    "scores",
    [
        np.zeros(10),  # 1-D
        np.zeros((1, 10)),  # one trial
        np.zeros((3, 1)),  # one period
        np.array([[1.0, np.nan, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]),
    ],
)
def test_bad_tables(scores):
    with pytest.raises(ValueError):
        cscv_pbo(scores, n_blocks=2)


def test_halves_too_short_for_a_standard_deviation():
    with pytest.raises(ValueError, match="at least 2 periods"):
        cscv_pbo(noise_scores(3, 3, seed=21), n_blocks=2)
    assert cscv_pbo(noise_scores(3, 3, seed=21), n_blocks=2, metric="mean").n_splits == 2
