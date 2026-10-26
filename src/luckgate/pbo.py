"""Probability of Backtest Overfitting (PBO) with combinatorially symmetric
cross-validation (CSCV).

Bailey, Borwein, López de Prado and Zhu, "The Probability of Backtest
Overfitting", Journal of Computational Finance 20(4), 2017.

The method, in short:

1. Cut the periods into S contiguous blocks (S is even).
2. For every way of choosing S/2 blocks as the in-sample (IS) half, the
   other S/2 blocks are the out-of-sample (OOS) half.
3. On the IS half, pick the trial with the best performance.
4. On the OOS half, find the rank r of that trial among all N trials
   (1 = worst, N = best). Its relative rank is omega = r / (N + 1), and
   its logit is lambda = ln(omega / (1 - omega)).
5. PBO is the share of splits with lambda <= 0: how often the IS winner
   ends up in the bottom half OOS.

Choices the paper leaves open:

- The paper defines PBO as the mass of the logit distribution up to 0.
  A logit of exactly 0 (the IS winner sits exactly on the OOS median,
  which needs an odd N or a tie) is counted as overfit here. That is the
  cautious reading.
- If several trials tie for the best IS score, the first one (the lowest
  row) is selected. Ties in the OOS ranks share the average rank.
- Two scores tie when they are equal, or when both are finite and differ
  by at most `TIE_TOLERANCE` (1e-12) times the size of the values behind
  them. That size is the mean absolute value over the half, divided by
  the standard deviation for the Sharpe ratio, and the larger of the two
  sizes is used. Adding the same numbers in a different order can change
  the last bits of a sum, so a mean that is exactly 0 on paper can come
  out as 1e-17. The tolerance stops such rounding from deciding a tie.
- A half whose values are all equal has zero spread. Its Sharpe ratio is
  +inf, -inf or 0, following the sign of that value.

Speed: a few numbers per block and trial are computed once: the count,
the sum, the sum of absolute values, the mean (with the part of it that
rounding left out) and the sum of squared distances from the block mean.
The mean and standard deviation of any half then come from combining its
S/2 blocks, instead of going back to every period. Trials with the same
values in a half get exactly the same score, so their tie is exact.
Splits are processed in chunks so that memory stays bounded.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from luckgate._validate import as_matrix

METRICS = ("sharpe", "mean")

#: Above this many splits, `cscv_pbo` asks for `max_splits` instead of
#: listing every split.
MAX_ENUMERATED_SPLITS = 1_000_000

#: Two finite scores tie when they are no further apart than this times
#: the size of the values behind them (see the module notes).
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PBOResult:
    """The result of `cscv_pbo`. Arrays have one entry per split."""

    pbo: float
    logits: np.ndarray
    relative_ranks: np.ndarray
    selected: np.ndarray
    is_performance: np.ndarray
    oos_performance: np.ndarray
    n_splits: int
    n_blocks: int
    n_trials: int
    n_periods: int
    metric: str

    @property
    def prob_oos_loss(self) -> float:
        """Share of splits where the IS winner's OOS performance is below 0.

        Only meaningful when 0 is a natural line for the metric, for example
        a gain over a baseline.
        """
        return float(np.mean(self.oos_performance < 0))

    @property
    def degradation_slope(self) -> float:
        """Least-squares slope of OOS performance on IS performance of the
        selected trial. A slope near 0 or below means that a better IS
        score did not lead to a better OOS score. NaN if IS performance does
        not vary across splits.
        """
        x = self.is_performance
        y = self.oos_performance
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < 2:
            return float("nan")
        x_centered = x - x.mean()
        spread = float(np.sum(x_centered * x_centered))
        if spread == 0:
            return float("nan")
        return float(np.sum(x_centered * (y - y.mean())) / spread)

    def to_dict(self, include_arrays: bool = False) -> dict:
        summary = {
            "pbo": self.pbo,
            "n_splits": self.n_splits,
            "n_blocks": self.n_blocks,
            "n_trials": self.n_trials,
            "n_periods": self.n_periods,
            "metric": self.metric,
            "logit_median": float(np.median(self.logits)),
            "prob_oos_loss": self.prob_oos_loss,
            "degradation_slope": self.degradation_slope,
        }
        if include_arrays:
            summary["logits"] = self.logits.tolist()
            summary["relative_ranks"] = self.relative_ranks.tolist()
            summary["selected"] = self.selected.tolist()
            summary["is_performance"] = self.is_performance.tolist()
            summary["oos_performance"] = self.oos_performance.tolist()
        return summary


def cscv_pbo(
    scores,
    *,
    n_blocks: int = 16,
    metric: str = "sharpe",
    max_splits: int | None = None,
    seed: int = 0,
    chunk_size: int = 1024,
) -> PBOResult:
    """Probability of Backtest Overfitting by CSCV.

    Parameters
    ----------
    scores:
        2-D array with one row per trial and one column per period, in time
        order. If your table has time down the rows, pass its transpose.
    n_blocks:
        S, the number of contiguous blocks. It must be even. The paper uses
        16. If the periods do not split evenly, block sizes differ by at most
        one period.
    metric:
        "sharpe" (mean / standard deviation over the half) or "mean". A
        half whose values are all equal gets a Sharpe ratio of +inf, -inf
        or 0, following the sign of that value. Scores that differ only by
        rounding tie (see `TIE_TOLERANCE` in the module notes).
    max_splits:
        If set and smaller than the number of possible splits, use this many
        distinct splits drawn at random (with `seed`) instead of all of them.
    seed:
        Random seed, used only when splits are sampled.
    chunk_size:
        How many splits to process at once. Lower it to save memory.
    """
    matrix = as_matrix(scores)
    n_trials, n_periods = matrix.shape
    n_blocks = _check_blocks(n_blocks, n_periods, metric)
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive whole number; got {chunk_size!r}")
    chunk_size = int(chunk_size)

    # One row per split: the in-sample blocks, and the out-of-sample blocks.
    train_blocks = _choose_splits(n_blocks, max_splits, seed)
    test_blocks = _other_blocks(train_blocks, n_blocks)
    n_splits = train_blocks.shape[0]

    sums = _block_sums(matrix, n_blocks)

    logits = np.empty(n_splits)
    relative_ranks = np.empty(n_splits)
    selected = np.empty(n_splits, dtype=int)
    is_performance = np.empty(n_splits)
    oos_performance = np.empty(n_splits)

    for start in range(0, n_splits, chunk_size):
        stop = min(start + chunk_size, n_splits)
        train, train_size = _performance(train_blocks[start:stop], sums, metric)
        test, test_size = _performance(test_blocks[start:stop], sums, metric)
        rows = np.arange(stop - start)
        # The IS winner of each split: the first row that ties with the best.
        top = np.argmax(train, axis=1)
        top_score = train[rows, top][:, None]
        top_size = train_size[rows, top][:, None]
        best = np.argmax(_tied(train, train_size, top_score, top_size), axis=1)
        best_oos = test[rows, best][:, None]

        # OOS rank of the IS winner: 1 = worst, N = best. Ties share the
        # average rank, so a tie never counts as a win.
        tied = _tied(test, test_size, best_oos, test_size[rows, best][:, None])
        n_lower = np.sum((test < best_oos) & ~tied, axis=1)
        n_tied_others = np.sum(tied, axis=1) - 1
        rank = 1.0 + n_lower + 0.5 * n_tied_others
        omega = rank / (n_trials + 1.0)

        logits[start:stop] = np.log(omega / (1.0 - omega))
        relative_ranks[start:stop] = omega
        selected[start:stop] = best
        is_performance[start:stop] = train[rows, best]
        oos_performance[start:stop] = best_oos[:, 0]

    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        logits=logits,
        relative_ranks=relative_ranks,
        selected=selected,
        is_performance=is_performance,
        oos_performance=oos_performance,
        n_splits=int(n_splits),
        n_blocks=n_blocks,
        n_trials=int(n_trials),
        n_periods=int(n_periods),
        metric=metric,
    )


def _check_blocks(n_blocks, n_periods: int, metric: str) -> int:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}; got {metric!r}")
    if isinstance(n_blocks, bool) or int(n_blocks) != n_blocks:
        raise ValueError(f"n_blocks must be a whole number; got {n_blocks!r}")
    n_blocks = int(n_blocks)
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be an even number of at least 2; got {n_blocks}")
    if n_periods < n_blocks:
        raise ValueError(f"{n_periods} periods cannot fill {n_blocks} blocks; use fewer blocks")
    # Each half holds n_blocks / 2 blocks. A standard deviation needs at
    # least two periods in a half.
    smallest_half = (n_blocks // 2) * (n_periods // n_blocks)
    if metric == "sharpe" and smallest_half < 2:
        raise ValueError(
            "each half needs at least 2 periods for the 'sharpe' metric; "
            "use more periods or more blocks"
        )
    return n_blocks


def _choose_splits(n_blocks: int, max_splits: int | None, seed: int) -> np.ndarray:
    """Return the in-sample block indices of each split, one split per row."""
    half = n_blocks // 2
    total = math.comb(n_blocks, half)
    if max_splits is not None:
        if isinstance(max_splits, bool) or int(max_splits) != max_splits or max_splits < 1:
            raise ValueError(f"max_splits must be a positive whole number; got {max_splits!r}")
        max_splits = int(max_splits)

    if max_splits is None or max_splits >= total:
        if total > MAX_ENUMERATED_SPLITS:
            raise ValueError(
                f"{n_blocks} blocks give {total} splits; set max_splits below that "
                "to sample some of them"
            )
        return np.array(list(itertools.combinations(range(n_blocks), half)), dtype=int)

    # Draw distinct splits at random. Sorting each draw lets the set spot
    # repeats.
    rng = np.random.default_rng(seed)
    chosen: set[tuple[int, ...]] = set()
    ordered: list[tuple[int, ...]] = []
    while len(ordered) < max_splits:
        draw = tuple(sorted(int(b) for b in rng.choice(n_blocks, size=half, replace=False)))
        if draw not in chosen:
            chosen.add(draw)
            ordered.append(draw)
    return np.array(ordered, dtype=int)


def _other_blocks(train_blocks: np.ndarray, n_blocks: int) -> np.ndarray:
    """For each split, the blocks that are not in-sample, in order."""
    rows = []
    for row in train_blocks.tolist():
        in_sample = set(row)
        rows.append([b for b in range(n_blocks) if b not in in_sample])
    return np.array(rows, dtype=int)


@dataclass(frozen=True)
class _BlockSums:
    """What `_performance` needs to know about each block, computed once.

    `count` has one entry per block. The other arrays have one row per
    block and one column per trial. Each one depends only on the values in
    its block, so trials with the same values in a half get exactly the
    same score.

    `total` is the plain sum of the values: the mean of a half comes from
    it. The spread is built from `squares`, the sum of squared distances
    from the block's own mean, and from the gaps between block means. A
    block mean is held as `mean` plus `residual`, the part that rounding
    left out of `mean`, so the gaps stay accurate when the mean is large
    next to the spread.
    """

    count: np.ndarray  # periods in each block
    total: np.ndarray  # sum of the values
    absolute: np.ndarray  # sum of the absolute values
    mean: np.ndarray  # mean of the values (exactly the value of a flat block)
    residual: np.ndarray  # mean distance from `mean`, a rounding error
    squares: np.ndarray  # sum of squared distances from `mean`
    level: np.ndarray  # the value of a flat block, NaN if the values differ


def _block_sums(matrix: np.ndarray, n_blocks: int) -> _BlockSums:
    """Per-block period counts, sums, means and spreads of every trial."""
    count, total, absolute, mean, residual, squares, level = [], [], [], [], [], [], []
    for block in np.array_split(np.arange(matrix.shape[1]), n_blocks):
        values = matrix[:, block]
        block_total = values.sum(axis=1)
        flat = values.max(axis=1) == values.min(axis=1)
        # The mean of a flat block is its value. Dividing the sum by the
        # count could be off in the last bit.
        block_mean = np.where(flat, values[:, 0], block_total / len(block))
        distances = values - block_mean[:, None]
        count.append(float(len(block)))
        total.append(block_total)
        absolute.append(np.abs(values).sum(axis=1))
        mean.append(block_mean)
        residual.append(distances.sum(axis=1) / len(block))
        squares.append((distances * distances).sum(axis=1))
        level.append(np.where(flat, block_mean, np.nan))
    return _BlockSums(
        count=np.array(count),
        total=np.vstack(total),
        absolute=np.vstack(absolute),
        mean=np.vstack(mean),
        residual=np.vstack(residual),
        squares=np.vstack(squares),
        level=np.vstack(level),
    )


def _performance(
    chosen: np.ndarray, sums: _BlockSums, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Performance of every trial on the blocks listed in `chosen`, and the
    size of the values behind it, which sets how close two scores must be
    to tie (see `_tied`).

    `chosen` has one row per split, holding block indices. Both results
    have one row per split and one column per trial.
    """
    n_splits, blocks_per_half = chosen.shape
    n_trials = sums.total.shape[1]
    gathered = np.empty((n_splits, n_trials))  # reused for each block's numbers

    # Add up the counts and sums of the chosen blocks, one block at a time.
    count = np.zeros(n_splits)
    total = np.zeros((n_splits, n_trials))
    absolute = np.zeros((n_splits, n_trials))
    for position in range(blocks_per_half):
        blocks = chosen[:, position]  # one block index per split
        count += sums.count[blocks]
        total += _gather(sums.total, blocks, gathered)
        absolute += _gather(sums.absolute, blocks, gathered)
    mean = total / count[:, None]
    typical = absolute / count[:, None]  # the mean absolute value
    if metric == "mean":
        return mean, typical

    # Squared distances from the mean of the half: each block's own, plus
    # count * gap**2 for the gap between the block mean and the half mean.
    # The residual adds the digits that rounding left out of the block
    # mean. The half mean is rounded too, but an error there shifts every
    # gap alike and only adds count * error**2 in total.
    squares = np.zeros((n_splits, n_trials))
    gap = np.empty((n_splits, n_trials))
    for position in range(blocks_per_half):
        blocks = chosen[:, position]
        squares += _gather(sums.squares, blocks, gathered)
        _gather(sums.mean, blocks, gap)
        gap -= mean
        gap += _gather(sums.residual, blocks, gathered)
        gap *= gap
        gap *= sums.count[blocks][:, None]
        squares += gap
    spread = np.sqrt(squares / (count[:, None] - 1.0))

    # A half is flat if every block in it is flat at the same value. Most
    # data has no flat block, and then the loop is skipped.
    level = sums.level[chosen[:, 0]]
    flat = ~np.isnan(level)
    if flat.any():
        for position in range(1, blocks_per_half):
            flat &= sums.level[chosen[:, position]] == level

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = mean / spread
        sharpe_size = typical / spread

    # A half with zero spread gets a Sharpe ratio of +inf, -inf or 0,
    # following the sign of its value. A spread that is not zero but too
    # small for floating point to hold follows the sign of the mean.
    zero_spread = flat | (spread == 0)
    if zero_spread.any():
        sign = np.where(flat, level, mean)
        flat_value = np.where(sign > 0, np.inf, np.where(sign < 0, -np.inf, 0.0))
        sharpe = np.where(zero_spread, flat_value, sharpe)
        sharpe_size = np.where(zero_spread, 0.0, sharpe_size)
    return sharpe, sharpe_size


def _gather(per_block: np.ndarray, blocks: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Row `blocks[k]` of `per_block` for every split k, written into `out`.

    With its default mode="raise", `np.take` writes to a temporary array
    first. mode="clip" writes into `out` directly, which is much faster.
    The block indices are always in range, so clipping never changes one.
    """
    return np.take(per_block, blocks, axis=0, out=out, mode="clip")


def _tied(
    values: np.ndarray, values_size: np.ndarray, target: np.ndarray, target_size: np.ndarray
) -> np.ndarray:
    """True where `values` ties with `target`: equal, or both finite and no
    more than `TIE_TOLERANCE` times the larger of the two sizes apart."""
    limit = TIE_TOLERANCE * np.maximum(values_size, target_size)
    with np.errstate(invalid="ignore"):
        close = np.abs(values - target) <= limit
    return (values == target) | (close & np.isfinite(values) & np.isfinite(target))
