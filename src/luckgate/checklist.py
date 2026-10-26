"""A configurable pass/fail checklist for the winner of a search.

Each check is a plain function that takes an `Evidence` object and returns
a `CheckResult`. A `Checklist` runs the checks named in its `GateConfig`,
then any extra check functions you give it, and returns a `Verdict`.

The default limits are opinions, not results from the papers. Change them
to suit your problem. PASS only means that none of the checks found that
luck or overfitting explains the winner, for the inputs given. It is not a
forecast.
"""

from __future__ import annotations

import json
import math
import numbers
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from luckgate._validate import (
    as_matrix,
    as_series,
    finite_number,
    n_trials_value,
    sr_std_value,
)
from luckgate.pbo import METRICS, cscv_pbo
from luckgate.sharpe import dsr_report, trial_sharpe_std

#: Built-in check names, in the order they run.
KNOWN_CHECKS = ("min_periods", "dsr", "pbo", "segments")

#: The checks a default `GateConfig` runs.
DEFAULT_CHECKS = ("min_periods", "dsr", "pbo")

VERDICT_NOTE = (
    "PASS means no check found that luck or overfitting explains the winner, "
    "for these inputs. It is not a forecast."
)


@dataclass(frozen=True)
class GateConfig:
    """Which checks to run and their limits. All limits are opinionated defaults.

    `clip_positive_skew` and `sr_std_floor` are cautious heuristics that are
    not part of the papers. They are off by default.

    Every limit must be a finite number, and `clip_positive_skew` must be
    True or False. A NaN limit would let a check pass whatever the value,
    and a string such as "0.9" or "false" in a JSON file is a mistake, so
    both are refused.
    """

    checks: tuple[str, ...] = DEFAULT_CHECKS
    # min_periods: the winner's series must be at least this long.
    min_periods: int = 30
    # dsr: the Deflated Sharpe Ratio must be at least this.
    min_dsr: float = 0.95
    # pbo: the probability of backtest overfitting must be at most this.
    # The PBO paper mentions 0.05 as a customary line; 0.2 is looser.
    max_pbo: float = 0.2
    pbo_blocks: int = 16
    pbo_metric: str = "sharpe"
    pbo_max_splits: int | None = None
    pbo_seed: int = 0
    # segments: in every segment the winner's mean must be at least
    # segment_floor, and every segment needs min_segment_periods periods.
    segment_floor: float = 0.0
    min_segment_periods: int = 20
    # Heuristics (not in the papers).
    clip_positive_skew: bool = False
    sr_std_floor: float | None = None

    def __post_init__(self):
        if isinstance(self.checks, str):
            # tuple("dsr") would be ("d", "s", "r"), so say what went wrong.
            raise ValueError(f"checks must be a list of names, not the string {self.checks!r}")
        try:
            checks = tuple(self.checks)
        except TypeError:
            raise ValueError(f"checks must be a list of names; got {self.checks!r}") from None
        object.__setattr__(self, "checks", checks)
        unknown = [name for name in checks if name not in KNOWN_CHECKS]
        if unknown:
            raise ValueError(f"unknown checks {unknown}; known checks are {KNOWN_CHECKS}")
        if len(set(checks)) != len(checks):
            raise ValueError(f"checks has repeated names: {checks}")

        for name in ("min_dsr", "max_pbo", "segment_floor"):
            object.__setattr__(self, name, _config_number(getattr(self, name), name))
        for name in ("min_periods", "pbo_blocks", "pbo_seed", "min_segment_periods"):
            object.__setattr__(self, name, _config_whole_number(getattr(self, name), name))
        if self.pbo_max_splits is not None:
            splits = _config_whole_number(self.pbo_max_splits, "pbo_max_splits")
            object.__setattr__(self, "pbo_max_splits", splits)
        if self.sr_std_floor is not None:
            floor = sr_std_value(_config_number(self.sr_std_floor, "sr_std_floor"))
            object.__setattr__(self, "sr_std_floor", floor)
        if not isinstance(self.clip_positive_skew, (bool, np.bool_)):
            raise ValueError(
                f"clip_positive_skew must be True or False; got {self.clip_positive_skew!r}"
            )
        object.__setattr__(self, "clip_positive_skew", bool(self.clip_positive_skew))

        if not 0.5 < self.min_dsr < 1.0:
            raise ValueError(f"min_dsr must be between 0.5 and 1; got {self.min_dsr}")
        if not 0.0 <= self.max_pbo <= 1.0:
            raise ValueError(f"max_pbo must be between 0 and 1; got {self.max_pbo}")
        if self.min_periods < 3:
            raise ValueError(f"min_periods must be at least 3; got {self.min_periods}")
        if self.min_segment_periods < 1:
            raise ValueError(
                f"min_segment_periods must be at least 1; got {self.min_segment_periods}"
            )
        if self.pbo_metric not in METRICS:
            raise ValueError(f"pbo_metric must be one of {METRICS}; got {self.pbo_metric!r}")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["checks"] = list(self.checks)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> GateConfig:
        """Build a config from a dict, for example one read from a JSON file.
        Unknown keys are an error, so a typo cannot be ignored silently."""
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown config keys {unknown}; allowed keys are {sorted(allowed)}")
        # __post_init__ checks every value, so a value of the wrong type, such
        # as the string "false" for a True/False field, is refused here too.
        return cls(**data)


def _config_number(value, name: str) -> float:
    """A limit in a `GateConfig` must be a real, finite number.

    True and False are refused, as everywhere in luckgate, and so are
    strings: "0.9" in a JSON file is a mistake, not a number.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number; got {value!r}")
    return finite_number(value, name)


def _config_whole_number(value, name: str) -> int:
    """Like `_config_number`, for a field that must be a whole number."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    number = _config_number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number; got {value!r}")
    return int(number)


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check.

    `rule` says how `value` was compared with `threshold`, for example
    "value >= threshold".
    """

    name: str
    passed: bool
    value: float | None
    threshold: float | None
    rule: str
    message: str
    details: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "rule": self.rule,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class Evidence:
    """Everything a check may look at. Built by `Checklist.run`.

    `winner` is the per-period series of the selected trial. `scores` has
    one row per stored trial and one column per period. `n_trials` is the
    honest number of trials, which may be larger than the number of rows.
    """

    winner: np.ndarray
    n_trials: float
    config: GateConfig
    scores: np.ndarray | None = None
    winner_row: int | None = None
    sr_std: float | None = None
    sr_std_source: str | None = None
    segments: np.ndarray | None = None


@dataclass(frozen=True)
class Verdict:
    """The result of a checklist run. `passed` is True only if every check passed."""

    passed: bool
    checks: tuple[CheckResult, ...]
    inputs: dict
    config: dict

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def failed(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]

    def __getitem__(self, name: str) -> CheckResult:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "verdict": self.status,
            "failed_checks": self.failed,
            "checks": [check.to_dict() for check in self.checks],
            "inputs": self.inputs,
            "config": self.config,
            "note": VERDICT_NOTE,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """The report as JSON. Infinite or undefined numbers are written as null."""
        return json.dumps(json_safe(self.to_dict()), indent=indent)

    def __str__(self) -> str:
        parts = ", ".join(f"{check.name} {check.status}" for check in self.checks)
        return f"Verdict: {self.status} ({parts})"


CheckFunction = Callable[[Evidence], CheckResult]


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------


def check_min_periods(evidence: Evidence) -> CheckResult:
    """The winner's series must be long enough to say anything at all."""
    n_periods = int(evidence.winner.size)
    limit = evidence.config.min_periods
    passed = n_periods >= limit
    return CheckResult(
        name="min_periods",
        passed=passed,
        value=float(n_periods),
        threshold=float(limit),
        rule="value >= threshold",
        message=f"{n_periods} periods; at least {limit} are required",
    )


def check_dsr(evidence: Evidence) -> CheckResult:
    """The Deflated Sharpe Ratio of the winner must be high enough."""
    config = evidence.config
    report = dsr_report(
        evidence.winner,
        n_trials=evidence.n_trials,
        sr_std=evidence.sr_std,
        confidence=config.min_dsr,
        clip_positive_skew=config.clip_positive_skew,
    )
    passed = report.dsr >= config.min_dsr
    message = (
        f"DSR {report.dsr:.4f}: winner Sharpe {report.sr:.4f} per period vs "
        f"{report.expected_max_sharpe:.4f} expected from the best of "
        f"{evidence.n_trials:g} noise trials"
    )
    return CheckResult(
        name="dsr",
        passed=passed,
        value=report.dsr,
        threshold=config.min_dsr,
        rule="value >= threshold",
        message=message,
        details=report.to_dict(),
    )


def check_pbo(evidence: Evidence) -> CheckResult:
    """The probability of backtest overfitting of the whole search must be low."""
    config = evidence.config
    result = cscv_pbo(
        evidence.scores,
        n_blocks=config.pbo_blocks,
        metric=config.pbo_metric,
        max_splits=config.pbo_max_splits,
        seed=config.pbo_seed,
    )
    passed = result.pbo <= config.max_pbo
    message = (
        f"PBO {result.pbo:.4f}: the in-sample best ranked in the bottom half "
        f"out of sample in {result.pbo:.1%} of {result.n_splits} splits"
    )
    return CheckResult(
        name="pbo",
        passed=passed,
        value=result.pbo,
        threshold=config.max_pbo,
        rule="value <= threshold",
        message=message,
        details=result.to_dict(),
    )


def check_segments(evidence: Evidence) -> CheckResult:
    """The winner must not depend on one segment of the data.

    Segments are labels such as weekday or weekend, user group or season,
    one label per period. In every segment the winner's mean must be at
    least `segment_floor`, and every segment must have at least
    `min_segment_periods` periods, so that a few lucky periods cannot
    carry it.
    """
    config = evidence.config
    labels = evidence.segments
    winner = evidence.winner

    # Keep the labels in the order they first appear.
    order: list = []
    for label in labels.tolist():
        if label not in order:
            order.append(label)

    details = {}
    problems = []
    lowest_mean = math.inf
    for label in order:
        values = winner[labels == label]
        mean = float(values.mean())
        lowest_mean = min(lowest_mean, mean)
        if values.size < config.min_segment_periods:
            status = "FAIL"
            problems.append(f"{label}: only {values.size} periods (< {config.min_segment_periods})")
        elif mean < config.segment_floor:
            status = "FAIL"
            problems.append(f"{label}: mean {mean:.4f} < {config.segment_floor:g}")
        else:
            status = "PASS"
        details[str(label)] = {"n_periods": int(values.size), "mean": mean, "status": status}

    passed = not problems
    if passed:
        message = f"all {len(order)} segments have mean >= {config.segment_floor:g}"
    else:
        message = "; ".join(problems)
    return CheckResult(
        name="segments",
        passed=passed,
        value=lowest_mean,
        threshold=config.segment_floor,
        rule="lowest segment mean >= threshold, with enough periods in every segment",
        message=message,
        details=details,
    )


BUILT_IN_CHECKS: dict[str, CheckFunction] = {
    "min_periods": check_min_periods,
    "dsr": check_dsr,
    "pbo": check_pbo,
    "segments": check_segments,
}


# ---------------------------------------------------------------------------
# The checklist
# ---------------------------------------------------------------------------


class Checklist:
    """Run the checks named in `config.checks`, then each function in `extra_checks`.

    An extra check is any function that takes an `Evidence` and returns a
    `CheckResult`. A checklist with no checks at all is refused, because
    its verdict would be PASS without testing anything.
    """

    def __init__(
        self,
        config: GateConfig | None = None,
        extra_checks: Sequence[CheckFunction] = (),
    ):
        self.config = config if config is not None else GateConfig()
        self.extra_checks = tuple(extra_checks)
        _refuse_an_empty_checklist(self.config, self.extra_checks)

    def run(
        self,
        winner=None,
        *,
        n_trials: float,
        scores=None,
        sr_std: float | None = None,
        segments=None,
    ) -> Verdict:
        """Check one winner.

        Parameters
        ----------
        winner:
            The selected trial's per-period series. If left out, the row of
            `scores` with the highest Sharpe ratio is used.
        n_trials:
            The honest number of trials: every configuration you tried,
            including the ones you did not keep.
        scores:
            One row per stored trial and one column per period. Needed for
            the "pbo" check, and used to measure `sr_std` if it is not given.
        sr_std:
            Standard deviation of all trials' per-period Sharpe ratios. If
            it is below the value measured from `scores`, a UserWarning
            says so: a smaller `sr_std` shrinks the DSR penalty.
        segments:
            One label per period of `winner`. Needed for the "segments" check.
        """
        _refuse_an_empty_checklist(self.config, self.extra_checks)
        evidence = self.gather(
            winner, n_trials=n_trials, scores=scores, sr_std=sr_std, segments=segments
        )
        results = [BUILT_IN_CHECKS[name](evidence) for name in self.config.checks]
        for extra in self.extra_checks:
            result = extra(evidence)
            if not isinstance(result, CheckResult):
                raise TypeError(
                    f"extra check {extra!r} returned {type(result).__name__}, not a CheckResult"
                )
            # verdict["name"] and the list of failed checks need unique names.
            if result.name in [earlier.name for earlier in results]:
                raise ValueError(f"two checks are named {result.name!r}; give each a new name")
            results.append(result)

        inputs = {
            "n_trials": evidence.n_trials,
            "n_periods": int(evidence.winner.size),
            "winner_row": evidence.winner_row,
            "n_rows_in_scores": None if evidence.scores is None else int(evidence.scores.shape[0]),
            "sr_std": evidence.sr_std,
            "sr_std_source": evidence.sr_std_source,
        }
        return Verdict(
            passed=all(result.passed for result in results),
            checks=tuple(results),
            inputs=inputs,
            config=self.config.to_dict(),
        )

    def gather(self, winner=None, *, n_trials, scores=None, sr_std=None, segments=None) -> Evidence:
        """Check the inputs and build the `Evidence` that the checks read."""
        config = self.config
        checks = config.checks
        n_trials = n_trials_value(n_trials)

        matrix = None if scores is None else as_matrix(scores)
        if matrix is not None and n_trials < matrix.shape[0]:
            raise ValueError(
                f"n_trials ({n_trials:g}) is smaller than the number of trials in scores "
                f"({matrix.shape[0]}). n_trials must count every trial you ran."
            )

        winner_row = None
        if winner is None:
            if matrix is None:
                raise ValueError("give the winner's series, or scores to pick the winner from")
            winner_row = best_row(matrix)
            series = matrix[winner_row].copy()
        else:
            series = as_series(winner, name="winner", min_length=3)
            if matrix is not None:
                winner_row = find_row(matrix, series)

        sr_std_source = None
        sr_std_given = sr_std is not None
        if sr_std_given:
            sr_std = sr_std_value(sr_std)
            sr_std_source = "given"
        elif matrix is not None and "dsr" in checks:
            sr_std = trial_sharpe_std(matrix)
            sr_std_source = "scores"
        if sr_std is not None and config.sr_std_floor is not None and sr_std < config.sr_std_floor:
            sr_std = float(config.sr_std_floor)
            sr_std_source = f"{sr_std_source}, raised to sr_std_floor"
        if sr_std_given and matrix is not None and "dsr" in checks:
            _warn_if_below_measured(sr_std, matrix)

        labels = None
        if segments is not None:
            labels = np.asarray(segments)
            if labels.ndim != 1 or labels.size != series.size:
                raise ValueError(
                    f"segments must have one label per period ({series.size}); "
                    f"got shape {labels.shape}"
                )

        if "dsr" in checks and sr_std is None:
            raise ValueError("the 'dsr' check needs sr_std, or scores to measure it from")
        if "pbo" in checks and matrix is None:
            raise ValueError(
                "the 'pbo' check needs scores; pass scores or remove 'pbo' from checks"
            )
        if "segments" in checks and labels is None:
            raise ValueError("the 'segments' check needs segments")
        if labels is not None and "segments" not in checks:
            raise ValueError("segments were given, but 'segments' is not in config.checks")

        return Evidence(
            winner=series,
            n_trials=n_trials,
            config=config,
            scores=matrix,
            winner_row=winner_row,
            sr_std=sr_std,
            sr_std_source=sr_std_source,
            segments=labels,
        )


def _refuse_an_empty_checklist(config: GateConfig, extra_checks: Sequence) -> None:
    """With no checks at all, every verdict would be PASS without testing anything."""
    if not config.checks and not extra_checks:
        raise ValueError(
            "there are no checks to run: config.checks is empty and there are no "
            "extra_checks, so the verdict would be PASS without testing anything"
        )


def _warn_if_below_measured(sr_std: float, scores: np.ndarray) -> None:
    """Warn if a given `sr_std` is below the spread measured from `scores`.

    A smaller `sr_std` lowers the expected maximum Sharpe ratio, so the DSR
    penalty for the search shrinks or vanishes. A value below what the
    scores show is allowed, because the rows may not be every trial, but
    it should never happen silently.
    """
    try:
        measured = trial_sharpe_std(scores)
    except ValueError:
        return  # a flat row has no Sharpe ratio, so there is nothing to compare with
    if sr_std < measured:
        warnings.warn(
            f"sr_std={sr_std:g} is below {measured:g}, the spread of the Sharpe ratios in "
            "scores. A smaller sr_std shrinks the DSR penalty. Leave sr_std out to use "
            "the measured value.",
            UserWarning,
            stacklevel=4,
        )


def best_row(scores: np.ndarray) -> int:
    """Index of the row with the highest per-period Sharpe ratio."""
    spreads = scores.std(axis=1, ddof=1)
    if np.any(spreads == 0):
        raise ValueError("some rows of scores have zero spread; their Sharpe ratio is undefined")
    return int(np.argmax(scores.mean(axis=1) / spreads))


def find_row(scores: np.ndarray, series: np.ndarray) -> int | None:
    """Index of the first row of `scores` equal to `series`, or None."""
    if scores.shape[1] != series.size:
        return None
    for index in range(scores.shape[0]):
        if np.array_equal(scores[index], series):
            return index
    return None


def json_safe(value):
    """Turn a report into plain JSON types. Infinite or NaN floats become None."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value
