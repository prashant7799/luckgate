"""GateConfig, Checklist, CheckResult and Verdict."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from synthetic import best_row, noise_scores

from luckgate import (
    Checklist,
    CheckResult,
    Evidence,
    GateConfig,
    Verdict,
    dsr_report,
    trial_sharpe_std,
)


def planted(seed: int = 1) -> np.ndarray:
    scores = noise_scores(60, 200, seed)
    scores[4] += 0.5
    return scores


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------


def test_default_config():
    config = GateConfig()
    assert config.checks == ("min_periods", "dsr", "pbo")
    assert config.clip_positive_skew is False
    assert config.sr_std_floor is None


def test_config_round_trips_through_a_dict():
    config = GateConfig(checks=["dsr"], min_dsr=0.9, sr_std_floor=0.05)
    assert config.checks == ("dsr",)
    assert GateConfig.from_dict(config.to_dict()) == config
    assert json.loads(json.dumps(config.to_dict()))["checks"] == ["dsr"]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"checks": ("dsr", "luck")}, "unknown checks"),
        ({"checks": "dsr"}, "list of names"),  # a string, not a list of names
        ({"checks": ("dsr", "dsr")}, "repeated"),
        ({"min_dsr": 0.5}, "min_dsr"),
        ({"min_dsr": 1.0}, "min_dsr"),
        ({"max_pbo": 1.5}, "max_pbo"),
        ({"min_periods": 2}, "min_periods"),
        ({"min_segment_periods": 0}, "min_segment_periods"),
        ({"pbo_metric": "median"}, "pbo_metric"),
        ({"sr_std_floor": -1.0}, "sr_std"),
        ({"checks": 5}, "list of names"),
        # A NaN limit would let a check pass whatever the value.
        ({"segment_floor": float("nan")}, "segment_floor must be finite"),
        ({"segment_floor": -float("inf")}, "segment_floor must be finite"),
        ({"min_dsr": float("nan")}, "min_dsr must be finite"),
        ({"max_pbo": float("nan")}, "max_pbo must be finite"),
        ({"min_periods": float("nan")}, "min_periods must be finite"),
        ({"min_periods": float("inf")}, "min_periods must be finite"),
        ({"min_segment_periods": float("nan")}, "min_segment_periods must be finite"),
        ({"sr_std_floor": float("inf")}, "sr_std_floor must be finite"),
        ({"pbo_max_splits": float("nan")}, "pbo_max_splits must be finite"),
        # Fields that count something must be whole numbers.
        ({"min_periods": 30.5}, "whole number"),
        ({"pbo_blocks": 2.5}, "whole number"),
        ({"pbo_seed": 0.5}, "whole number"),
        # Strings and True/False are not numbers, and 0/1 or "false" are not True/False.
        ({"min_dsr": "0.9"}, "min_dsr must be a number"),
        ({"segment_floor": True}, "segment_floor must be a number"),
        ({"min_periods": True}, "min_periods must be a number"),
        ({"sr_std_floor": "0.05"}, "sr_std_floor must be a number"),
        ({"clip_positive_skew": "false"}, "clip_positive_skew must be True or False"),
        ({"clip_positive_skew": 1}, "clip_positive_skew must be True or False"),
    ],
)
def test_bad_config(values, message):
    with pytest.raises(ValueError, match=message):
        GateConfig(**values)


def test_config_from_dict_rejects_typos():
    with pytest.raises(ValueError, match="min_dsrr"):
        GateConfig.from_dict({"min_dsrr": 0.9})


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"clip_positive_skew": "false"}', "clip_positive_skew must be True or False"),
        ('{"segment_floor": NaN}', "segment_floor must be finite"),  # json.loads reads NaN
        ('{"min_periods": NaN}', "min_periods must be finite"),
        ('{"max_pbo": "0.2"}', "max_pbo must be a number"),
        ('{"checks": "dsr"}', "list of names"),
    ],
)
def test_config_from_dict_checks_types(text, message):
    """A JSON file can hold any type. The string "false" is truthy, and NaN
    compares False with everything, so both would change what the checks do."""
    with pytest.raises(ValueError, match=message):
        GateConfig.from_dict(json.loads(text))


def test_config_from_dict_accepts_whole_floats_and_normalises_types():
    config = GateConfig.from_dict({"min_periods": 30.0, "pbo_blocks": 16.0, "min_dsr": 0.95})
    assert config == GateConfig()
    assert type(config.min_periods) is int and type(config.pbo_blocks) is int
    numpy_values = GateConfig(min_dsr=np.float64(0.9), pbo_seed=np.int64(3))
    assert type(numpy_values.min_dsr) is float and type(numpy_values.pbo_seed) is int
    json.dumps(numpy_values.to_dict())  # plain Python types, so JSON can write them


# ---------------------------------------------------------------------------
# Running the checklist
# ---------------------------------------------------------------------------


def test_winner_defaults_to_the_best_row_by_sharpe():
    scores = planted()
    verdict = Checklist().run(scores=scores, n_trials=60)
    assert verdict.inputs["winner_row"] == best_row(scores) == 4
    assert verdict.inputs["sr_std_source"] == "scores"
    assert verdict.inputs["n_rows_in_scores"] == 60


def test_a_given_winner_is_found_in_the_scores():
    scores = planted()
    verdict = Checklist().run(scores[9], scores=scores, n_trials=60)
    assert verdict.inputs["winner_row"] == 9
    outside = Checklist().run(scores[9] + 1.0, scores=scores, n_trials=60)
    assert outside.inputs["winner_row"] is None


def test_dsr_check_uses_dsr_report():
    scores = planted()
    verdict = Checklist().run(scores=scores, n_trials=500)
    expected = dsr_report(scores[4], n_trials=500, sr_std=verdict.inputs["sr_std"])
    assert verdict["dsr"].value == expected.dsr
    assert verdict["dsr"].details == expected.to_dict()


def test_honest_n_trials_must_cover_every_stored_row():
    scores = planted()
    with pytest.raises(ValueError, match="n_trials"):
        Checklist().run(scores=scores, n_trials=10)


def test_n_trials_true_is_not_one_trial():
    with pytest.raises(ValueError, match="n_trials"):
        Checklist(GateConfig(checks=("dsr",))).run(np.arange(40.0), n_trials=True, sr_std=0.1)


def test_more_trials_than_rows_is_allowed_and_stricter():
    scores = planted()
    few = Checklist().run(scores=scores, n_trials=60)
    many = Checklist().run(scores=scores, n_trials=60_000)
    assert many["dsr"].value < few["dsr"].value


@pytest.mark.parametrize(
    ("config", "kwargs", "message"),
    [
        (GateConfig(), {"n_trials": 5}, "winner's series"),
        (
            GateConfig(checks=("dsr",)),
            {"winner": np.ones(40) + np.arange(40), "n_trials": 5},
            "sr_std",
        ),
        (GateConfig(checks=("pbo",)), {"winner": np.arange(40.0), "n_trials": 5}, "scores"),
        (GateConfig(checks=("segments",)), {"winner": np.arange(40.0), "n_trials": 5}, "segments"),
        (
            GateConfig(checks=("min_periods",)),
            {"winner": np.arange(40.0), "n_trials": 5, "segments": ["a"] * 40},
            "not in",
        ),
        (
            GateConfig(checks=("segments",)),
            {"winner": np.arange(40.0), "n_trials": 5, "segments": ["a"] * 39},
            "one label",
        ),
    ],
)
def test_missing_or_mismatched_inputs(config, kwargs, message):
    with pytest.raises(ValueError, match=message):
        Checklist(config).run(**kwargs)


def test_sr_std_floor_is_reported():
    config = GateConfig(checks=("dsr",), sr_std_floor=0.2)
    verdict = Checklist(config).run(
        np.random.default_rng(0).normal(size=100), n_trials=10, sr_std=0.01
    )
    assert verdict.inputs["sr_std"] == 0.2
    assert verdict.inputs["sr_std_source"] == "given, raised to sr_std_floor"


def test_an_empty_checklist_is_refused():
    """With no checks, all([]) is True and the verdict would be PASS without
    testing anything."""
    with pytest.raises(ValueError, match="no checks to run"):
        Checklist(GateConfig(checks=()))
    with pytest.raises(ValueError, match="no checks to run"):
        Checklist(GateConfig.from_dict({"checks": []}))


def test_only_extra_checks_is_not_empty():
    def positive_mean(evidence: Evidence) -> CheckResult:
        mean = float(evidence.winner.mean())
        return CheckResult("positive_mean", mean > 0, mean, 0.0, "value > threshold", "ok")

    checklist = Checklist(GateConfig(checks=()), extra_checks=[positive_mean])
    verdict = checklist.run(np.arange(1.0, 41.0), n_trials=1)
    assert [check.name for check in verdict.checks] == ["positive_mean"]
    checklist.extra_checks = ()  # emptied after it was built
    with pytest.raises(ValueError, match="no checks to run"):
        checklist.run(np.arange(1.0, 41.0), n_trials=1)


def test_a_given_sr_std_below_the_measured_one_warns():
    """A small sr_std shrinks the DSR penalty. It is allowed (the rows may
    not be every trial), but it must not happen silently."""
    scores = planted()
    measured = trial_sharpe_std(scores)
    with pytest.warns(UserWarning, match="below") as caught:
        verdict = Checklist().run(scores=scores, n_trials=60, sr_std=measured / 10)
    assert verdict.inputs["sr_std"] == measured / 10
    assert verdict.inputs["sr_std_source"] == "given"
    assert caught[0].filename == __file__  # it points at the caller's line

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        Checklist().run(scores=scores, n_trials=60, sr_std=measured)
        Checklist().run(scores=scores, n_trials=60, sr_std=2 * measured)
        Checklist().run(scores=scores, n_trials=60)
        Checklist(GateConfig(checks=("pbo",))).run(scores=scores, n_trials=60, sr_std=0.0)


def test_only_the_configured_checks_run_in_order():
    config = GateConfig(checks=("pbo", "min_periods"))
    verdict = Checklist(config).run(scores=planted(), n_trials=60)
    assert [check.name for check in verdict.checks] == ["pbo", "min_periods"]


# ---------------------------------------------------------------------------
# Custom checks
# ---------------------------------------------------------------------------


def test_extra_checks_run_after_the_built_in_ones():
    seen = []

    def positive_mean(evidence: Evidence) -> CheckResult:
        seen.append(evidence)
        mean = float(evidence.winner.mean())
        return CheckResult("positive_mean", mean > 0, mean, 0.0, "value > threshold", "ok")

    verdict = Checklist(extra_checks=[positive_mean]).run(scores=planted(), n_trials=60)
    assert verdict.checks[-1].name == "positive_mean"
    assert verdict.passed
    assert seen[0].winner_row == 4 and seen[0].scores.shape == (60, 200)


def test_an_extra_check_must_return_a_check_result():
    with pytest.raises(TypeError, match="CheckResult"):
        Checklist(extra_checks=[lambda evidence: True]).run(scores=planted(), n_trials=60)


def test_extra_checks_need_their_own_names():
    """Two checks with one name would make verdict["dsr"] and the list of
    failed checks ambiguous."""

    def second_dsr(evidence):
        return CheckResult("dsr", True, None, None, "always", "a second check called dsr")

    with pytest.raises(ValueError, match="two checks are named 'dsr'"):
        Checklist(extra_checks=[second_dsr]).run(scores=planted(), n_trials=60)


def test_a_failing_extra_check_fails_the_verdict():
    def always_fail(evidence):
        return CheckResult("always_fail", False, None, None, "never passes", "no")

    verdict = Checklist(extra_checks=[always_fail]).run(scores=planted(), n_trials=60)
    assert not verdict.passed
    assert verdict.failed == ["always_fail"]


# ---------------------------------------------------------------------------
# Verdict and its JSON report
# ---------------------------------------------------------------------------


def test_verdict_text():
    verdict = Checklist().run(scores=planted(), n_trials=60)
    assert str(verdict) == "Verdict: PASS (min_periods PASS, dsr PASS, pbo PASS)"
    assert verdict.status == "PASS"
    with pytest.raises(KeyError):
        verdict["missing"]


def test_json_report_is_strict_json():
    """A losing winner has an infinite minimum track record length. The
    report writes it as null, so strict JSON parsers can read it."""
    losing = -np.abs(np.random.default_rng(1).normal(size=50))
    config = GateConfig(checks=("min_periods", "dsr"))
    verdict = Checklist(config).run(losing, n_trials=3, sr_std=0.1)
    text = verdict.to_json()
    assert "Infinity" not in text and "NaN" not in text
    report = json.loads(text)
    assert report["verdict"] == "FAIL"
    assert report["failed_checks"] == ["dsr"]
    assert report["checks"][1]["details"]["min_track_record"] is None
    assert "not a forecast" in report["note"]
    assert report["config"]["checks"] == ["min_periods", "dsr"]


def test_verdict_is_a_plain_value():
    verdict = Checklist().run(scores=planted(), n_trials=60)
    assert isinstance(verdict, Verdict)
    assert verdict.to_dict()["verdict"] == "PASS"
    assert verdict.to_dict()["failed_checks"] == []
