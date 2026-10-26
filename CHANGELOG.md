# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/).

## [0.1.0] - 2026-10-23

First release.

### Added

- Sharpe-ratio statistics, all per period: `sharpe_ratio`, `sharpe_stats`
  (with raw kurtosis), `trial_sharpe_std`, `sharpe_se`, `psr`,
  `expected_max_sharpe`, `dsr` and `min_track_record_length`, after
  Bailey and López de Prado (2012, 2014).
- An explicit `kurtosis_kind="raw"` or `"excess"` on every function that
  takes a kurtosis. A raw kurtosis below `1 + skew**2` is rejected with a
  hint about excess kurtosis.
- `dsr_report` and `dsr_report_from_stats`: PSR, DSR, the expected
  maximum Sharpe ratio and both minimum track record lengths in one
  `DSRReport`.
- `cscv_pbo`: the probability of backtest overfitting by CSCV (Bailey,
  Borwein, López de Prado and Zhu), with omega = r / (N + 1), PBO = share
  of logits <= 0, average ranks for ties, uneven blocks, the `"sharpe"`
  and `"mean"` metrics, optional sampled splits, and chunked processing
  from per-block sums. `PBOResult` also gives `prob_oos_loss` and
  `degradation_slope`.
- Ties in `cscv_pbo` follow the documented rules and not rounding. Means
  come from plain per-block sums and spreads from each block's own
  squared distances, so trials with the same values in a half get exactly
  the same score, and the spread stays accurate with a large mean or a
  jump between the halves. A half whose values are all equal has zero
  spread: its Sharpe ratio is +inf, -inf or 0 by the sign of that value,
  so a half of zeros scores exactly 0 for every trial. Two scores that
  differ by at most `luckgate.pbo.TIE_TOLERANCE` (1e-12) times the size
  of the values behind them tie. The first row wins a tie for the in-sample best.
- A configurable checklist: `GateConfig`, `Checklist`, `Evidence`,
  `CheckResult` and `Verdict`, with the checks `min_periods`, `dsr`, `pbo`
  and `segments`, custom check functions, and a JSON report.
- A checklist cannot pass without testing anything: one with no checks
  (an empty `checks` and no extra checks) is an error, and
  `luckgate check` exits with 2 for it. `GateConfig` and
  `GateConfig.from_dict` require finite numbers for limits, whole
  numbers for counts, `pbo_blocks` and `pbo_seed`, and `True` or `False`
  for `clip_positive_skew`. NaN, infinity, strings such as
  `"0.9"` or `"false"`, and `True` or `False` given as a number are
  rejected.
- A `UserWarning` when a given `sr_std` is below the one measured from
  `scores`, since a smaller `sr_std` shrinks the DSR penalty. The given
  value is still used.
- Two opt-in heuristics that are not in the papers, labelled as such:
  `clip_positive_skew` and `sr_std_floor`.
- A command line: `luckgate dsr`, `luckgate pbo` and `luckgate check`,
  with JSON output and exit codes (0 PASS or success, 1 FAIL, 2 bad
  input). Only a first line with no numbers counts as a header, so a
  first row of data with a typo is an error and is never dropped.
- Input checks with a hint for impossible inputs: excess kurtosis
  labelled as raw, a variance term at or below 0, `True` or `False`
  passed as a number, a string for `checks`, and two checks with the
  same name.
- Tests: golden values from the DSR paper's example, regression tests for
  the excess-kurtosis and annualised-Sharpe mistakes, property tests
  (identities, monotonicity, invariance to units), the expected maximum
  against exact integration, CSCV against a slow reference
  implementation (also on tables full of ties, with no offset added to
  keep trials apart), adversarial scenarios that assert the exact failed
  checks, Monte Carlo pass rates, and README output checks.
- Examples on synthetic data: the paper example, A/B variants, forecast
  models with segments and a custom check, false-pass rates, and a demo
  file for the command line.

### Planned

- A release on PyPI, so `pip install luckgate` works.
- Bootstrap confidence intervals for dependent data (iid, block and
  stationary resampling), as an extra check.
- Custom performance functions in `cscv_pbo`, beyond Sharpe ratio and
  mean.
- An estimate of the effective number of independent trials from the
  correlation between trials.
- An optional gap (embargo) between in-sample and out-of-sample blocks.
- Adapters that read trials and scores from experiment trackers and
  hyperparameter search tools.

[0.1.0]: https://github.com/prashant7799/luckgate/releases/tag/v0.1.0
