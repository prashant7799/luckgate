# luckgate

If you try many versions of something and keep the best one, the best one
nearly always looks good, even when every version is pure noise. luckgate
asks one question about the winner of such a search: did it beat luck?
It gives the Probabilistic and Deflated Sharpe Ratio (PSR and DSR), the
Sharpe ratio that the best of N noise trials would reach, the minimum
track record length, and the Probability of Backtest Overfitting (PBO) by
combinatorially symmetric cross-validation (CSCV). A configurable
checklist turns these into PASS or FAIL, with a JSON report and a command
line. It works for any score that is measured once per period: A/B test
variants, model or prompt versions, forecasting models, sensor rules or
trading-strategy research. It is for researchers who pick one winner from
many configurations, and for reviewers who need to check such a claim.

## What it does NOT do

- It does not forecast anything. PASS only means that none of the checks
  found that luck or overfitting explains the winner, for the inputs you
  gave. It is not a promise about the future, and it is not financial
  advice.
- It needs a score for every period (a time series) for every trial, on
  the same periods, with no missing values. One number per trial is not
  enough.
- It cannot see what you did not tell it. If you leave out trials, the
  result is too kind. `n_trials` must count every version you tried.
- It does not correct for autocorrelation. The Sharpe-ratio formulas
  treat periods as independent draws.
- It does not annualise. Every Sharpe ratio is per period.
- It does not estimate an "effective" number of independent trials from
  correlated trials. You pass `n_trials` yourself.
- It does not put a gap (embargo) between the in-sample and
  out-of-sample blocks of CSCV.
- It ranks trials in CSCV by Sharpe ratio or by mean only. Custom
  performance functions are planned.
- It does not compute bootstrap confidence intervals in this version.
  They are planned.
- It does not replace a final hold-out that the search never touched.
- It does not run your search, fetch data or plot anything.

## Install

```bash
pip install git+https://github.com/prashant7799/luckgate
```

It needs Python 3.10 or newer, numpy and scipy. A PyPI package is
planned.

## Quickstart (60 seconds)

200 versions of a ranking model are each scored once a day for 250 days.
The score is the day's gain over the current model. First, every version
is pure noise. Then version 17 gets a real gain.

```python
"""Quickstart: did the best of 200 model versions beat luck? (synthetic data)"""

import numpy as np

from luckgate import Checklist

# 200 versions of a ranking model, each scored once a day for 250 days.
# A score is that day's gain over the current model. All data is synthetic.
rng = np.random.default_rng(7)
scores = rng.normal(0.0, 1.0, size=(200, 250))  # pure noise: no version is better

checklist = Checklist()  # default checks: min_periods, dsr, pbo


def show(title, verdict):
    dsr = verdict["dsr"]
    print(title)
    print(" ", verdict)
    print(
        f"  best row {verdict.inputs['winner_row']}: Sharpe {dsr.details['sr']:.3f} per day,"
        f" DSR {dsr.value:.3f}, PBO {verdict['pbo'].value:.3f}"
    )


show("Noise only:", checklist.run(scores=scores, n_trials=200))

scores[17] += 0.4  # give version 17 a real gain of 0.4 per day
show("One real winner:", checklist.run(scores=scores, n_trials=200))
```

Output:

```text
Noise only:
  Verdict: FAIL (min_periods PASS, dsr FAIL, pbo FAIL)
  best row 67: Sharpe 0.162 per day, DSR 0.361, PBO 0.499
One real winner:
  Verdict: PASS (min_periods PASS, dsr PASS, pbo PASS)
  best row 17: Sharpe 0.357 per day, DSR 0.994, PBO 0.032
```

In the noise-only search, row 67 is the best of 200 by luck alone. Its
DSR is 0.361, far below 0.95. The search's PBO is about one half: the
in-sample best lands in the bottom half out of sample about as often as a
coin flip would say. With a real gain, row 17 wins, its DSR is 0.994 and
the PBO is 0.032.

The same script is in [`examples/quickstart.py`](examples/quickstart.py).

## Concepts

### Scores, periods and trials

A **trial** is one version you tried: one parameter setting, one model,
one prompt, one page variant. A **period** is one time step: a day, a
week, one evaluation batch. `scores` is a table with one row per trial and
one column per period, in time order. If your table has time down the
rows, pass its transpose.

The **Sharpe ratio** of a series is its mean divided by its standard
deviation. Here it is always **per period**: a daily series gives a daily
Sharpe ratio. Do not annualise it. `n_obs` is the number of periods.
`sr_std` is the standard deviation of the per-period Sharpe ratios of all
trials.

### PSR and DSR

The PSR is the probability that the true Sharpe ratio is above a
benchmark, given the measured one. It uses the standard error of the
Sharpe ratio. For a positive Sharpe ratio, the standard error grows with
negative skew and with fat tails (see Lo, 2002, and Opdyke, 2007):

```text
SE = sqrt( (1 - skew * SR + (kurtosis - 1) / 4 * SR**2) / (n_obs - 1) )      (raw kurtosis)
PSR(SR*) = Phi( (SR - SR*) / SE )
```

The DSR is the PSR against a higher benchmark: SR0, the Sharpe ratio that
the best of N pure-noise trials would reach on average.

```text
   Sharpe ratios of N trials that are all pure noise

              .:::::::.
           .:::::::::::::.
        .:::::::::::::::::::.
   ..:::::::::::::::::::::::::::..         |        |
 ------------------+---------------------- SR0 ---- SR ----->
                   0                       |        |
                          best of N noise trials    the winner
                          on average (SR0)

   PSR(0) asks: is the winner's true Sharpe above 0?
   DSR asks:    is it above SR0, the level that luck alone gives the best of N?

   SR0 = sr_std * ( (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)) )
         g = 0.5772... (Euler-Mascheroni), Z = inverse normal CDF
```

### The honest N

`n_trials` is the number of versions you tried, including the ones you
threw away and the ones from earlier rounds. If you report only a
shortlist and claim a small N, the DSR penalty shrinks. The scenario
tests show this: a shortlist of 10 noise trials, claimed as N = 10, passes
the DSR check. The checklist refuses an `n_trials` smaller than the
number of rows in `scores`, but it cannot see trials you never gave it.

### Raw and excess kurtosis

Raw kurtosis is 3 for a normal distribution. Excess kurtosis is raw
kurtosis minus 3, so it is 0 for a normal distribution. The papers use
raw kurtosis. If you put excess kurtosis into the raw formula, the
standard error comes out too small. Then the DSR is too sure of itself:
too high for any winner above SR0, which is the case that matters. So
every function that takes a kurtosis also takes `kurtosis_kind="raw"` or
`"excess"`. A raw kurtosis below `1 + skew**2` is impossible, so it is
rejected with a hint.

### Minimum track record length

MinTRL is the number of periods needed before the PSR against a benchmark
reaches a confidence level, such as 0.95:

```text
MinTRL = 1 + (1 - skew * SR + (kurtosis - 1) / 4 * SR**2) * ( Z(confidence) / (SR - SR*) )**2
```

`dsr_report` gives two: against 0, and against SR0 (the periods needed
for the DSR to reach the confidence level).

### PBO by CSCV

The DSR looks at the winner. PBO looks at the search. It asks: when you
pick the best trial on one half of the time, how often is it in the
bottom half on the other half?

```text
  periods ->  | B1 | B2 | B3 | B4 | B5 | B6 |    S = 6 blocks: C(6, 3) = 20 splits
  one split:    IS   OOS  IS   OOS  OOS  IS     (IS = in-sample, OOS = out-of-sample)

  1. On the IS blocks, find the best trial n*.
  2. On the OOS blocks, rank all N trials. r = rank of n* (1 = worst, N = best).
  3. omega = r / (N + 1)        logit = ln( omega / (1 - omega) )
  4. Repeat for every split.  PBO = share of splits with logit <= 0.
```

A logit at or below 0 means that the in-sample best was at or below the
out-of-sample median. For a search of pure noise, PBO is about 0.5. For a
search with one clearly better trial, it is near 0.

Some details the paper leaves open. The paper defines PBO as the share
of logits up to 0. A logit of exactly 0 (only possible with an odd N or a
tie) counts as overfit here, which is the cautious reading. Ties in the
OOS ranks share the average rank, and a tie for the in-sample best goes
to the first row.

So that rounding does not decide a tie, two scores tie when they are
equal, or when they differ by at most 1e-12 times the size of the values
behind them: the mean absolute value over the half, divided by the
standard deviation for the Sharpe ratio (the larger of the two sizes).
Two trials whose halves have a mean of exactly 0 on paper tie, even when
floating point leaves 1e-17 in one sum and not in the other. A half whose values are all
equal has zero spread, and its Sharpe ratio is +inf, -inf or 0, following
the sign of that value: a half of zeros scores exactly 0 for every trial.

luckgate combines a few numbers per block (the sum, the mean and the
spread) instead of going back to every period. Trials with the same
values in a half get exactly the same score. N = 200 trials, 250 periods
and 16 blocks (12,870 splits) took about 0.25 seconds on the author's
machine.

### The checklist

| Check | Passes when | Default limit | Where the idea comes from |
| --- | --- | --- | --- |
| `min_periods` | the winner has at least `min_periods` periods | 30 | an opinion |
| `dsr` | DSR >= `min_dsr` | 0.95 | the DSR paper; the 0.95 line is a convention |
| `pbo` | PBO <= `max_pbo` | 0.2 | the PBO paper; the 0.2 line is an opinion (see below) |
| `segments` | in every segment, the winner's mean >= `segment_floor`, with at least `min_segment_periods` periods | 0.0 and 20 | an opinion |

The default checks are `min_periods`, `dsr` and `pbo`. `segments` runs
only if you add it to `checks`. All limits are opinionated defaults:
change them to suit your problem.

The PBO paper mentions rejecting a model whose PBO is above 0.05 as a
customary choice. The default of 0.2 is looser. Set `max_pbo=0.05` to
follow the paper.

If a listed check is missing its input (for example `pbo` without
`scores`), luckgate raises an error. It does not skip the check silently.
A checklist with no checks at all (an empty `checks` and no extra checks)
is an error too, because its verdict would be PASS without testing
anything.

Two cautious heuristics are **not** part of the papers, and are off by
default:

- `clip_positive_skew=True` uses a skew of 0 when the measured skew is
  positive. For a positive Sharpe ratio, positive skew shrinks the
  standard error, so a few large wins can make a weak series look
  reliable.
- `sr_std_floor` puts a floor under `sr_std`. Near-duplicate trials have
  almost the same Sharpe ratio, so their spread, and with it the DSR
  penalty, can collapse.

## API reference

All functions are in the top-level `luckgate` package. Arguments after
`*` are keyword-only.

### Sharpe-ratio statistics

| Function | Returns |
| --- | --- |
| `sharpe_ratio(returns)` | per-period mean / standard deviation (ddof=1) |
| `sharpe_stats(returns)` | `SharpeStats(sr, n_obs, skew, kurtosis)`, with raw kurtosis |
| `trial_sharpe_std(scores)` | standard deviation (ddof=1) of every row's Sharpe ratio |
| `sharpe_se(sr, *, n_obs, skew=0, kurtosis=3, kurtosis_kind="raw")` | standard error of the Sharpe ratio |
| `psr(sr, *, n_obs, sr_benchmark=0, skew=0, kurtosis=3, kurtosis_kind="raw")` | Probabilistic Sharpe Ratio |
| `expected_max_sharpe(n_trials, sr_std=1.0)` | SR0, the expected best Sharpe ratio of N noise trials |
| `dsr(sr, *, n_obs, n_trials, sr_std, skew=0, kurtosis=3, kurtosis_kind="raw")` | Deflated Sharpe Ratio |
| `min_track_record_length(sr, *, sr_benchmark=0, skew=0, kurtosis=3, kurtosis_kind="raw", confidence=0.95)` | MinTRL in periods (`inf` if `sr <= sr_benchmark`) |
| `dsr_report(returns, *, n_trials, sr_std, confidence=0.95, clip_positive_skew=False)` | a `DSRReport` with all of the above for one series |
| `dsr_report_from_stats(measured, *, n_trials, sr_std, confidence=0.95, clip_positive_skew=False)` | the same, from a `SharpeStats` |

`DSRReport` fields: `sr`, `n_obs`, `skew`, `kurtosis` (raw), `skew_used`,
`n_trials`, `sr_std`, `expected_max_sharpe`, `sharpe_se`, `psr` (against
0), `dsr`, `confidence`, `min_track_record` (against 0) and
`min_track_record_deflated` (against SR0). `to_dict()` gives a dict.

Skewness and kurtosis use the plain moment estimators (`bias=True` in
scipy). `n_trials` may be a non-integer effective number of trials. With
`n_trials=1` there is no deflation, and the DSR equals PSR(0).

### Probability of Backtest Overfitting

```python
cscv_pbo(scores, *, n_blocks=16, metric="sharpe", max_splits=None, seed=0, chunk_size=1024)
```

- `scores`: one row per trial, one column per period, in time order.
- `n_blocks`: S, an even number. Blocks are contiguous. If the periods do
  not split evenly, block sizes differ by at most one period.
- `metric`: `"sharpe"` (mean / standard deviation over the half) or
  `"mean"`. A half whose values are all equal has zero spread and gets a
  Sharpe ratio of +inf, -inf or 0, following the sign of that value.
  Scores tie by the rule in [PBO by CSCV](#pbo-by-cscv). If several
  trials tie for the best in-sample score, the first row is selected.
- `max_splits`: if set and smaller than C(S, S/2), use that many distinct
  splits drawn at random with `seed`. Above one million splits it is
  required.
- `chunk_size`: splits processed at once. Lower it to save memory.

It returns a `PBOResult` with `pbo`, and one entry per split in
`logits`, `relative_ranks` (omega), `selected` (the in-sample best row),
`is_performance` and `oos_performance` (of the selected row). It also has
`n_splits`, `n_blocks`, `n_trials`, `n_periods`, `metric`, and two
properties: `prob_oos_loss` (the share of splits where the selected row
scores below 0 out of sample) and `degradation_slope` (the least-squares
slope of OOS on IS performance). `to_dict(include_arrays=False)` gives a
summary dict.

### Checklist

```python
Checklist(config=None, extra_checks=())
Checklist.run(winner=None, *, n_trials, scores=None, sr_std=None, segments=None) -> Verdict
```

- `winner`: the selected trial's series. If left out, the row of `scores`
  with the highest Sharpe ratio is used.
- `n_trials`: the honest number of trials. It must be at least the number
  of rows in `scores`.
- `sr_std`: if left out, it is measured from `scores` (when the `dsr` check
  runs). A given `sr_std` below the one measured from `scores` is still
  used, with a `UserWarning`, because a smaller `sr_std` shrinks the DSR
  penalty.
- `segments`: one label per period of `winner`, for the `segments` check.

`GateConfig` fields (all optional): `checks`, `min_periods`, `min_dsr`,
`max_pbo`, `pbo_blocks`, `pbo_metric`, `pbo_max_splits`, `pbo_seed`,
`segment_floor`, `min_segment_periods`, `clip_positive_skew` and
`sr_std_floor`. Every limit must be a finite number, `min_periods`,
`min_segment_periods`, `pbo_blocks`, `pbo_seed` and `pbo_max_splits` must
be whole numbers, and `clip_positive_skew` must be `True` or `False`.
NaN, strings such as `"0.9"` or `"false"`, and `True` or `False` given as
a number are rejected. `GateConfig.from_dict(data)` also rejects unknown
keys, so a typo cannot be ignored. `to_dict()` goes the other way.

`Verdict` has `passed`, `status` ("PASS" or "FAIL"), `failed` (names of
the failed checks), `checks` (a tuple of `CheckResult`), `inputs` and
`config`. `verdict["dsr"]` gets one check. `str(verdict)` is a one-line
summary. `to_dict()` and `to_json()` give the report. In JSON, infinite or
undefined numbers are written as `null`.

`CheckResult(name, passed, value, threshold, rule, message, details={})`
is one check's outcome. Its `status` is "PASS" or "FAIL".

A custom check is any function that takes an `Evidence` and returns a
`CheckResult`. `Evidence` has `winner`, `n_trials`, `config`, `scores`,
`winner_row`, `sr_std`, `sr_std_source` and `segments`.

```python
from luckgate import CheckResult, Checklist


def worst_day(evidence):
    worst = float(evidence.winner.min())
    return CheckResult(
        "worst_day", worst >= -3.5, worst, -3.5, "value >= threshold", f"worst day {worst:.2f}"
    )


checklist = Checklist(extra_checks=[worst_day])
```

### Command line

`luckgate dsr`, `luckgate pbo` and `luckgate check` print JSON. The exit
code is 0 on success, 1 when `check` gives FAIL, and 2 on bad input.
Input files hold numbers separated by commas or spaces, one row per line.
A first line with no numbers in it is a header and is skipped. A first
line with one bad number is an error, so no row of data is dropped
quietly. Lines that start with `#` are skipped. Score files have one row per trial; add
`--transpose` if time runs down the rows. Run `luckgate <command> --help`
for every option.

The DSR paper's example, from its summary numbers (per day):

```bash
luckgate dsr --sr 0.158113883 --n-obs 1250 --skew -3 --kurtosis 10 --n-trials 100 --sr-std 0.0447213595
```

```text
{
  "sr": 0.158114,
  "n_obs": 1250,
  "skew": -3.0,
  "kurtosis": 10.0,
  "skew_used": -3.0,
  "n_trials": 100.0,
  "sr_std": 0.044721,
  "expected_max_sharpe": 0.113172,
  "sharpe_se": 0.035006,
  "psr": 0.999997,
  "dsr": 0.900397,
  "confidence": 0.95,
  "min_track_record": 166.643289,
  "min_track_record_deflated": 2051.271417
}
```

`luckgate dsr winner.csv --n-trials 100 --scores scores.csv` does the same
from a series file, and measures `sr_std` from the scores file.

Make a synthetic scores file to try `pbo` and `check`:

```bash
python examples/make_demo_csv.py demo_scores.csv
```

```text
wrote demo_scores.csv: 100 trials x 120 periods
```

```bash
luckgate pbo demo_scores.csv
```

```text
{
  "pbo": 0.045377,
  "n_splits": 12870,
  "n_blocks": 16,
  "n_trials": 100,
  "n_periods": 120,
  "metric": "sharpe",
  "logit_median": 4.60517,
  "prob_oos_loss": 0.052137,
  "degradation_slope": -0.423834
}
```

```bash
luckgate check demo_scores.csv --n-trials 100
```

The report starts like this, and the exit code is 0 (PASS):

```text
{
  "verdict": "PASS",
  "failed_checks": [],
  "checks": [
    {
      "name": "min_periods",
      "status": "PASS",
      "value": 120.0,
      "threshold": 30.0,
      "rule": "value >= threshold",
      "message": "120 periods; at least 30 are required",
      "details": {}
    },
...
```

`check` also takes `--winner FILE` or `--winner-row K`, `--sr-std`,
`--segments FILE` (one label per line) and `--config gate.json` (a JSON
object of `GateConfig` fields, for example
`{"checks": ["min_periods", "dsr"], "min_dsr": 0.9}`).

## Examples

All examples use synthetic data. Each one prints the output shown here,
and a test checks that it still does.

### The DSR paper's example, and two common mistakes

[`examples/paper_example.py`](examples/paper_example.py) rebuilds the
paper's numbers from its summary statistics. The same data with excess
kurtosis gives the same answer once it is declared. Putting annual
numbers into the per-period formula turns 0.9004 into near certainty.

```text
daily Sharpe of the winner:          0.1581
expected best of 100 noise trials:   0.1132
DSR (raw kurtosis 10):               0.9004
DSR (same data, excess kurtosis 7):  0.9004
DSR with annual numbers (a mistake): 1.0000
```

### A/B test: the best of 40 variants

[`examples/ab_variants.py`](examples/ab_variants.py): 40 page variants
are compared with one control for 120 days, and none of them is better.
Tested alone, the best variant has p = 0.015. Counting all 40 tries, it
fails. PBO uses the mean daily lift (`pbo_metric="mean"`).

```text
variants with p < 0.05 on their own: 1 of 40
best variant: 22, mean lift +0.124 points a day, p = 0.0151
Verdict: FAIL (min_periods PASS, dsr FAIL, pbo FAIL)
  min_periods: 120 periods; at least 30 are required
  dsr: DSR 0.6966: winner Sharpe 0.2251 per period vs 0.1760 expected from the best of 40 noise trials
  pbo: PBO 0.4844: the in-sample best ranked in the bottom half out of sample in 48.4% of 12870 splits
```

### Segments and a custom check

[`examples/forecast_models.py`](examples/forecast_models.py): 60 versions
of a demand-forecast model are scored daily for a year. Version 5 is much
better in winter and worse in summer. It passes the DSR and PBO checks,
and the `segments` check catches the summer problem. A custom
`worst_day` check runs too.

```text
Verdict: FAIL (min_periods PASS, dsr PASS, pbo PASS, segments FAIL, worst_day PASS)
winner row: 5
  winter  mean +0.852 over 91 days: PASS
  spring  mean +0.197 over 91 days: PASS
  summer  mean -0.099 over 91 days: FAIL
  autumn  mean +0.438 over 91 days: PASS
failed: ['segments']
```

### Measured false-pass rates

[`examples/false_pass_rates.py`](examples/false_pass_rates.py) runs 100
synthetic searches per row. Each search tries 100 configurations on 250
periods. In the noise rows every PASS is a false pass. "shortlist" means
that only the 10 best trials are reported, with `n_trials=10`. It takes
under a minute.

```text
search                      dsr pass  pbo pass    PASS
noise, honest                   0.00      0.02    0.00
noise, shortlist of 10          0.82      0.00    0.00
real gain 0.4, honest           0.96      0.98    0.96
real gain 0.4, shortlist        1.00      0.95    0.95
```

With an honest N, no noise search passed. With a shortlist, the DSR check
is fooled in 82% of noise searches, and the PBO check catches every one of
them. These rates are for this setup only. They are not general error
rates.

## How correctness is tested

Run `pytest -q`. The tests use synthetic data and no network.

- **Golden values from the DSR paper.** The numerical example of Bailey
  and López de Prado (2014): 100 trials, variance of the annual Sharpe
  ratios 0.5, best annual Sharpe 2.5, 1250 daily values, skew -3, raw
  kurtosis 10. The paper gives a DSR of 0.9004, and so does luckgate. The
  expected maximum works out to 0.1132 per day (1.79 a year). Each step is
  also written out by hand in the test. This is the only published worked
  example in the tests. The PSR and the minimum track record length are
  checked by hand computations and identities, not against a printed
  example.
- **The mistakes these formulas are known for.** Excess kurtosis put into
  the raw formula: a test shows a case where this turns a FAIL at 0.95
  into a PASS, and another shows that excess kurtosis 0 labelled as raw is
  rejected. Annual numbers in the per-period formula: a test shows that
  the paper example jumps from 0.9004 to above 0.9999.
- **Identities and properties.** DSR equals PSR against SR0. With one
  trial, DSR equals PSR(0). Property tests (hypothesis) check that the PSR
  at the minimum track record length equals the confidence level, that
  the PSR is always a probability, that a higher benchmark never raises
  the PSR, that more trials or more spread between trials never raise the
  DSR, and that rescaling a series never changes its report.
- **Bad inputs.** Impossible inputs raise an error with a hint: excess
  kurtosis labelled as raw, a variance term at or below 0 (often an
  annualised Sharpe ratio), `True` passed as a number, a string where a
  list of checks belongs, two checks with the same name, an empty list of
  checks, a NaN or string limit in a config, and a first row of a data
  file with one bad number.
- **The expected maximum formula** is compared with the exact expected
  maximum of N normal draws (by numerical integration). It is within 3%
  for N from 10 to 10,000.
- **PBO against a slow reference.** `tests/reference_pbo.py` follows the
  paper step by step, with the tie rules above: it loops over every split
  and recomputes every trial from the raw columns. The fast version must
  give the same logits on fixed tables (with even and uneven blocks), on
  random tables from hypothesis with no offset to keep trials apart, on
  tables built from a few values so that many scores tie, on rows of 0
  and 1, a half of zeros, rows that agree on some blocks only, duplicated
  trials, and data with a huge mean next to its spread or a jump between
  the halves.
- **PBO cases with a known answer.** Rows that sum to zero give PBO = 1
  (the in-sample best is always the out-of-sample worst). One dominant
  trial gives PBO = 0. Identical trials tie with omega = 1/2, and so do
  trials whose halves all have a mean of 0 on paper. Pure noise gives
  about 0.5 on average. Scaling all scores, scaling each trial on
  its own (for the Sharpe metric), reordering trials, or (for the mean
  metric) shifting all scores changes nothing. With S = 16 there are
  C(16, 8) = 12,870 splits. The paper prints 12,780, which is a typo.
- **Adversarial scenarios** (`tests/test_scenarios.py`). Each has its own
  seed and asserts the exact list of failed checks: a real winner (none),
  the best of noise (`dsr`, `pbo`), a shortlist with an understated N
  (`pbo` only, so DSR is fooled), a short record (`min_periods`, `dsr`), a
  winner that only works in one season (`segments`), a segment with too
  few periods (`segments`), a collapsed `sr_std` (fools `dsr`, caught by
  `sr_std_floor`), and a lottery payoff with strong positive skew (passes,
  caught by `clip_positive_skew`).
- **Monte Carlo rates.** In each of two sets of 40 noise searches, the
  default checklist passes at most 5%. In 40 searches with a clear edge,
  it passes at least 90%.
- **The README.** The quickstart, every example and every command line
  output shown above are run and compared with this file.

## Limitations and assumptions

- The PSR, DSR and MinTRL treat the periods as independent draws.
  Positive autocorrelation makes the true standard error larger than the
  formula says, so the results are too kind.
- SR0 assumes that the trials' Sharpe ratios are roughly normal, with a
  true value of 0 under the null, and that the N trials are independent.
  Strongly correlated trials are fewer "effective" trials than N. The
  formula is an approximation: within 3% of the exact value for N >= 10,
  rough for N below 10 (about 8% too low at N = 2), and clipped at 0
  below about 1.3 trials. From N = 3 up it is a little above the exact
  value, which makes the DSR a little stricter.
- `sr_std` measured from a few rows, or from near-duplicate rows, can be
  too small. Then the DSR penalty is too small.
- CSCV uses contiguous blocks with no gap between them. If one period's
  score depends on data from nearby periods (for example overlapping
  windows), information leaks across the border between halves. PBO also
  needs the same periods for every trial.
- PBO describes the selection process on this data. It is not a property
  of the winner alone, and a low PBO does not prove a real edge.
- `cscv_pbo` ranks by Sharpe ratio or mean only in this version.
- The `segments` check compares means with a floor. It is not a
  significance test.
- The default limits are opinions. Change them to suit your problem.

## Related approaches

- **Corrections on p-values.** Bonferroni and Holm control the chance of
  any false discovery. Benjamini and Hochberg (1995) control the false
  discovery rate. They are simple and general, but they ignore the
  correlation between trials.
- **Bootstrap tests of the best of many.** White's Reality Check (White,
  2000), Hansen's test for superior predictive ability (Hansen, 2005) and
  the stepwise method of Romano and Wolf (2005) compare the best trial
  with a benchmark while accounting for the search.
- **Haircut Sharpe ratios** (Harvey and Liu, 2015) adjust a Sharpe ratio
  for multiple testing through its p-value.
- **Time-series bootstraps**, such as the stationary bootstrap (Politis
  and Romano, 1994), give confidence intervals for dependent data.
  luckgate does not include them yet.
- **A final untouched hold-out**, or nested walk-forward validation, is
  the strongest guard when you have enough data. luckgate is a check to
  run before you spend that hold-out, not a replacement for it.
- If you compare luckgate's numbers with another implementation of the
  DSR or CSCV, first check that both use the same kurtosis form, the same
  rank rule (omega = r / (N + 1), ties, logit <= 0) and the same blocks.

## References and citation

- Bailey, D. H. and López de Prado, M. (2012). The Sharpe Ratio Efficient
  Frontier. *Journal of Risk*, 15(2), 3-44. doi:10.21314/JOR.2012.255.
  (PSR, minimum track record length.)
- Bailey, D. H. and López de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
  *Journal of Portfolio Management*, 40(5), 94-107.
  doi:10.3905/jpm.2014.40.5.094. (DSR, expected maximum Sharpe ratio,
  the numerical example.)
- Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2014).
  Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest
  Overfitting on Out-of-Sample Performance. *Notices of the American
  Mathematical Society*, 61(5), 458-471. doi:10.1090/noti1105.
- Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2017).
  The Probability of Backtest Overfitting. *Journal of Computational
  Finance*, 20(4), 39-69. doi:10.21314/JCF.2016.322. (CSCV and PBO.)
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts
  Journal*, 58(4), 36-52.
- Opdyke, J. D. (2007). Comparing Sharpe Ratios: So Where Are the
  p-Values? *Journal of Asset Management*, 8(5), 308-336.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the False Discovery
  Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of
  the Royal Statistical Society, Series B*, 57(1), 289-300.
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*,
  68(5), 1097-1126.
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal
  of Business and Economic Statistics*, 23(4), 365-380.
- Romano, J. P. and Wolf, M. (2005). Stepwise Multiple Testing as
  Formalized Data Snooping. *Econometrica*, 73(4), 1237-1282.
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio
  Management*, 42(1), 13-28.
- Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap.
  *Journal of the American Statistical Association*, 89(428), 1303-1313.

If you use luckgate in your work, please cite the papers above.
[`CITATION.cff`](CITATION.cff) also describes how to cite this software.

## Licence

MIT. See [`LICENSE`](LICENSE).
