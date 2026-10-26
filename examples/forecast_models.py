"""Segments and a custom check: a forecast model that only helps in winter.

60 versions of a demand-forecast model are scored once a day for a year.
A score is that day's error reduction over the current model (positive is
better). Version 5 is much better in winter, and worse in summer. All data
is synthetic.
"""

import numpy as np

from luckgate import Checklist, CheckResult, GateConfig

rng = np.random.default_rng(3)
n_versions, n_days = 60, 364
seasons = np.repeat(["winter", "spring", "summer", "autumn"], n_days // 4)

scores = rng.normal(0.0, 1.0, size=(n_versions, n_days))
season_effect = {"winter": 0.9, "spring": 0.3, "summer": -0.25, "autumn": 0.3}
for season, effect in season_effect.items():
    scores[5, seasons == season] += effect


def worst_day(evidence):
    """A custom check: no single day may be worse than -3.5."""
    worst = float(evidence.winner.min())
    return CheckResult(
        name="worst_day",
        passed=worst >= -3.5,
        value=worst,
        threshold=-3.5,
        rule="value >= threshold",
        message=f"worst day {worst:.2f}",
    )


config = GateConfig(checks=("min_periods", "dsr", "pbo", "segments"))
checklist = Checklist(config, extra_checks=[worst_day])
verdict = checklist.run(scores=scores, n_trials=n_versions, segments=seasons)

print(verdict)
print("winner row:", verdict.inputs["winner_row"])
for season, numbers in verdict["segments"].details.items():
    print(
        f"  {season:<7} mean {numbers['mean']:+.3f} over {numbers['n_periods']} days: "
        f"{numbers['status']}"
    )
print("failed:", verdict.failed)
