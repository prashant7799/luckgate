"""A/B test: 40 page variants against one control, and none of them is better.

Each day we record a variant's conversion rate minus the control's, in
percentage points. The true lift of every variant is 0. All data is
synthetic.
"""

import numpy as np
from scipy import stats

from luckgate import Checklist, GateConfig

rng = np.random.default_rng(11)
n_variants, n_days = 40, 120
lift = rng.normal(0.0, 0.5, size=(n_variants, n_days))  # daily lift, in points

# The usual way: test each variant on its own, then ship the best one.
p_values = stats.ttest_1samp(lift, 0.0, axis=1).pvalue
best = int(np.argmax(lift.mean(axis=1)))
print(f"variants with p < 0.05 on their own: {np.sum(p_values < 0.05)} of {n_variants}")
print(
    f"best variant: {best}, mean lift {lift[best].mean():+.3f} points a day, "
    f"p = {p_values[best]:.4f}"
)

# The luckgate way: count all 40 tries, and rank variants by their mean lift.
config = GateConfig(pbo_metric="mean")
verdict = Checklist(config).run(lift[best], n_trials=n_variants, scores=lift)
print(verdict)
for check in verdict.checks:
    print(f"  {check.name}: {check.message}")
