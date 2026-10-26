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
