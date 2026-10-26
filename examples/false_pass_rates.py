"""How often does the checklist pass a winner that is pure luck?

Monte Carlo on synthetic searches. Each search tries 100 configurations
and scores each one on 250 periods. In the noise searches no
configuration is better than any other, so every PASS is a false pass.
"""

import numpy as np

from luckgate import Checklist

N_RUNS = 100
N_TRIALS, N_PERIODS = 100, 250
checklist = Checklist()  # default checks and limits


def sharpe_of_rows(scores):
    return scores.mean(axis=1) / scores.std(axis=1, ddof=1)


def honest(scores):
    """Report every trial and the true number of trials."""
    return checklist.run(scores=scores, n_trials=N_TRIALS)


def shortlist(scores):
    """Keep only the 10 best trials and claim that only 10 were tried."""
    top10 = np.argsort(sharpe_of_rows(scores))[-10:]
    return checklist.run(scores=scores[top10], n_trials=10)


def run(scenario, planted_gain):
    passes = {"dsr": 0, "pbo": 0, "verdict": 0}
    for seed in range(N_RUNS):
        rng = np.random.default_rng(seed)
        scores = rng.normal(0.0, 1.0, size=(N_TRIALS, N_PERIODS))
        scores[0] += planted_gain
        verdict = scenario(scores)
        passes["dsr"] += verdict["dsr"].passed
        passes["pbo"] += verdict["pbo"].passed
        passes["verdict"] += verdict.passed
    return {name: count / N_RUNS for name, count in passes.items()}


rows = [
    ("noise, honest", honest, 0.0),
    ("noise, shortlist of 10", shortlist, 0.0),
    ("real gain 0.4, honest", honest, 0.4),
    ("real gain 0.4, shortlist", shortlist, 0.4),
]
print(f"{'search':<26}{'dsr pass':>10}{'pbo pass':>10}{'PASS':>8}")
for title, scenario, gain in rows:
    rates = run(scenario, gain)
    print(f"{title:<26}{rates['dsr']:>10.2f}{rates['pbo']:>10.2f}{rates['verdict']:>8.2f}")
