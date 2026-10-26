"""Write a synthetic scores file for trying the command line.

The file has one row per trial (100 trials) and one column per period
(120 periods). Row 17 has a real gain of 0.4 per period; the other rows
are pure noise.

Usage: python examples/make_demo_csv.py [path]   (default: demo_scores.csv)
"""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "demo_scores.csv"
rng = np.random.default_rng(42)
scores = rng.normal(0.0, 1.0, size=(100, 120))
scores[17] += 0.4
np.savetxt(path, scores, delimiter=",", fmt="%.6f")
print(f"wrote {path}: {scores.shape[0]} trials x {scores.shape[1]} periods")
