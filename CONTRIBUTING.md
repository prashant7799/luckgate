# Contributing

Thank you for helping. Bug reports, questions and small pull requests are
all welcome.

## Set up

```bash
git clone https://github.com/prashant7799/luckgate
cd luckgate
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Before you open a pull request

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format .
```

## Rules

- Any change to a formula must keep the golden test from the DSR paper
  passing, and must cite the paper and equation it follows.
- Any change to `cscv_pbo` must still agree with the slow reference in
  `tests/reference_pbo.py`. Keep the reference slow and obvious.
- A heuristic that is not in the papers must be opt-in, and must be
  labelled as a heuristic in the code and the README.
- Add a scenario test for every new check, asserting the exact list of
  failed checks.
- Use synthetic data only in tests and examples. Never commit real data,
  keys or credentials.
- If a README output changes, update the README. A test checks it.
- Write plain code with clear names. A plain loop is better than a clever
  one-liner.

## Reporting a bug

Please include a small synthetic input that shows the problem, the call you
made, what you expected, and what you got.
