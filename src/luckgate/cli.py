"""Command line interface: `luckgate dsr`, `luckgate pbo` and `luckgate check`.

Every command prints JSON to standard output. Exit codes:

- 0: success (for `check`: the verdict is PASS)
- 1: `check` only: the verdict is FAIL
- 2: bad input or bad arguments (the message goes to standard error)

Input files are plain text: numbers separated by commas or spaces, one
row per line. A first line with no numbers in it is treated as a header
and skipped. Lines that start with '#' are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from luckgate import __version__
from luckgate.checklist import Checklist, GateConfig, json_safe
from luckgate.pbo import METRICS, cscv_pbo
from luckgate.sharpe import (
    SharpeStats,
    dsr_report,
    dsr_report_from_stats,
    trial_sharpe_std,
)

#: Printed numbers are rounded to this many decimal places.
DIGITS = 6


# ---------------------------------------------------------------------------
# Reading input files
# ---------------------------------------------------------------------------


def is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def read_table(path: str) -> np.ndarray:
    """Read a 2-D table of numbers from a text file.

    Only a first line in which no cell is a number counts as a header. A
    first line with a typo in one number is an error, so a row of data is
    never dropped without a word.
    """
    rows: list[list[float]] = []
    header_skipped = False
    # "utf-8-sig" also removes the byte-order mark that some spreadsheet
    # programs write at the start of a file.
    with open(path, encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            cells = [cell.strip() for cell in text.split(",")] if "," in text else text.split()
            if all(is_number(cell) for cell in cells):
                rows.append([float(cell) for cell in cells])
                continue
            looks_like_header = not any(is_number(cell) for cell in cells)
            if not rows and not header_skipped and looks_like_header:
                header_skipped = True
                continue
            message = f"{path}, line {line_number}: not a number in {text[:60]!r}"
            raise ValueError(message)
    if not rows:
        raise ValueError(f"{path} contains no numbers")
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"{path}: rows have different lengths {sorted(lengths)}")
    return np.array(rows, dtype=float)


def read_series(path: str) -> np.ndarray:
    """Read one series: a single row or a single column of numbers."""
    table = read_table(path)
    if table.shape[0] == 1:
        return table[0]
    if table.shape[1] == 1:
        return table[:, 0]
    raise ValueError(f"{path} must hold one row or one column; got shape {table.shape}")


def read_scores(path: str, transpose: bool) -> np.ndarray:
    table = read_table(path)
    return table.T if transpose else table


def read_labels(path: str) -> np.ndarray:
    """Read one label per line. Blank lines and lines starting with '#' are skipped."""
    labels = []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("#"):
                labels.append(text)
    return np.array(labels)


def read_config(path: str | None) -> GateConfig:
    if path is None:
        return GateConfig()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return GateConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def rounded(value):
    """Round every float in a nested structure, for stable printed output."""
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, DIGITS)
    return value


def print_json(data: dict) -> None:
    print(json.dumps(rounded(json_safe(data)), indent=2))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def run_dsr(args) -> int:
    if args.sr_std is not None:
        sr_std = args.sr_std
    elif args.scores is not None:
        sr_std = trial_sharpe_std(read_scores(args.scores, args.transpose))
    else:
        raise ValueError("give --sr-std, or --scores to measure it from")

    if args.returns is not None:
        if args.sr is not None or args.n_obs is not None:
            raise ValueError("give a returns file or --sr and --n-obs, not both")
        report = dsr_report(
            read_series(args.returns),
            n_trials=args.n_trials,
            sr_std=sr_std,
            confidence=args.confidence,
            clip_positive_skew=args.clip_positive_skew,
        )
    else:
        if args.sr is None or args.n_obs is None:
            raise ValueError("give a returns file, or both --sr and --n-obs")
        kurtosis = args.kurtosis + 3.0 if args.kurtosis_kind == "excess" else args.kurtosis
        measured = SharpeStats(sr=args.sr, n_obs=args.n_obs, skew=args.skew, kurtosis=kurtosis)
        report = dsr_report_from_stats(
            measured,
            n_trials=args.n_trials,
            sr_std=sr_std,
            confidence=args.confidence,
            clip_positive_skew=args.clip_positive_skew,
        )
    print_json(report.to_dict())
    return 0


def run_pbo(args) -> int:
    result = cscv_pbo(
        read_scores(args.scores, args.transpose),
        n_blocks=args.blocks,
        metric=args.metric,
        max_splits=args.max_splits,
        seed=args.seed,
    )
    print_json(result.to_dict(include_arrays=args.arrays))
    return 0


def run_check(args) -> int:
    config = read_config(args.config)
    scores = read_scores(args.scores, args.transpose)
    if args.winner is not None and args.winner_row is not None:
        raise ValueError("give --winner or --winner-row, not both")
    winner = None
    if args.winner is not None:
        winner = read_series(args.winner)
    elif args.winner_row is not None:
        if not 0 <= args.winner_row < scores.shape[0]:
            raise ValueError(f"--winner-row must be between 0 and {scores.shape[0] - 1}")
        winner = scores[args.winner_row]
    segments = None if args.segments is None else read_labels(args.segments)

    verdict = Checklist(config).run(
        winner,
        n_trials=args.n_trials,
        scores=scores,
        sr_std=args.sr_std,
        segments=segments,
    )
    print_json(verdict.to_dict())
    return 0 if verdict.passed else 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="luckgate",
        description="Did the winner of a search beat luck? All Sharpe ratios are per period.",
    )
    parser.add_argument("--version", action="version", version=f"luckgate {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    dsr = commands.add_parser(
        "dsr",
        help="PSR, DSR and minimum track record length for one series",
        description="Deflated Sharpe Ratio of a winner, from a series file or from summary "
        "numbers (--sr and --n-obs).",
    )
    dsr.add_argument("returns", nargs="?", help="file with the winner's per-period values")
    dsr.add_argument("--n-trials", type=float, required=True, help="honest number of trials")
    dsr.add_argument("--sr-std", type=float, help="std of all trials' per-period Sharpe ratios")
    dsr.add_argument("--scores", help="trials-by-periods file to measure --sr-std from")
    dsr.add_argument("--transpose", action="store_true", help="scores file is periods-by-trials")
    dsr.add_argument("--sr", type=float, help="per-period Sharpe ratio (instead of a file)")
    dsr.add_argument("--n-obs", type=int, help="number of periods (instead of a file)")
    dsr.add_argument("--skew", type=float, default=0.0, help="skewness (default 0)")
    dsr.add_argument("--kurtosis", type=float, default=3.0, help="kurtosis (default 3, raw)")
    dsr.add_argument("--kurtosis-kind", choices=("raw", "excess"), default="raw")
    dsr.add_argument("--confidence", type=float, default=0.95, help="for the track record lengths")
    dsr.add_argument("--clip-positive-skew", action="store_true", help="heuristic, not in papers")
    dsr.set_defaults(handler=run_dsr)

    pbo = commands.add_parser(
        "pbo",
        help="probability of backtest overfitting (CSCV)",
        description="Probability of Backtest Overfitting of a whole search, by CSCV.",
    )
    pbo.add_argument("scores", help="file with one row per trial and one column per period")
    pbo.add_argument("--transpose", action="store_true", help="the file is periods-by-trials")
    pbo.add_argument("--blocks", type=int, default=16, help="number of blocks S (even)")
    pbo.add_argument("--metric", choices=METRICS, default="sharpe")
    pbo.add_argument("--max-splits", type=int, help="sample this many splits instead of all")
    pbo.add_argument("--seed", type=int, default=0, help="seed for sampled splits")
    pbo.add_argument("--arrays", action="store_true", help="also print the per-split arrays")
    pbo.set_defaults(handler=run_pbo)

    check = commands.add_parser(
        "check",
        help="run the pass/fail checklist (exit 0 = PASS, 1 = FAIL)",
        description="Run the checklist on the winner of a search. Exit code 0 means PASS, "
        "1 means FAIL, 2 means bad input.",
    )
    check.add_argument("scores", help="file with one row per trial and one column per period")
    check.add_argument("--n-trials", type=float, required=True, help="honest number of trials")
    check.add_argument("--transpose", action="store_true", help="the file is periods-by-trials")
    check.add_argument("--winner", help="file with the winner's series (default: best row)")
    check.add_argument("--winner-row", type=int, help="row of the scores file to check")
    check.add_argument("--sr-std", type=float, help="std of all trials' Sharpe ratios")
    check.add_argument("--segments", help="file with one segment label per period")
    check.add_argument("--config", help="JSON file with GateConfig fields")
    check.set_defaults(handler=run_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, TypeError, OSError) as error:
        print(f"luckgate: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
