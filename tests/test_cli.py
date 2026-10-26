"""The command line: JSON output and exit codes."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from synthetic import noise_scores

from luckgate import Checklist, cscv_pbo, dsr_report, trial_sharpe_std
from luckgate.cli import main, read_table

PAPER_ARGS = [
    "dsr",
    "--sr", "0.158113883",
    "--n-obs", "1250",
    "--skew", "-3",
    "--kurtosis", "10",
    "--n-trials", "100",
    "--sr-std", "0.0447213595",
]  # fmt: skip


def run(argv, capsys):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_scores(path, scores, header=False):
    with open(path, "w", encoding="utf-8") as handle:
        if header:
            handle.write(",".join(f"p{k}" for k in range(scores.shape[1])) + "\n")
        for row in scores:
            handle.write(",".join(f"{value:.17g}" for value in row) + "\n")
    return str(path)


@pytest.fixture
def planted_file(tmp_path):
    scores = noise_scores(40, 120, seed=1)
    scores[3] += 0.6
    return scores, write_scores(tmp_path / "planted.csv", scores)


@pytest.fixture
def noise_file(tmp_path):
    scores = noise_scores(40, 120, seed=2)
    return scores, write_scores(tmp_path / "noise.csv", scores)


# ---------------------------------------------------------------------------
# luckgate dsr
# ---------------------------------------------------------------------------


def test_dsr_from_summary_numbers_reproduces_the_paper(capsys):
    code, out, _ = run(PAPER_ARGS, capsys)
    assert code == 0
    report = json.loads(out)
    assert report["dsr"] == pytest.approx(0.9004, abs=5e-5)
    assert report["expected_max_sharpe"] == pytest.approx(0.1132, abs=5e-5)


def test_dsr_excess_kurtosis_flag(capsys):
    args = [a if a != "10" else "7" for a in PAPER_ARGS] + ["--kurtosis-kind", "excess"]
    code, out, _ = run(args, capsys)
    assert code == 0
    assert json.loads(out)["kurtosis"] == 10.0
    assert json.loads(out)["dsr"] == pytest.approx(0.9004, abs=5e-5)


def test_dsr_from_a_series_file(tmp_path, planted_file, capsys):
    scores, scores_path = planted_file
    series_path = tmp_path / "winner.txt"
    np.savetxt(series_path, scores[3], fmt="%.17g")  # one value per line
    code, out, _ = run(
        ["dsr", str(series_path), "--n-trials", "40", "--scores", scores_path], capsys
    )
    assert code == 0
    expected = dsr_report(scores[3], n_trials=40, sr_std=trial_sharpe_std(scores))
    assert json.loads(out)["dsr"] == pytest.approx(expected.dsr, abs=1e-6)


@pytest.mark.parametrize(
    "extra",
    [
        ["--sr", "0.1", "--n-obs", "100", "--n-trials", "5"],  # no --sr-std or --scores
        ["--n-trials", "5", "--sr-std", "0.1"],  # no file and no --sr
        ["--sr", "0.1", "--n-trials", "5", "--sr-std", "0.1"],  # --sr without --n-obs
        ["--sr", "0.1", "--n-obs", "100", "--n-trials", "5", "--sr-std", "-1"],
        [
            "--sr",
            "0.1",
            "--n-obs",
            "100",
            "--n-trials",
            "5",
            "--sr-std",
            "0.1",
            "--kurtosis",
            "0",
        ],  # excess kurtosis passed as raw
    ],
)
def test_dsr_bad_input_exits_2(extra, capsys):
    code, out, err = run(["dsr", *extra], capsys)
    assert code == 2
    assert out == ""
    assert err.startswith("luckgate: error:")


# ---------------------------------------------------------------------------
# luckgate pbo
# ---------------------------------------------------------------------------


def test_pbo_matches_the_library(planted_file, capsys):
    scores, path = planted_file
    code, out, _ = run(["pbo", path, "--blocks", "8"], capsys)
    assert code == 0
    report = json.loads(out)
    assert report["pbo"] == pytest.approx(cscv_pbo(scores, n_blocks=8).pbo, abs=1e-6)
    assert report["n_splits"] == 70
    assert "logits" not in report


def test_pbo_transpose_header_and_arrays(tmp_path, planted_file, capsys):
    scores, path = planted_file
    flipped = write_scores(tmp_path / "flipped.csv", scores.T, header=True)
    _, plain, _ = run(["pbo", path], capsys)
    _, from_flipped, _ = run(["pbo", flipped, "--transpose"], capsys)
    assert plain == from_flipped
    _, with_arrays, _ = run(["pbo", path, "--blocks", "4", "--arrays"], capsys)
    assert len(json.loads(with_arrays)["logits"]) == 6


def test_pbo_sampled_splits(planted_file, capsys):
    _, path = planted_file
    code, out, _ = run(["pbo", path, "--max-splits", "100", "--seed", "5"], capsys)
    assert code == 0
    assert json.loads(out)["n_splits"] == 100


def test_pbo_bad_file_exits_2(tmp_path, capsys):
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("1,2,3\n4,5\n", encoding="utf-8")
    assert run(["pbo", str(ragged)], capsys)[0] == 2
    words = tmp_path / "words.csv"
    words.write_text("a,b\n1,2\nx,y\n", encoding="utf-8")
    assert run(["pbo", str(words)], capsys)[0] == 2
    assert run(["pbo", str(tmp_path / "missing.csv")], capsys)[0] == 2


def test_read_table_skips_comments_blank_lines_and_one_header(tmp_path):
    path = tmp_path / "table.txt"
    path.write_text("# a comment\ntrial_a trial_b\n\n1 2\n3 4\n", encoding="utf-8")
    np.testing.assert_array_equal(read_table(str(path)), [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    "text",
    [
        "1.0,2.O,3.0\n4.0,5.0,6.0\n",  # a letter O instead of a zero in the first row
        "1,2,\n3,4,\n",  # a trailing comma leaves an empty cell
    ],
)
def test_read_table_never_drops_a_first_row_of_data(tmp_path, text):
    """Only a line with no numbers at all is a header. A first row of data
    with one bad cell is an error, not a header to skip."""
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_table(str(path))


def test_read_table_ignores_a_byte_order_mark(tmp_path):
    path = tmp_path / "exported.csv"
    path.write_text("1,2\n3,4\n", encoding="utf-8-sig")  # starts with a byte-order mark
    np.testing.assert_array_equal(read_table(str(path)), [[1.0, 2.0], [3.0, 4.0]])


# ---------------------------------------------------------------------------
# luckgate check
# ---------------------------------------------------------------------------


def test_check_passes_with_exit_0(planted_file, capsys):
    scores, path = planted_file
    code, out, _ = run(["check", path, "--n-trials", "40"], capsys)
    report = json.loads(out)
    assert code == 0
    assert report["verdict"] == "PASS"
    assert report["inputs"]["winner_row"] == 3
    library = Checklist().run(scores=scores, n_trials=40)
    assert report["checks"][1]["value"] == pytest.approx(library["dsr"].value, abs=1e-6)


def test_check_fails_with_exit_1(noise_file, capsys):
    _, path = noise_file
    code, out, _ = run(["check", path, "--n-trials", "40"], capsys)
    assert code == 1
    assert json.loads(out)["verdict"] == "FAIL"


def test_check_winner_row_winner_file_and_segments(tmp_path, planted_file, capsys):
    scores, path = planted_file
    code, out, _ = run(["check", path, "--n-trials", "40", "--winner-row", "0"], capsys)
    assert code == 1
    assert json.loads(out)["inputs"]["winner_row"] == 0

    winner = tmp_path / "winner.csv"
    winner.write_text(",".join(f"{v:.17g}" for v in scores[3]) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(["first"] * 60 + ["second"] * 60) + "\n", encoding="utf-8")
    config = tmp_path / "gate.json"
    config.write_text(json.dumps({"checks": ["dsr", "segments"]}), encoding="utf-8")
    code, out, _ = run(
        [
            "check",
            path,
            "--n-trials",
            "40",
            "--winner",
            str(winner),
            "--segments",
            str(labels),
            "--config",
            str(config),
        ],
        capsys,
    )
    report = json.loads(out)
    assert code == 0
    assert [check["name"] for check in report["checks"]] == ["dsr", "segments"]
    assert report["inputs"]["winner_row"] == 3


@pytest.mark.parametrize(
    "extra",
    [
        ["--n-trials", "5"],  # fewer than the 40 rows in the file
        ["--n-trials", "40", "--winner-row", "99"],
        ["--n-trials", "40", "--winner-row", "1", "--winner", "x.csv"],
    ],
)
def test_check_bad_input_exits_2(extra, planted_file, capsys):
    _, path = planted_file
    assert run(["check", path, *extra], capsys)[0] == 2


def test_check_config_with_a_typo_exits_2(tmp_path, planted_file, capsys):
    _, path = planted_file
    config = tmp_path / "gate.json"
    config.write_text(json.dumps({"max_pbbo": 0.1}), encoding="utf-8")
    code, _, err = run(["check", path, "--n-trials", "40", "--config", str(config)], capsys)
    assert code == 2
    assert "max_pbbo" in err


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ('{"checks": []}', "no checks to run"),  # would otherwise PASS with zero checks
        ('{"clip_positive_skew": "false"}', "clip_positive_skew must be True or False"),
        ('{"checks": ["segments"], "segment_floor": NaN}', "segment_floor must be finite"),
    ],
)
def test_check_bad_config_values_exit_2(tmp_path, planted_file, capsys, config, message):
    _, path = planted_file
    config_path = tmp_path / "gate.json"
    config_path.write_text(config, encoding="utf-8")
    code, out, err = run(["check", path, "--n-trials", "40", "--config", str(config_path)], capsys)
    assert code == 2
    assert out == ""
    assert message in err


def test_check_warns_about_an_sr_std_below_the_measured_one(planted_file, capsys):
    _, path = planted_file
    with pytest.warns(UserWarning, match="sr_std=0.001 is below"):
        _, out, _ = run(["check", path, "--n-trials", "40", "--sr-std", "0.001"], capsys)
    assert json.loads(out)["inputs"]["sr_std"] == 0.001


def test_python_dash_m_and_version():
    done = subprocess.run(
        [sys.executable, "-m", "luckgate", "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert done.stdout.strip().startswith("luckgate 0.1.0")


def test_python_dash_m_runs_the_paper_example():
    done = subprocess.run(
        [sys.executable, "-m", "luckgate", *PAPER_ARGS],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0
    assert json.loads(done.stdout)["dsr"] == pytest.approx(0.9004, abs=5e-5)
