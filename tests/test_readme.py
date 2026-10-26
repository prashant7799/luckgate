"""The README must tell the truth: its code runs and its outputs match."""

from __future__ import annotations

import contextlib
import io
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")


def section(heading: str) -> str:
    """The text after `heading`, up to the next heading of the same level or
    higher. Lines inside code fences are never headings."""
    level = len(heading.split(" ")[0])
    lines = README.splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith(heading))
    in_code = False
    collected = []
    for line in lines[start + 1 :]:
        if line.startswith("```"):
            in_code = not in_code
        elif not in_code and line.startswith("#"):
            this_level = len(line) - len(line.lstrip("#"))
            if this_level <= level and line[this_level : this_level + 1] == " ":
                break
        collected.append(line)
    return "\n".join(collected) + "\n"


def code_blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, flags=re.DOTALL)


def clean(text: str) -> list[str]:
    return [line.rstrip() for line in text.strip("\n").splitlines()]


def run_script(path: str) -> str:
    return subprocess.run(
        [sys.executable, path],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    ).stdout


def test_quickstart_runs_and_prints_what_the_readme_shows():
    text = section("## Quickstart (60 seconds)")
    code = code_blocks(text, "python")[0]
    shown = code_blocks(text, "text")[0]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, "README quickstart", "exec"), {})
    assert clean(printed.getvalue()) == clean(shown)


def test_quickstart_file_is_the_readme_code():
    code = code_blocks(section("## Quickstart (60 seconds)"), "python")[0]
    assert (REPO / "examples" / "quickstart.py").read_text(encoding="utf-8") == code


@pytest.mark.parametrize(
    "script", ["paper_example", "ab_variants", "forecast_models", "false_pass_rates"]
)
def test_example_output_matches_the_readme(script):
    link = f"(examples/{script}.py)"
    after_link = README[README.index(link) :]
    shown = code_blocks(after_link, "text")[0]
    assert clean(run_script(f"examples/{script}.py")) == clean(shown)


def test_custom_check_snippet_runs():
    text = section("### Checklist")
    code = code_blocks(text, "python")[1]
    namespace: dict = {}
    exec(compile(code, "README custom check", "exec"), namespace)
    import numpy as np

    scores = np.random.default_rng(0).normal(size=(10, 60))
    verdict = namespace["checklist"].run(scores=scores, n_trials=10)
    assert verdict.checks[-1].name == "worst_day"


def command_line(command: str) -> list[str]:
    """Turn a README command into one that runs here, with this Python."""
    words = shlex.split(command)
    if words[0] == "python":
        return [sys.executable, str(REPO / words[1]), *words[2:]]
    if words[0] == "luckgate":
        script = Path(sys.executable).with_name("luckgate")
        if script.exists():  # the installed console script
            return [str(script), *words[1:]]
        return [sys.executable, "-m", "luckgate", *words[1:]]
    raise ValueError(f"unexpected command {command!r}")


def test_command_line_outputs_match_the_readme(tmp_path):
    text = section("### Command line")
    pairs = re.findall(r"```bash\n(.*?)```\n(.*?)```text\n(.*?)```", text, flags=re.DOTALL)
    assert len(pairs) == 4
    for command, _between, shown in pairs:
        done = subprocess.run(
            command_line(command.strip()),
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stderr
        expected = clean(shown)
        if expected[-1] == "...":  # the README shows only the start of a long report
            expected = expected[:-1]
            assert clean(done.stdout)[: len(expected)] == expected
        else:
            assert clean(done.stdout) == expected
