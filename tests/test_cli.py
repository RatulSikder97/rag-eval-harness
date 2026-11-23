from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_eval.cli import app

from .conftest import CORPUS, GOLDEN

runner = CliRunner()


def test_ingest_then_query(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", "--corpus", str(CORPUS)])
    assert result.exit_code == 0, result.output
    assert "Indexed" in result.output
    assert (tmp_path / ".rag-index.json").exists()

    result = runner.invoke(app, ["query", "When are production deploys frozen?", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["retrieved"][0]["id"].startswith("deployments.md#")
    assert payload["answer"]["citations"]

    result = runner.invoke(app, ["query", "canary traffic", "--no-generate", "--k", "2"])
    assert result.exit_code == 0
    assert result.output.count("\n") == 2


def test_query_without_index_fails_cleanly() -> None:
    result = runner.invoke(app, ["query", "anything"])
    assert result.exit_code == 2
    assert "rag-eval ingest" in result.output


def test_run_and_compare_gate(tmp_path: Path) -> None:
    common = ["--golden", str(GOLDEN), "--corpus", str(CORPUS)]
    good = runner.invoke(app, ["run", *common, "--out", "base.json", "--report", "r.md"])
    assert good.exit_code == 0, good.output
    assert (tmp_path / "r.md").read_text().startswith("# RAG eval report")

    worse = runner.invoke(
        app,
        [
            "run",
            *common,
            "--out",
            "worse.json",
            "--no-report-md",
            "--chunk-size",
            "150",
            "--chunk-overlap",
            "20",
        ],
    )
    assert worse.exit_code == 0, worse.output

    same = runner.invoke(app, ["compare", "base.json", "base.json", "--fail-on-regression", "0.02"])
    assert same.exit_code == 0
    assert "No regressions." in same.output

    gate = runner.invoke(
        app,
        ["compare", "base.json", "worse.json", "--fail-on-regression", "0.02", "--summary", "s.md"],
    )
    assert gate.exit_code == 1
    assert "REGRESSED" in gate.output
    assert "REGRESSED" in (tmp_path / "s.md").read_text()

    report_only = runner.invoke(app, ["compare", "base.json", "worse.json"])
    assert report_only.exit_code == 0


@pytest.mark.parametrize(
    "args",
    [
        ["run", "--embedder", "nope"],
        ["run", "--embedder", "openai"],
        ["run", "--mode", "fuzzy"],
        ["run", "--chunk-size", "100", "--chunk-overlap", "100"],
    ],
)
def test_bad_configuration_is_reported(args: list[str]) -> None:
    result = runner.invoke(app, [*args, "--golden", str(GOLDEN), "--corpus", str(CORPUS)])
    assert result.exit_code == 2
