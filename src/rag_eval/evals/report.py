"""Render an eval result as a compact markdown report."""

from __future__ import annotations

from typing import Any


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_report(result: dict[str, Any], *, max_failures: int = 10) -> str:
    run = result["run"]
    lines = [
        f"# RAG eval report — `{run['golden_set']}`",
        "",
        f"- **Items:** {run['n_items']} ({run['n_answerable']} answerable, "
        f"{run['n_errors']} errors)",
        f"- **Commit:** `{run.get('git_sha') or 'n/a'}` · **Created:** {run['created_at']}",
    ]
    config = run.get("config") or {}
    if config:
        lines.append("- **Config:** " + ", ".join(f"{k}=`{v}`" for k, v in sorted(config.items())))

    lines += ["", "## Metrics", "", "| metric | value |", "|---|---:|"]
    lines += [f"| `{name}` | {_fmt(value)} |" for name, value in result["metrics"].items()]

    if result.get("timings"):
        lines += ["", "## Latency", "", "| stage | ms |", "|---|---:|"]
        lines += [f"| `{name}` | {value:.1f} |" for name, value in result["timings"].items()]

    items = result["items"]
    k_cols = sorted(
        {n for i in items for n in i["metrics"] if n.startswith("retrieval.recall@")},
        key=lambda n: int(n.rsplit("@", 1)[1]),
    )
    recall_col = k_cols[-1] if k_cols else None
    lines += [
        "",
        "## Per item",
        "",
        f"| id | {recall_col or 'recall'} | mrr | citations valid | judge |",
        "|---|---:|---:|---:|:---:|",
    ]
    for item in items:
        m = item["metrics"]
        judge = item.get("judge")
        verdict = "-" if judge is None else ("pass" if judge["passed"] else "fail")
        lines.append(
            f"| `{item['id']}` | {_fmt(m.get(recall_col) if recall_col else None)} | "
            f"{_fmt(m.get('retrieval.mrr'))} | {_fmt(m.get('generation.citation_validity'))} | "
            f"{verdict} |"
        )

    failures = [
        i
        for i in items
        if "error" in i
        or (i["metrics"].get("retrieval.mrr") == 0.0)
        or (i.get("judge") and not i["judge"]["passed"])
    ][:max_failures]
    if failures:
        lines += ["", "## Failures to look at", ""]
        for item in failures:
            reason = item.get("error") or (
                "no relevant chunk retrieved"
                if item["metrics"].get("retrieval.mrr") == 0.0
                else f"judge: {item['judge']['rationale']}"
            )
            lines.append(f"- `{item['id']}` — {item['question']}  \n  {reason}")
    return "\n".join(lines) + "\n"
