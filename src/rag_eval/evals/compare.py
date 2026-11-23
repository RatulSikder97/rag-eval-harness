"""Compare two eval results and flag regressions — the CI gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Status = Literal["ok", "improved", "regressed", "missing", "new"]


@dataclass(frozen=True, slots=True)
class MetricDelta:
    name: str
    baseline: float | None
    current: float | None
    status: Status

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline


@dataclass(frozen=True, slots=True)
class Comparison:
    deltas: list[MetricDelta]
    threshold: float

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.status in ("regressed", "missing")]


def compare_results(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    threshold: float = 0.02,
    prefixes: tuple[str, ...] = (),
) -> Comparison:
    """All metrics are higher-is-better in [0, 1]; a drop larger than ``threshold`` fails.

    A metric present in the baseline but absent from the current run counts as a regression:
    silently losing a metric (e.g. a judge that started erroring) must not pass the gate.
    """
    base_m: dict[str, float] = baseline["metrics"]
    cur_m: dict[str, float] = current["metrics"]

    def wanted(name: str) -> bool:
        return not prefixes or name.startswith(prefixes)

    deltas: list[MetricDelta] = []
    for name in sorted(set(base_m) | set(cur_m)):
        if not wanted(name):
            continue
        b, c = base_m.get(name), cur_m.get(name)
        status: Status
        if b is None:
            status = "new"
        elif c is None:
            status = "missing"
        elif c < b - threshold - 1e-9:
            status = "regressed"
        elif c > b + threshold + 1e-9:
            status = "improved"
        else:
            status = "ok"
        deltas.append(MetricDelta(name, b, c, status))
    return Comparison(deltas=deltas, threshold=threshold)


_ICON: dict[Status, str] = {
    "ok": "ok",
    "improved": "improved",
    "regressed": "**REGRESSED**",
    "missing": "**MISSING**",
    "new": "new",
}


def render_comparison(cmp: Comparison) -> str:
    def f(v: float | None, signed: bool = False) -> str:
        if v is None:
            return "-"
        return f"{v:+.3f}" if signed else f"{v:.3f}"

    lines = [
        f"### Eval comparison (fail threshold: -{cmp.threshold:.3f})",
        "",
        "| metric | baseline | current | delta | status |",
        "|---|---:|---:|---:|---|",
    ]
    for d in cmp.deltas:
        lines.append(
            f"| `{d.name}` | {f(d.baseline)} | {f(d.current)} | {f(d.delta, True)} | "
            f"{_ICON[d.status]} |"
        )
    verdict = f"**{len(cmp.regressions)} regression(s)**" if cmp.regressions else "No regressions."
    lines += ["", verdict]
    return "\n".join(lines) + "\n"
