"""Run outputs: the final draft, a provenance report, and a self-contained HTML page with a replayable timeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .kernel import Kernel

PAGE = (Path(__file__).resolve().parent / "report.html").read_text()


def provenance(kernel: Kernel) -> dict:
    draft = kernel.store.latest("draft")
    return {
        "final_draft": draft.id if draft else None,
        "derived_from": sorted(kernel.store.ancestors(draft.id), key=lambda i: int(i[1:])) if draft else [],
        "guard_verdict": kernel.verdicts.get(draft.id) if draft else None,
        "gate_decisions": kernel.gate_decisions,
        "artifacts": [
            {"id": a.id, "kind": a.kind, "producer": a.producer, "parents": a.parents, "sha256": a.sha256}
            for a in kernel.store.items.values()
        ],
        "model_calls": kernel.model_calls,
        "tokens": kernel.tokens,
        "policy_rejections": [e for e in kernel.events if e["action"] == "check" and not e["allowed"]],
    }


def write(kernel: Kernel, goal: str, finished: bool) -> Path:
    """Writes paper.md, provenance.json and report.html into the run directory; returns the HTML path."""
    draft = kernel.store.latest("draft")
    if draft:
        (kernel.workdir / "paper.md").write_text(str(draft.content))
    prov = provenance(kernel)
    (kernel.workdir / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2))
    data = {
        "goal": goal,
        "finished": finished,
        "model": kernel.model.name,
        "events": kernel.events,
        "artifacts": [asdict(a) for a in kernel.store.items.values()],
        "provenance": prov,
    }
    out = kernel.workdir / "report.html"
    out.write_text(PAGE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    return out
