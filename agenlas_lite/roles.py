"""Role runners. A pack names a runner per role; the runner turns (task, input artifacts) into content."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .artifacts import Artifact
from .llm import parse_json

if TYPE_CHECKING:
    from .kernel import Kernel


def render_inputs(inputs: list[Artifact]) -> str:
    return "\n\n".join(
        f"[{a.id}] ({a.kind}, by {a.producer})\n{json.dumps(a.content, ensure_ascii=False)[:40000]}" for a in inputs
    )


def ask(kernel: Kernel, role: str, system: str, task: str, inputs: list[Artifact]) -> str:
    user = f"Task: {task}\n\nInputs:\n{render_inputs(inputs)}"
    return kernel.complete(role, [{"role": "system", "content": system}, {"role": "user", "content": user}])


def run(kernel: Kernel, role: str, spec: dict, task: str, inputs: list[Artifact]) -> tuple[object, list[str]]:
    """Returns (content, extra parent ids) for the artifact the role produces."""
    runner = spec["runner"]
    if runner == "llm":
        reply = ask(kernel, role, spec["system"], task, inputs)
        return (reply if spec["output"] == "draft" else parse_json(reply)), []
    if runner == "search":
        return run_search(kernel, role, spec, task, inputs)
    if runner == "roundtable":
        return run_roundtable(kernel, role, spec, task, inputs), []
    if runner == "experiment":
        return run_experiment(kernel, role, spec, task, inputs)
    raise ValueError(f"pack names unknown runner {runner!r} for role {role}")


def run_search(kernel: Kernel, role: str, spec: dict, task: str, inputs: list[Artifact]):
    brief = next(a for a in inputs if a.kind == "brief")
    papers = kernel.tool(role, "arxiv_search", query=" ".join(brief.content["keywords"][:4]), limit=8)
    source = kernel.store.put("papers", role, papers, [brief.id])
    kernel.log(role, "artifact", id=source.id, kind=source.kind, parents=source.parents)
    summary = parse_json(ask(kernel, role, spec["system"], task, inputs + [source]))
    return summary, [source.id]


def run_roundtable(kernel: Kernel, role: str, spec: dict, task: str, inputs: list[Artifact]) -> dict:
    proposals = []
    for expert in spec["experts"]:
        reply = ask(kernel, role, f"{spec['system']} You speak as the {expert}.", task, inputs)
        for idea in parse_json(reply)["ideas"]:
            proposals.append({**idea, "proposed_by": expert})
        kernel.log(role, "roundtable_turn", expert=expert, ideas=len(proposals))
    return {"ideas": proposals}


def run_experiment(kernel: Kernel, role: str, spec: dict, task: str, inputs: list[Artifact]):
    papers = next(a for a in kernel.store.items.values() if a.kind == "papers")
    reply = ask(kernel, role, spec["system"], task, inputs + [papers])
    code = reply.split("```python")[1].split("```")[0] if "```python" in reply else reply
    result = kernel.tool(role, "run_python", code=code, files={"papers.json": papers.content})
    return {"code": code, "metrics": result["metrics"], "returncode": result["returncode"], "stdout": result["stdout"], "stderr": result["stderr"]}, [papers.id]
