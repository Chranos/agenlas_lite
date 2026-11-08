"""The root agent. It is free to choose the next step; the kernel decides whether that step is allowed."""

from __future__ import annotations

import json

from .kernel import Kernel, PolicyError
from .llm import parse_json

SYSTEM = """You coordinate a virtual research team. You never do the work yourself: you call roles, open human gates, or finish.
Reply with exactly one JSON object, one of:
  {"action": "call", "role": <role>, "task": <instruction>, "inputs": [<artifact ids the role needs>]}
  {"action": "gate", "gate": <gate>, "options": [<choices for the human>]}
  {"action": "finish"}
The kernel rejects steps whose preconditions are not met and tells you why. A draft whose guard verdict fails must be rewritten."""


def state(kernel: Kernel, goal: str, last_error: str | None) -> dict:
    return {
        "goal": goal,
        "playbook": kernel.pack["playbook"],
        "roles": {name: spec["output"] for name, spec in kernel.pack["roles"].items()},
        "gates": list(kernel.pack["gates"]),
        "artifacts": [
            {"id": a.id, "kind": a.kind, "by": a.producer, "parents": a.parents, "preview": json.dumps(a.content, ensure_ascii=False)[:1200]}
            for a in kernel.store.items.values()
        ],
        "guard_verdicts": kernel.verdicts,
        "gate_decisions": kernel.gate_decisions,
        "last_error": last_error,
    }


def run(kernel: Kernel, goal: str) -> bool:
    """Drives the team until the orchestrator finishes; returns False when the step or model budget runs out."""
    kernel.log("user", "goal", goal=goal)
    last_error = None
    for step in range(kernel.policy["max_steps"]):
        prompt = json.dumps(state(kernel, goal, last_error), ensure_ascii=False)
        try:
            reply = kernel.complete("orchestrator", [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        except PolicyError as err:
            kernel.log("kernel", "stopped", error=str(err))
            return False
        last_error = None
        try:
            decision = parse_json(reply)
            kernel.log("orchestrator", "decide", step=step, decision=decision)
            if decision["action"] == "call":
                kernel.call_agent("orchestrator", decision["role"], decision["task"], decision.get("inputs", []))
            elif decision["action"] == "gate":
                kernel.human_gate("orchestrator", decision["gate"], decision["options"])
            elif decision["action"] == "finish":
                kernel.check("orchestrator", "finish", "finish")
                kernel.log("orchestrator", "finished")
                return True
            else:
                last_error = f"unknown action {decision['action']!r}"
        except (PolicyError, KeyError, ValueError, OSError) as err:
            last_error = str(err)
            kernel.log("kernel", "rejected", error=last_error)
    kernel.log("kernel", "step_budget_exhausted")
    return False
