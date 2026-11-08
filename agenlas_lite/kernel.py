"""The kernel: the only path through which agents call models, tools, humans and each other.

Agents decide *what* to do; the kernel decides whether it is allowed. Every request passes
`check()`, is charged against the budget and is appended to the audit log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .artifacts import Artifact, ArtifactStore
from .guard import verdict
from . import roles, tools


class PolicyError(Exception):
    """Raised when an agent asks for something the pack policy does not allow."""


class Kernel:
    def __init__(self, pack: dict, model, ask_human: Callable[[str, str, list[str]], str], workdir: Path, offline: bool):
        self.pack = pack
        self.policy = pack["policy"]
        self.model = model
        self.ask_human_fn = ask_human
        self.workdir = workdir
        self.offline = offline
        self.store = ArtifactStore(workdir / "artifacts")
        self.events: list[dict] = []
        self.gate_decisions: dict[str, str] = {}
        self.verdicts: dict[str, dict] = {}
        self.model_calls = 0
        self.tokens = 0
        self.listeners: list[Callable[[dict], None]] = []

    # ---- audit -------------------------------------------------------------------------
    def log(self, actor: str, action: str, **detail) -> None:
        event = {"t": round(time.time(), 3), "actor": actor, "action": action, **detail}
        self.events.append(event)
        with (self.workdir / "audit.jsonl").open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        for fn in self.listeners:
            fn(event)

    def check(self, actor: str, action: str, target: str) -> None:
        allowed, reason = self._decide(actor, action, target)
        self.log(actor, "check", request=action, target=target, allowed=allowed, reason=reason)
        if not allowed:
            raise PolicyError(f"{actor} may not {action} {target}: {reason}")

    def _decide(self, actor: str, action: str, target: str) -> tuple[bool, str]:
        if action == "model":
            if self.model_calls >= self.policy["max_model_calls"]:
                return False, "model call budget exhausted"
            if self.tokens >= self.policy["max_tokens"]:
                return False, "token budget exhausted"
            return True, "within budget"
        if action == "tool":
            if target not in self.pack["roles"][actor]["tools"]:
                return False, "tool not granted to this role"
            return True, "granted by pack"
        if action == "net":
            host = target.split("/")[2]
            return (host in self.policy["net_allow"], f"host {host}")
        if action == "code":
            bad = [p for p in self.policy["code_deny"] if p in target]
            return (not bad, f"denied patterns {bad}" if bad else "no denied patterns")
        if action in ("call", "finish"):
            missing = [r for r in self.pack["requires"].get(target, []) if not self._satisfied(r)]
            if action == "call" and target not in self.pack["roles"]:
                return False, "no such role in pack"
            return (not missing, f"missing {missing}" if missing else "preconditions met")
        if action == "gate":
            gate = self.pack["gates"].get(target)
            if gate is None:
                return False, "no such gate"
            if self.store.latest(gate["after"]) is None:
                return False, f"gate needs a {gate['after']} artifact first"
            return True, "gate input ready"
        return False, "unknown action"

    def _satisfied(self, requirement: str) -> bool:
        if requirement.startswith("gate:"):
            decision = self.gate_decisions.get(requirement[5:], "")
            return bool(decision) and not decision.startswith("revise")
        art = self.store.latest(requirement)
        if art is None:
            return False
        return requirement != "draft" or self.verdicts[art.id]["pass"]

    # ---- services offered to agents ----------------------------------------------------
    def complete(self, actor: str, messages: list[dict]) -> str:
        self.check(actor, "model", self.model.name)
        text, used = self.model.complete(actor, messages)
        self.model_calls += 1
        self.tokens += used
        self.log(actor, "model", tokens=used, chars=len(text))
        return text

    def tool(self, actor: str, name: str, **args):
        self.check(actor, "tool", name)
        if name == "arxiv_search":
            url = tools.arxiv_url(args["query"], args.get("limit", 5))
            self.check(actor, "net", url)
            result = tools.arxiv_search(url, self.offline)
        elif name == "run_python":
            self.check(actor, "code", args["code"])
            result = tools.run_python(args["code"], args["files"], self.workdir / "sandbox", self.policy["run_timeout_s"])
        self.log(actor, "tool", tool=name, summary=str(result)[:200])
        return result

    def call_agent(self, caller: str, role: str, task: str, inputs: list[str]) -> Artifact:
        """Runs `role` on `task` with the given input artifacts and stores its output."""
        self.check(caller, "call", role)
        self.log(caller, "spawn", role=role, task=task, inputs=inputs)
        spec = self.pack["roles"][role]
        content, extra_parents = roles.run(self, role, spec, task, [self.store.get(i) for i in inputs])
        art = self.store.put(spec["output"], role, content, inputs + extra_parents)
        self.log(role, "artifact", id=art.id, kind=art.kind, parents=art.parents)
        if art.kind == "draft":
            self.verdicts[art.id] = verdict(self.store, art)
            self.log("guard", "verdict", id=art.id, **self.verdicts[art.id])
        return art

    def human_gate(self, caller: str, gate: str, options: list[str]) -> Artifact:
        """Ends autonomous work until the human decides; the decision becomes an artifact agents can cite."""
        self.check(caller, "gate", gate)
        question = self.pack["gates"][gate]["question"]
        self.log(caller, "gate_open", gate=gate, question=question, options=options)
        decision = self.ask_human_fn(gate, question, options)
        self.gate_decisions[gate] = decision
        basis = self.store.latest(self.pack["gates"][gate]["after"])
        art = self.store.put("decision", "human", {"gate": gate, "question": question, "decision": decision}, [basis.id])
        self.log("human", "gate_decision", gate=gate, decision=decision, id=art.id)
        return art
