"""Command line: `python -m agenlas_lite "your research question"`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import orchestrator, report
from .kernel import Kernel
from .llm import ChatModel, ScriptedModel

PACKS = Path(__file__).resolve().parent.parent / "packs"
COLORS = {"orchestrator": 36, "kernel": 31, "guard": 35, "human": 33, "user": 33}


def console_gate(gate: str, question: str, options: list[str]) -> str:
    print(f"\n\033[1;33m== Human gate: {gate} ==\033[0m\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    raw = input("Your choice (number, or free text): ").strip()
    return options[int(raw) - 1] if raw.isdigit() else raw


def auto_gate(gate: str, question: str, options: list[str]) -> str:
    print(f"\033[1;33m[gate {gate}] auto-approved: {options[0]}\033[0m")
    return options[0]


def show(event: dict) -> None:
    actor, action = event["actor"], event["action"]
    if action == "check" and event["allowed"]:
        return
    color = COLORS.get(actor, 32)
    detail = {k: v for k, v in event.items() if k not in ("t", "actor", "action")}
    print(f"\033[{color}m{actor:>12}\033[0m {action:<14} {json.dumps(detail, ensure_ascii=False)[:150]}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="agenlas-lite", description="Run a virtual research team on one question.")
    ap.add_argument("goal", help="one-sentence research question")
    ap.add_argument("--pack", default="research", help="scenario pack under packs/")
    ap.add_argument("--offline", action="store_true", help="use the scripted model and bundled papers (no API key)")
    ap.add_argument("--auto", action="store_true", help="answer every human gate with its first option")
    ap.add_argument("--out", default="runs", help="directory for run outputs")
    args = ap.parse_args()

    pack = json.loads((PACKS / args.pack / "pack.json").read_text())
    workdir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    workdir.mkdir(parents=True)
    model = ScriptedModel() if args.offline else ChatModel()
    kernel = Kernel(pack, model, auto_gate if args.auto else console_gate, workdir, offline=args.offline)
    kernel.listeners.append(show)

    finished = orchestrator.run(kernel, args.goal)
    out = report.write(kernel, args.goal, finished)
    print(f"\nfinished={finished}  model_calls={kernel.model_calls}  tokens={kernel.tokens}  artifacts={len(kernel.store.items)}")
    print(f"report: {out}")
