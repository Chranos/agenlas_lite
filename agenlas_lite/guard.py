"""Alignment guard: a deterministic check that every factual sentence in a draft is backed by evidence.

Its verdict gates the flow (the kernel will not let a failing draft reach review); reviewer scores never do.
"""

from __future__ import annotations

import re

from .artifacts import Artifact, ArtifactStore

CITE = re.compile(r"\[(A\d+)\]")
NUMBER = re.compile(r"\d+\.\d+|\d+%")


def verdict(store: ArtifactStore, draft: Artifact) -> dict:
    """Passes when every sentence containing a number cites an artifact the draft was derived from."""
    allowed = store.ancestors(draft.id)
    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(draft.content)):
        cites = CITE.findall(sentence)
        unknown = [c for c in cites if c not in allowed]
        if unknown:
            problems.append(f"cites {unknown} which the draft was not derived from: {sentence[:80]}")
        elif NUMBER.search(sentence) and not cites:
            problems.append(f"uncited number: {sentence[:80]}")
    return {"pass": not problems, "problems": problems}
