"""Versioned artifact store: every output is immutable and records what it was derived from."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Artifact:
    id: str
    kind: str
    producer: str
    content: object
    parents: list[str] = field(default_factory=list)
    sha256: str = ""
    created_at: float = 0.0


class ArtifactStore:
    """Append-only store. Ids are sequential (`A1`, `A2`, ...) so drafts can cite them as `[A3]`."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, Artifact] = {}

    def put(self, kind: str, producer: str, content: object, parents: list[str]) -> Artifact:
        for p in parents:
            if p not in self.items:
                raise KeyError(f"unknown parent artifact {p}")
        art_id = f"A{len(self.items) + 1}"
        body = json.dumps(content, ensure_ascii=False, sort_keys=True)
        art = Artifact(
            id=art_id,
            kind=kind,
            producer=producer,
            content=content,
            parents=list(parents),
            sha256=hashlib.sha256(body.encode()).hexdigest(),
            created_at=time.time(),
        )
        self.items[art_id] = art
        (self.root / f"{art_id}.json").write_text(json.dumps(asdict(art), ensure_ascii=False, indent=2))
        return art

    def get(self, art_id: str) -> Artifact:
        return self.items[art_id]

    def latest(self, kind: str) -> Artifact | None:
        found = [a for a in self.items.values() if a.kind == kind]
        return found[-1] if found else None

    def ancestors(self, art_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.items[art_id].parents)
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(self.items[cur].parents)
        return seen
