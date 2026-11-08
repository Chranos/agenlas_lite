"""Answers for the offline model. They read the same prompts a real model gets and reply in the same formats,
so the kernel, guard, gates and report run unchanged without an API key. The script tries one premature step
on purpose so the kernel's refusal shows up in the demo."""

from __future__ import annotations

import json
import re

INPUT = re.compile(r"^\[(A\d+)\] \((\w+), by [\w-]+\)\n(.*)$", re.M)

EXPERIMENT = '''import json, math, re
from collections import Counter

papers = json.load(open("papers.json"))
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
docs = [tok(p["abstract"]) for p in papers]
df = Counter(w for d in docs for w in set(d))
idf = {w: math.log(len(docs) / c) + 1 for w, c in df.items()}

def tfidf(words):
    v = Counter(words)
    return {w: n * idf.get(w, 1.0) for w, n in v.items()}

def cosine(a, b):
    dot = sum(x * b.get(w, 0) for w, x in a.items())
    na = math.sqrt(sum(x * x for x in a.values())); nb = math.sqrt(sum(x * x for x in b.values()))
    return dot / (na * nb) if na and nb else 0.0

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0

doc_vecs = [tfidf(d) for d in docs]
def evaluate(score):
    hits1, rr = 0, 0.0
    for i, p in enumerate(papers):
        q = tok(p["title"])
        ranked = sorted(range(len(docs)), key=lambda j: -score(q, j))
        rank = ranked.index(i) + 1
        hits1 += rank == 1
        rr += 1 / rank
    return round(hits1 / len(papers), 3), round(rr / len(papers), 3)

t1, tm = evaluate(lambda q, j: cosine(tfidf(q), doc_vecs[j]))
j1, jm = evaluate(lambda q, j: jaccard(q, docs[j]))
print(json.dumps({"n_papers": len(papers), "tfidf_recall@1": t1, "tfidf_mrr": tm, "jaccard_recall@1": j1, "jaccard_mrr": jm}))
'''


def inputs(text: str) -> dict[str, tuple[str, object]]:
    found = {}
    for art_id, kind, body in INPUT.findall(text):
        if kind in ("brief", "ideas", "evidence", "decision", "related_work", "draft"):
            found[kind] = (art_id, json.loads(body))
        else:
            found[kind] = (art_id, None)
    return found


def answer(role: str, messages: list[dict]) -> str:
    system, user = messages[0]["content"], messages[-1]["content"]
    if role == "orchestrator":
        return json.dumps(orchestrate(json.loads(user)))
    task = user.split("\n\nInputs:")[0].removeprefix("Task: ")
    got = inputs(user)
    if role == "intake":
        return json.dumps({
            "question": task,
            "scope": "Lexical retrieval baselines only; arXiv abstracts as the corpus; no model training.",
            "keywords": ["citation hallucination", "retrieval augmented generation", "claim source retrieval", "grounding"],
            "success_criteria": ["a measurable retrieval metric on real abstracts", "an honest comparison of two baselines"],
        })
    if role == "scout":
        titles = re.findall(r'"title": "([^"]+)"', user)
        return json.dumps({
            "summary": f"Found {len(titles)} papers. Work on citation-grounded generation (e.g. '{next(t for t in titles if 'Citation' in t)}') "
                       "and claim-source retrieval shows that matching a claim to its source is the step where AI writing tools fail.",
            "gaps": ["few cheap baselines for checking whether a cited paper matches the citing sentence", "little evidence on how far lexical matching goes"],
        })
    if role == "roundtable":
        expert = system.split("You speak as the ")[1].rstrip(".")
        ideas = {
            "methods expert": [{"title": "Lexical citation check", "hypothesis": "TF-IDF matching of a citing sentence to the cited abstract ranks the true source first more often than word overlap", "experiment": "title-to-abstract retrieval on arXiv abstracts, recall@1 and MRR"}],
            "data expert": [{"title": "Citation-noise benchmark", "hypothesis": "injected wrong citations can be detected from abstract similarity", "experiment": "swap 20% of citations and measure detection AUC"}],
            "skeptic": [{"title": "Where lexical matching fails", "hypothesis": "paraphrased claims defeat lexical retrieval", "experiment": "compare exact-title queries against paraphrased queries"}],
        }[expert]
        return json.dumps({"ideas": ideas})
    if role == "worker":
        return f"```python\n{EXPERIMENT}```"
    if role == "writer":
        return write(task, got)
    if role == "reviewer":
        return json.dumps({
            "scores": {"clarity": 4, "novelty": 2, "evidence": 4, "limitations": 4},
            "fixes": ["test paraphrased queries, not only exact titles", "report variance over several corpora"],
        })
    raise ValueError(f"no script for role {role}")


def write(task: str, got: dict) -> str:
    ev_id, ev = got["evidence"]
    pa_id = got["papers"][0]
    m = ev["metrics"]
    revised = "guard" in task.lower() or "revise" in task.lower()
    first_claim = (
        f"On {m['n_papers']} real arXiv abstracts, TF-IDF ranks the true source first for {m['tfidf_recall@1']} of queries "
        f"(MRR {m['tfidf_mrr']}) against {m['jaccard_recall@1']} for word overlap (MRR {m['jaccard_mrr']}) [{ev_id}]."
    )
    intro = "Prior work reports that cited sources are often wrong in model-written text" + (f" [{pa_id}]." if revised else ", in up to 30.0% of cases.")
    return "\n".join([
        "# Checking Citations with Lexical Retrieval: A Small Baseline Study",
        "",
        "## Abstract",
        f"{intro} We ask how far a lexical retriever goes in matching a citing sentence to its source. {first_claim}",
        "",
        "## Method",
        f"Each paper title is used as a query against all abstracts in the corpus [{pa_id}]. We compare TF-IDF cosine similarity with Jaccard word overlap and report recall at rank one and mean reciprocal rank [{ev_id}].",
        "",
        "## Results",
        first_claim,
        "",
        "## Limitations",
        "Titles share vocabulary with their own abstracts, so this setting is easier than real citing sentences. The corpus is small and comes from three keyword searches.",
    ])


def orchestrate(s: dict) -> dict:
    arts = s["artifacts"]
    latest = {a["kind"]: a for a in arts}
    ids = lambda *kinds: [latest[k]["id"] for k in kinds if k in latest]
    newer = lambda a, b: arts.index(latest[a]) > arts.index(latest[b])
    gates = s["gate_decisions"]
    if "brief" not in latest:
        return {"action": "call", "role": "intake", "task": s["goal"], "inputs": []}
    if "related_work" not in latest:
        return {"action": "call", "role": "scout", "task": "Survey prior work for the brief.", "inputs": ids("brief")}
    if "ideas" not in latest:
        return {"action": "call", "role": "roundtable", "task": "Propose testable ideas that fit a one-day study.", "inputs": ids("brief", "related_work")}
    if "idea" not in gates:
        if s["last_error"] is None:
            return {"action": "call", "role": "worker", "task": "Start the first idea right away.", "inputs": ids("brief", "ideas")}
        titles = re.findall(r'"title": "([^"]+)"', latest["ideas"]["preview"])
        return {"action": "gate", "gate": "idea", "options": titles}
    if "evidence" not in latest:
        return {"action": "call", "role": "worker", "task": f"Run the chosen idea: {gates['idea']}", "inputs": ids("brief", "decision")}
    if "draft" not in latest:
        return {"action": "call", "role": "writer", "task": "Write the paper.", "inputs": ids("brief", "related_work", "papers", "evidence")}
    verdict = s["guard_verdicts"][latest["draft"]["id"]]
    if not verdict["pass"]:
        return {"action": "call", "role": "writer", "task": f"Revise the draft; the guard found: {verdict['problems']}", "inputs": ids("brief", "related_work", "papers", "evidence", "draft")}
    final = gates.get("final", "")
    if final.startswith("revise") and newer("decision", "draft"):
        return {"action": "call", "role": "writer", "task": f"Revise the draft as the human asked: {final}", "inputs": ids("brief", "related_work", "papers", "evidence", "draft", "review", "decision")}
    if "review" not in latest or newer("draft", "review"):
        return {"action": "call", "role": "reviewer", "task": "Review the draft against the rubric.", "inputs": ids("draft")}
    if not final or (final.startswith("revise") and newer("review", "decision")):
        return {"action": "gate", "gate": "final", "options": ["accept", "revise: address the reviewer's fixes"]}
    return {"action": "finish"}
