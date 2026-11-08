"""Tools agents can request through the kernel: literature search and a confined Python runner."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "packs" / "research" / "fixtures" / "papers.json"
ATOM = "{http://www.w3.org/2005/Atom}"


def arxiv_url(query: str, limit: int) -> str:
    q = urllib.parse.quote(f"all:{query}")
    return f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results={limit}"


def arxiv_search(url: str, offline: bool) -> list[dict]:
    """Returns [{title, abstract, url}]. Offline mode reads the bundled fixture instead of the network."""
    if offline:
        return json.loads(FIXTURE.read_text())
    with urllib.request.urlopen(url, timeout=30) as resp:
        root = ET.fromstring(resp.read())
    return [
        {
            "title": " ".join(e.findtext(f"{ATOM}title", "").split()),
            "abstract": " ".join(e.findtext(f"{ATOM}summary", "").split()),
            "url": e.findtext(f"{ATOM}id", ""),
        }
        for e in root.iter(f"{ATOM}entry")
    ]


def run_python(code: str, files: dict[str, object], sandbox: Path, timeout_s: int) -> dict:
    """Runs `code` in a fresh directory with only `files` present, isolated mode, no inherited env."""
    sandbox.mkdir(parents=True, exist_ok=True)
    run_dir = sandbox / f"run{len(list(sandbox.iterdir())) + 1}"
    run_dir.mkdir()
    for name, data in files.items():
        (run_dir / name).write_text(json.dumps(data, ensure_ascii=False))
    (run_dir / "main.py").write_text(code)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "main.py"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={"PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"killed after {timeout_s}s", "metrics": None, "dir": str(run_dir)}
    lines = proc.stdout.strip().splitlines()
    metrics = json.loads(lines[-1]) if proc.returncode == 0 and lines else None
    return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:], "metrics": metrics, "dir": str(run_dir)}
