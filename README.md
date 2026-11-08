# agenlas-lite

A small demo of a virtual research team. You give it one research question; a group of AI agents with different roles searches the literature, proposes ideas, runs a small experiment and drafts a short paper. You only decide at a few key points: which idea to pursue, and whether the draft is ready.

This repository is a compact demo written for a student competition. It is not the Agenlas product.

## Design ideas

- **A team built for the goal.** The agents decide the next step at run time instead of following a fixed script.
- **The human decides at the gates.** The team cannot skip a human decision, and every decision is recorded.
- **Everything is traceable.** Every number in the draft links back to the experiment or paper it came from.
- **Bounded by design.** Each run stays within a budget, and actions outside what the scenario allows are refused.

## Quick start

Requires Python 3.10 or newer. No third-party packages.

```sh
git clone https://github.com/Chranos/agenlas_lite.git
cd agenlas_lite

# Offline demo: no API key, no network.
python3 -m agenlas_lite --offline --auto "How well can lightweight lexical retrieval ground citations in AI-written related-work sections?"

# With a model (any OpenAI-compatible endpoint, DeepSeek by default). Gates ask you in the terminal.
export AGENLAS_API_KEY=sk-...
python3 -m agenlas_lite "your one-sentence research question"
```

Each run writes `runs/<timestamp>/`. Open `report.html` in a browser to see the timeline, the human decisions, the draft and where each of its numbers came from; add `?replay=1` to the address to replay the run. The draft is also saved as `paper.md`.

Options: `--offline`, `--auto` (answer each gate with its first option), `--out DIR`.
