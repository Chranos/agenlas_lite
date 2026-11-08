"""Model access: any OpenAI-compatible chat endpoint, or a scripted offline model for demos and tests."""

from __future__ import annotations

import json
import os
import urllib.request

from . import scripted


class ChatModel:
    """Calls `{base_url}/chat/completions`. Configure with AGENLAS_API_KEY, AGENLAS_BASE_URL, AGENLAS_MODEL."""

    def __init__(self) -> None:
        self.api_key = os.environ["AGENLAS_API_KEY"]
        self.base_url = os.environ.get("AGENLAS_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.environ.get("AGENLAS_MODEL", "deepseek-chat")
        self.name = self.model

    def complete(self, role: str, messages: list[dict]) -> tuple[str, int]:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.4}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"], data.get("usage", {}).get("total_tokens", 0)


class ScriptedModel:
    """Deterministic stand-in that answers each role from its prompt; no network, no key."""

    name = "scripted-offline"

    def complete(self, role: str, messages: list[dict]) -> tuple[str, int]:
        text = scripted.answer(role, messages)
        tokens = sum(len(m["content"]) for m in messages) // 4 + len(text) // 4
        return text, tokens


def parse_json(text: str) -> object:
    """Extracts the first JSON value from a model reply, tolerating ```json fences."""
    if "```" in text:
        text = text.split("```")[1].removeprefix("json")
    start = min(i for i in (text.find("{"), text.find("[")) if i >= 0)
    return json.JSONDecoder().raw_decode(text[start:])[0]
