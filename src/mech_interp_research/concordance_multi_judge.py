"""Multi-judge discriminative concordance for the JumpReLU auto-interp run.

Re-validates ICD concordance with a cross-provider judge panel, a top-5
retrieval arm, a shuffled null, a human-adjudication sheet, and an independent
second explainer. Reuses a completed auto_interp run; no GPU.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Judge:
    """Uniform ``.complete(prompt) -> str`` over Anthropic and OpenRouter backends."""

    def __init__(self, slug, backend, model=None, client=None, max_retries=6):
        self.slug = slug
        self.backend = backend
        self.model = model
        self.max_retries = max_retries
        self._client = client

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        if self.backend == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        if self.backend == "openrouter":
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        raise ValueError(f"unknown backend {self.backend!r}")


def build_judges(judge_cfgs, *, anthropic_client=None, openrouter_client=None):
    """Build callable Judges, skipping ``backend == 'reuse'`` entries."""
    judges = []
    for cfg in judge_cfgs:
        backend = cfg["backend"]
        if backend == "reuse":
            continue
        client = anthropic_client if backend == "anthropic" else openrouter_client
        judges.append(Judge(cfg["slug"], backend, model=cfg.get("model"), client=client))
    return judges
