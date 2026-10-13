"""Thin client for a local Ollama server. No paid API involved. Falls back to a
second local model if the primary one isn't pulled, and raises a clear error
(rather than silently mocking a response) if Ollama is unreachable entirely."""
from __future__ import annotations

import httpx

from app.config.settings import settings


class GenerationError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.llm_model

    def _available_models(self) -> set[str]:
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            return {m["name"] for m in resp.json().get("models", [])}
        except httpx.HTTPError as exc:
            raise GenerationError(f"Cannot reach Ollama at {self.host}: {exc}") from exc

    def _resolve_model(self) -> str:
        available = self._available_models()
        if self.model in available:
            return self.model
        if settings.llm_fallback_model in available:
            return settings.llm_fallback_model
        raise GenerationError(
            f"Neither '{self.model}' nor fallback '{settings.llm_fallback_model}' are "
            f"pulled in Ollama. Available: {sorted(available)}. Run: ollama pull {self.model}"
        )

    def generate(self, prompt: str, temperature: float = 0.2) -> dict:
        model = self._resolve_model()
        try:
            resp = httpx.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
                timeout=settings.generation_timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise GenerationError(f"Ollama generation failed: {exc}") from exc

        data = resp.json()
        return {
            "text": data.get("response", ""),
            "model": model,
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }


class GroqClient:
    """Hosted alternative to Ollama for deployments without a local LLM server.
    Talks to Groq's OpenAI-compatible chat completions endpoint. Selected with
    LLM_PROVIDER=groq; needs GROQ_API_KEY."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, model: str | None = None):
        import os

        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def _available_models(self) -> set[str]:
        if not self.api_key:
            raise GenerationError("GROQ_API_KEY is not set")
        return {self.model}

    def generate(self, prompt: str, temperature: float = 0.2) -> dict:
        self._available_models()
        try:
            resp = httpx.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=settings.generation_timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise GenerationError(f"Groq generation failed: {exc}") from exc

        data = resp.json()
        usage = data.get("usage", {})
        return {
            "text": data["choices"][0]["message"]["content"],
            "model": self.model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }


def get_llm_client():
    """Ollama by default (local, no key); Groq when LLM_PROVIDER=groq."""
    import os

    if os.getenv("LLM_PROVIDER", "ollama").lower() == "groq":
        return GroqClient()
    return OllamaClient()
