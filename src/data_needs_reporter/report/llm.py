from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from data_needs_reporter.utils.cost_guard import BudgetExceededError, CostGuard


class LLMError(RuntimeError):
    """Raised when the LLM client cannot produce a valid response."""


class LLMProvider:
    """Abstract provider interface."""

    def json_complete(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class MockProvider(LLMProvider):
    response: Dict[str, Any]
    fail_on_first: bool = False
    _called: int = 0

    def json_complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._called += 1
        if self.fail_on_first and self._called == 1:
            raise ValueError("Mock parse failure")
        return self.response


class LLMClient:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        api_key_env: str,
        timeout_s: float,
        max_output_tokens: int,
        cache_dir: Path | str = Path(".cache/llm"),
        cache_enabled: bool = True,
    ) -> None:
        if not api_key_env:
            raise ValueError("api_key_env must be provided")
        if api_key_env not in os.environ:
            raise LLMError(f"Missing API key in environment variable {api_key_env}")

        self.provider = provider
        self.model = model
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def json_complete(
        self,
        prompt: str,
        temperature: float,
        *,
        guard: Optional[CostGuard] = None,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        cache_path = (
            self._cache_path(prompt, temperature) if self.cache_enabled else None
        )
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        parsed = self._invoke_provider(prompt, temperature, guard, channel)
        self._write_cache(cache_path, parsed)
        return parsed

    def _invoke_provider(
        self,
        prompt: str,
        temperature: float,
        guard: Optional[CostGuard],
        channel: Optional[str],
    ) -> Dict[str, Any]:
        self._consume_budget(prompt, guard, channel)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_format": "json",
        }
        start = time.time()
        try:
            result = self.provider.json_complete(payload)
        except BudgetExceededError as exc:  # pragma: no cover - guard errors
            raise LLMError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - provider errors
            raise LLMError(f"Provider error: {exc}") from exc

        if time.time() - start > self.timeout_s:
            raise LLMError("LLM request exceeded timeout")

        return self._validate_json(result)

    def _consume_budget(
        self, prompt: str, guard: Optional[CostGuard], channel: Optional[str]
    ) -> None:
        if guard is None:
            return
        channel_name = channel or "default"
        tokens = self._estimate_total_tokens(prompt)
        try:
            guard.consume(tokens, channel=channel_name)
        except BudgetExceededError as exc:
            raise LLMError(f"LLM budget exceeded for channel '{channel_name}'") from exc

    def _estimate_total_tokens(self, prompt: str) -> int:
        if not prompt:
            prompt_tokens = 1
        else:
            char_estimate = max(len(prompt) // 4, 1)
            word_estimate = max(int(len(prompt.split()) * 0.75), 1)
            prompt_tokens = max(char_estimate, word_estimate)
        output_cap = max(self.max_output_tokens, 0)
        return prompt_tokens + output_cap

    def _cache_path(self, prompt: str, temperature: float) -> Path:
        raw = f"{self.model}|{temperature}|{prompt}".encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, cache_path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if cache_path is None:
            return None
        if not cache_path.exists():
            return None
        return json.loads(cache_path.read_text(encoding="utf-8"))

    def _write_cache(self, cache_path: Optional[Path], payload: Dict[str, Any]) -> None:
        if cache_path is None:
            return
        cache_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _validate_json(self, result: Any) -> Dict[str, Any]:
        if not isinstance(result, dict):
            raise LLMError("LLM response must be a JSON object")
        content = result.get("content")
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise LLMError(f"Response content not valid JSON: {exc}") from exc
        raise LLMError("LLM response missing JSON content")


class RepairingLLMClient(LLMClient):
    def __init__(self, *args: Any, repair_attempts: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.repair_attempts = repair_attempts

    def json_complete(
        self,
        prompt: str,
        temperature: float,
        *,
        guard: Optional[CostGuard] = None,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        cache_path = (
            self._cache_path(prompt, temperature) if self.cache_enabled else None
        )
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        attempts = 0
        last_error: Optional[Exception] = None
        while attempts <= self.repair_attempts:
            attempts += 1
            try:
                prompt_variant = (
                    prompt if attempts == 1 else f"{prompt}\nRespond with STRICT JSON."
                )
                parsed = self._invoke_provider(
                    prompt_variant, temperature, guard, channel
                )
                self._write_cache(cache_path, parsed)
                return parsed
            except LLMError as exc:
                last_error = exc
                if guard is not None and guard.stopped_due_to_cap:
                    raise
                continue
            except Exception as exc:  # pragma: no cover
                last_error = exc
                continue
        raise LLMError(f"LLM failed after repairs: {last_error}") from last_error


__all__ = [
    "LLMClient",
    "RepairingLLMClient",
    "LLMError",
    "MockProvider",
]
