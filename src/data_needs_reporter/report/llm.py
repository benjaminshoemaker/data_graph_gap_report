from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


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
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def json_complete(self, prompt: str, temperature: float) -> Dict[str, Any]:
        cache_path = self._cache_path(prompt, temperature)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

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
        except Exception as exc:  # pragma: no cover - provider errors
            raise LLMError(f"Provider error: {exc}") from exc

        if time.time() - start > self.timeout_s:
            raise LLMError("LLM request exceeded timeout")

        parsed = self._validate_json(result)
        cache_path.write_text(json.dumps(parsed, sort_keys=True), encoding="utf-8")
        return parsed

    def _cache_path(self, prompt: str, temperature: float) -> Path:
        key = json.dumps(
            {"prompt": prompt, "temperature": temperature, "model": self.model},
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return self.cache_dir / f"{digest}.json"

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

    def json_complete(self, prompt: str, temperature: float) -> Dict[str, Any]:
        cache_path = self._cache_path(prompt, temperature)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        attempts = 0
        last_error: Optional[Exception] = None
        while attempts <= self.repair_attempts:
            attempts += 1
            try:
                payload = {
                    "model": self.model,
                    "prompt": (
                        prompt
                        if attempts == 1
                        else f"{prompt}\nRespond with STRICT JSON."
                    ),
                    "temperature": temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "response_format": "json",
                }
                start = time.time()
                result = self.provider.json_complete(payload)
                if time.time() - start > self.timeout_s:
                    raise LLMError("LLM request exceeded timeout")
                parsed = self._validate_json(result)
                cache_path.write_text(
                    json.dumps(parsed, sort_keys=True), encoding="utf-8"
                )
                return parsed
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
