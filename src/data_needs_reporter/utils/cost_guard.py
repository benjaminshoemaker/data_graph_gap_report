from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from data_needs_reporter.utils.io import write_json_atomic


class BudgetExceededError(RuntimeError):
    """Raised when attempting to consume tokens beyond the configured cap."""


@dataclass
class CostGuard:
    cap_usd: float
    price_per_1k_tokens: float
    safety_margin: float = 0.0
    tokens_used: int = 0
    stopped_due_to_cap: bool = False
    _channel_usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cap_usd <= 0:
            raise ValueError("cap_usd must be positive")
        if self.price_per_1k_tokens <= 0:
            raise ValueError("price_per_1k_tokens must be positive")
        self._price_per_token = self.price_per_1k_tokens / 1000.0
        margin = min(max(self.safety_margin, 0.0), 0.95)
        margin_cap = self.cap_usd * (1 - margin)
        # Reserve at least the cost of one token to avoid zero budget
        reserve = max(self._price_per_token, 1e-9)
        self._cap_with_margin = max(margin_cap - reserve, 0.0)

    def can_consume(self, tokens: int) -> bool:
        if tokens <= 0:
            return True
        required = (self.tokens_used + tokens) * self._price_per_token
        return required <= self._cap_with_margin + 1e-12

    def try_consume(self, tokens: int, channel: Optional[str] = None) -> bool:
        if not self.can_consume(tokens):
            self.stopped_due_to_cap = True
            return False
        self.tokens_used += tokens
        if channel:
            usage = self._channel_usage.setdefault(
                channel, {"tokens": 0, "messages": 0}
            )
            usage["tokens"] += tokens
            usage["messages"] += 1
        return True

    def record_message(self, channel: str, tokens: int) -> bool:
        return self.try_consume(tokens, channel)

    @property
    def cost_usd(self) -> float:
        return round(self.tokens_used * self._price_per_token, 4)

    @property
    def token_budget(self) -> int:
        if self._price_per_token == 0:
            return 0
        return max(int(self._cap_with_margin / self._price_per_token), 0)

    def budget_snapshot(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "cap_usd": round(self.cap_usd, 4),
            "price_per_1k_tokens": round(self.price_per_1k_tokens, 6),
            "tokens_used": int(self.tokens_used),
            "cost_usd": self.cost_usd,
            "stopped_due_to_cap": self.stopped_due_to_cap,
            "channels": self._channel_usage,
            "token_budget": self.token_budget,
        }
        if extra:
            data.update(extra)
        return data

    def write_budget(self, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        write_json_atomic(path, self.budget_snapshot(extra))
