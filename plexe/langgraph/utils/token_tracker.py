"""
Cumulative token usage tracker for the multi-agent pipeline.

Tracks input/output tokens per agent and enforces optional budgets.
Thread-safe since multiple agents may run callbacks concurrently.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenTracker:
    """Accumulates token usage across all agents in a pipeline run.

    Attributes:
        budget: Optional hard cap on total tokens. ``None`` means unlimited.
    """

    budget: Optional[int] = None
    total_input: int = 0
    total_output: int = 0
    per_agent: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, agent_name: str, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from a single LLM call."""
        with self._lock:
            self.total_input += input_tokens
            self.total_output += output_tokens
            if agent_name not in self.per_agent:
                self.per_agent[agent_name] = {"input": 0, "output": 0}
            self.per_agent[agent_name]["input"] += input_tokens
            self.per_agent[agent_name]["output"] += output_tokens

    @property
    def total(self) -> int:
        return self.total_input + self.total_output

    def is_over_budget(self) -> bool:
        return self.budget is not None and self.total > self.budget

    def remaining_pct(self) -> Optional[float]:
        """Return percentage of budget remaining, or None if no budget."""
        if self.budget is None or self.budget == 0:
            return None
        return max(0.0, 1.0 - self.total / self.budget) * 100

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_input_tokens": self.total_input,
                "total_output_tokens": self.total_output,
                "total": self.total,
                "budget": self.budget,
                "per_agent": {k: dict(v) for k, v in self.per_agent.items()},
            }
