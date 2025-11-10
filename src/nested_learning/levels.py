from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, MutableMapping, Sequence


@dataclass(frozen=True)
class LevelSpec:
    """Configuration for a nested-learning level."""

    name: str
    update_period: int
    warmup_steps: int = 0
    jitter: int = 0
    optimizer_key: str | None = None

    def __post_init__(self) -> None:
        if self.update_period <= 0:
            msg = f"update_period for level {self.name} must be positive"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = f"warmup_steps for level {self.name} must be non-negative"
            raise ValueError(msg)
        if self.jitter < 0:
            msg = f"jitter for level {self.name} must be non-negative"
            raise ValueError(msg)


@dataclass
class LevelState:
    last_step: int = -1
    updates: int = 0


class LevelClock:
    """Deterministic scheduler for Nested Learning level updates."""

    def __init__(self, specs: Sequence[LevelSpec]):
        if not specs:
            raise ValueError("LevelClock requires at least one LevelSpec")
        self._specs: Dict[str, LevelSpec] = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate level names provided to LevelClock")
        self._state: MutableMapping[str, LevelState] = {
            name: LevelState() for name in self._specs
        }
        self._step: int = 0
        self._timeline: List[dict] = []

    @property
    def step(self) -> int:
        return self._step

    def tick(self) -> None:
        self._step += 1

    def should_update(self, name: str) -> bool:
        spec = self._specs[name]
        state = self._state[name]
        if self._step < spec.warmup_steps:
            return False
        delta = self._step - state.last_step
        period = spec.update_period
        if spec.jitter:
            period = period + (self._step % (spec.jitter + 1))
        return state.last_step < 0 or delta >= period

    def record_update(self, name: str) -> None:
        state = self._state[name]
        state.last_step = self._step
        state.updates += 1
        self._timeline.append({"step": self._step, "level": name})

    def levels_in_frequency_order(self) -> List[LevelSpec]:
        return sorted(self._specs.values(), key=lambda spec: spec.update_period)

    def stats(self) -> Dict[str, LevelState]:
        return {name: LevelState(state.last_step, state.updates) for name, state in self._state.items()}

    def timeline(self) -> List[dict]:
        return list(self._timeline)


def ensure_level_specs(entries: Iterable[LevelSpec]) -> List[LevelSpec]:
    """Ensure deterministic ordering and validate duplicates."""

    specs = list(entries)
    seen = set()
    ordered: List[LevelSpec] = []
    for spec in specs:
        if spec.name in seen:
            msg = f"Duplicate level spec {spec.name}"
            raise ValueError(msg)
        seen.add(spec.name)
        ordered.append(spec)
    return ordered
