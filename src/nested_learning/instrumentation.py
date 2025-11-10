from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UpdateEvent:
    step: int
    level: str
    magnitude: float | None = None


@dataclass
class UpdateLog:
    """Lightweight container for tracking update magnitudes per level."""

    events: List[UpdateEvent] = field(default_factory=list)

    def record(self, *, step: int, level: str, magnitude: float | None = None) -> None:
        self.events.append(UpdateEvent(step=step, level=level, magnitude=magnitude))

    def summary(self) -> Dict[str, Dict[str, float]]:
        counts: Dict[str, int] = {}
        totals: Dict[str, float] = {}
        for event in self.events:
            counts[event.level] = counts.get(event.level, 0) + 1
            if event.magnitude is not None:
                totals[event.level] = totals.get(event.level, 0.0) + event.magnitude
        return {
            level: {
                "updates": counts[level],
                "avg_magnitude": (
                    totals[level] / counts[level] if level in totals else float("nan")
                ),
            }
            for level in counts
        }
