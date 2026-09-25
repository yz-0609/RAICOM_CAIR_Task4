"""Diagnostic angle unwrapping; these readings do not establish field position."""

from __future__ import annotations


class AngleAccumulator:
    def __init__(self, initial: float) -> None:
        self.previous = float(initial)
        self.total = 0.0

    def add(self, current: float) -> float:
        current = float(current)
        delta = (current - self.previous + 180) % 360 - 180
        self.total += delta
        self.previous = current
        return self.total
