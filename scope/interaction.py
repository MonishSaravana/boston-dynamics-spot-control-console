"""Future transport boundary only. No reasoning, audio, or action dispatch."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TextCommand:
    text: str
    source: str = "typed"
    request_id: str | None = None


@dataclass(frozen=True)
class TextResponse:
    text: str
    request_id: str | None = None
