"""Spoken or typed console commands, parsed by fixed rules, not a language model.

"go behind the couch" becomes a navigation request for the object phrase
"couch" with the relation BEHIND. A bare phrase ("red chair") is a search with
the existing front approach. Nothing here moves the robot; the parsed command
feeds the same search, confirm, preview and GO steps as the buttons.
"""
from dataclasses import dataclass
from enum import StrEnum
import re


class Relation(StrEnum):
    FRONT = "front"
    BEHIND = "behind"
    LEFT = "left"
    RIGHT = "right"
    NEAR = "near"


class Action(StrEnum):
    FIND = "find"
    GO = "go"
    CONFIRM = "confirm"
    REJECT = "reject"
    STOP = "stop"


@dataclass(frozen=True)
class ParsedCommand:
    action: Action
    text: str
    target: str | None = None
    relation: Relation = Relation.FRONT
    pointing: bool = False


_WAKE = r"(?:(?:hey|ok|okay)\s+)?(?:spot|scope)[,\s]+"
_POLITE = r"(?:(?:please|can you|could you|would you)\s+)*"
_MOVE = r"(?:go|walk|move|head|navigate|get|come|drive|stand)"
_RELATIONS = (
    (Relation.BEHIND, r"(?:to\s+)?(?:behind|in\s+back\s+of|to\s+the\s+back\s+of|around\s+the\s+back\s+of|around\s+the\s+other\s+side\s+of|to\s+the\s+other\s+side\s+of)"),
    (Relation.LEFT, r"(?:to\s+)?(?:the\s+)?left\s+(?:hand\s+)?(?:side\s+)?(?:of|on)"),
    (Relation.RIGHT, r"(?:to\s+)?(?:the\s+)?right\s+(?:hand\s+)?(?:side\s+)?(?:of|on)"),
    (Relation.NEAR, r"(?:to\s+)?(?:next\s+to|near|beside|by|close\s+to|alongside)"),
    # The generic "to" stays last so "to the left of" is not read as FRONT.
    (Relation.FRONT, r"(?:to\s+)?(?:in\s+front\s+of|to\s+the\s+front\s+of|up\s+to|over\s+to|to(?:wards?)?)"),
)
_DETERMINER = r"(?:the|a|an|that|this|my|our|your)\s+"
_POINTING = {"that", "this", "it", "there", "that one", "this one", "that thing",
             "this thing", "where i'm pointing", "where i am pointing"}
_STOP = {"stop", "halt", "freeze", "stop moving", "stop now", "wait", "abort", "cancel"}
_GO = {"go", "go ahead", "go now", "proceed", "continue", "execute", "move now"}
_CONFIRM = {"yes", "yeah", "yep", "confirm", "correct", "that's it", "that one is right",
            "yes that one", "confirm target"}
_REJECT = {"no", "nope", "not that one", "wrong one", "reject", "next", "next one"}
_FIND = r"(?:find|look\s+for|search\s+for|where\s+is|where's|show\s+me|locate|target)\s+"


def _clean(text):
    text = " ".join(str(text).strip().casefold().split())
    text = re.sub(r"[.!?,;:]+$", "", text).strip()
    return re.sub(r"^" + _WAKE, "", text).strip()


def _target(phrase):
    phrase = re.sub(r"^" + _DETERMINER, "", phrase.strip()).strip()
    phrase = re.sub(r"\s+(?:please|now|thanks|thank you)$", "", phrase).strip()
    return phrase


def parse_command(text):
    raw = str(text)
    clean = _clean(raw)
    if not clean or len(clean) > 180:
        raise ValueError("Say or type a command of 1–180 characters")
    bare = re.sub(r"^" + _POLITE, "", clean).strip()
    if bare in _STOP or re.fullmatch(r"(?:stop|halt|freeze)\b.*", bare):
        return ParsedCommand(Action.STOP, raw)
    if bare in _GO:
        return ParsedCommand(Action.GO, raw)
    if bare in _CONFIRM:
        return ParsedCommand(Action.CONFIRM, raw)
    if bare in _REJECT:
        return ParsedCommand(Action.REJECT, raw)
    found = re.fullmatch(_FIND + r"(.+)", bare)
    if found:
        return _with_target(Action.FIND, raw, found.group(1), Relation.FRONT)
    moving = re.fullmatch(_MOVE + r"\s+(.+)", bare)
    if moving:
        rest = moving.group(1)
        for relation, pattern in _RELATIONS:
            hit = re.fullmatch(pattern + r"\s+(.+)", rest)
            if hit:
                return _with_target(Action.FIND, raw, hit.group(1), relation)
        return _with_target(Action.FIND, raw, rest, Relation.FRONT)
    return _with_target(Action.FIND, raw, bare, Relation.FRONT)


def _with_target(action, raw, phrase, relation):
    if phrase.strip() in _POINTING:
        return ParsedCommand(action, raw, None, relation, pointing=True)
    target = _target(phrase)
    if target in _POINTING:
        return ParsedCommand(action, raw, None, relation, pointing=True)
    if not target:
        raise ValueError("Name an object, for example: go behind the couch")
    return ParsedCommand(action, raw, target, relation)
