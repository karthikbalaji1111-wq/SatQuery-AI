"""Mechanical validation of a generated answer against deterministic evidence.

    DraftAnswer + AgentEvidence -> AnswerValidation

**This is containment, not proof.** The validator can establish three narrow,
checkable properties and nothing more:

* every number the answer states is traceable to a value in the evidence, at
  the precision the answer itself used;
* the answer cites only evidence ids that exist;
* the answer uses none of the phrases that would mischaracterise the system's
  output.

It explicitly does **not** establish qualitative correctness, causal
attribution, or the semantic truth of prose. An answer saying "the index rose"
states no number, so numeric grounding has nothing to check and it passes. An
answer drawing a wrong conclusion from correctly-quoted figures passes too.
Those limits are real and are not worked around here, because a validator that
implied otherwise would be a worse lie than the one it prevents.

The structural mitigation is elsewhere: the deterministic evidence is always
returned alongside the prose, and a failed check withholds the *answer* while
keeping the *evidence*. The measurements are the product; the sentence is a
presentation of them.

Purity: no provider, no network, no filesystem, no service handle, no clock, no
randomness. The same answer and evidence always yield the same result, and
neither input is mutated.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.services.agent.schemas import AgentEvidence, AnswerValidation
from app.services.query.schemas import SatQueryIntent, TemporalComparison, TimeRange

# --------------------------------------------------------------------------- #
# Forbidden vocabulary
#
# Phase 14 introduced this protection, but its only home was a constant inside
# ``tests/test_temporal_ndwi.py`` - there was no production source of truth to
# import. Defining it here makes it enforceable at runtime rather than only in
# a test, and ``test_agent_grounding`` asserts this list remains a SUPERSET of
# the Phase 14 one, so the two can never drift apart.
#
# Note what is deliberately absent: the bare word "change". Phase 14.1 settled
# that the word is fine and only the mischaracterisation is forbidden.
# --------------------------------------------------------------------------- #

FORBIDDEN_PHRASES: tuple[str, ...] = (
    "per-pixel",
    "per pixel",
    "pixel-level",
    "pixel level",
    "change detection",
    "change mask",
    "changed pixels",
    "detected change",
    "spatial change",
    "land-cover change",
    "land cover change",
    "co-registered",
    "coregistered",
)

#: Absolute slack for float representation only - not a similarity threshold.
#: Numeric agreement is decided by rounding, below; this exists so that
#: ``0.1 + 0.2`` style artefacts do not fail an otherwise exact match.
_FLOAT_EPSILON = 5e-9

#: A number that STARTS a token: not preceded by a letter, digit, underscore or
#: dot. That leading boundary is what stops ``S2B_44PLV_20241026_0_L2A`` from
#: decomposing into invented measurements - every digit run inside it follows a
#: letter or an underscore - and it holds independently of any masking.
#:
#: A trailing letter is deliberately NOT excluded. Requiring one cost more than
#: it bought: "12km2", "500m" and "0.99x" produced no match at all, so the
#: number was never checked and an unsupported claim passed as grounded. A
#: skipped number is an unchecked claim, which is the one failure mode this
#: validator exists to prevent. The unit is not interpreted - "500m" is
#: grounded iff 500 is in the evidence.
#:
#: The remaining guards stop a partial match inside a longer number: a
#: following digit, or a following ``.digit``. A sentence-final period is
#: allowed, so "the mean was 0.85." still yields the claim 0.85.
#:
#: Scientific notation is matched too, for the same reason: without it,
#: "1.5e10" produced no match and went unchecked.
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?!\d)(?!\.\d)"
)

#: An identifier-shaped token: underscore-joined segments carrying at least one
#: letter and one digit - the shape of a STAC scene id. Narrow on purpose:
#: requiring an underscore keeps ordinary prose like "Sentinel-2" and "L2A" out
#: of the check, so only something genuinely claiming to be a scene id is
#: challenged.
_IDENTIFIER = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")

#: A Sentinel platform or product identifier: ``Sentinel-2``, ``Sentinel-1``,
#: ``Sentinel-2A``, ``sentinel-2-optical``, ``sentinel-1-sar``. The digit names
#: a satellite; it measures nothing, so it must not be read as a claim.
#:
#: Deliberately narrow, and narrow in the fail-closed direction. The digit is a
#: single ``[12]`` and everything that may follow is letters, so this pattern
#: can never blank out a multi-digit number: "Sentinel-12345" and "Sentinel-25"
#: match nothing and keep their digits under scrutiny. Substitution is
#: span-local, so a bare number elsewhere in the same sentence is still read.
_PLATFORM_IDENTIFIER = re.compile(
    r"\bsentinel-[12][A-Za-z]?(?:-[A-Za-z]+)*\b", re.IGNORECASE
)

_YEAR = re.compile(r"(?<![A-Za-z0-9_.])(\d{4})(?![A-Za-z0-9_.])")
_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_YEAR = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(\d{4})\b", re.IGNORECASE
)

#: The ``YYYY-MM-DD`` prefix of an ISO timestamp such as
#: ``2024-01-15T05:00:00Z``.
_ISO_DATE_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


class DraftAnswer(BaseModel):
    """A generated answer awaiting validation.

    Produced by the synthesizer (Commit 5), consumed by :func:`validate_answer`
    below. Deliberately defined here rather than in ``schemas.py``: it is the
    input to grounding, not part of the agent's external contract, and Commit 1
    is not modified to accommodate it.

    It carries the prose and the evidence the generator claims to have used -
    and nothing else. There is no field for reasoning, confidence or tool calls,
    and ``extra="forbid"`` means none can be smuggled in.
    """

    model_config = ConfigDict(extra="forbid")

    #: The prose shown to the user. Required and non-empty: an answer that says
    #: nothing would still be reported as a successful one.
    summary: str = Field(min_length=1)
    #: The evidence ids the generator claims to have used. Required with no
    #: default, so citing nothing is an explicit ``[]`` rather than an
    #: omission - a provider that simply forgot the field is a failure, not a
    #: silently uncited answer.
    evidence_refs: list[str]


# --------------------------------------------------------------------------- #
# Allowed values, drawn only from what the system actually established
# --------------------------------------------------------------------------- #


def _allowed_values(evidence: AgentEvidence) -> set[float]:
    """Every numeric value the answer is permitted to state."""

    return {
        item.measurement.value
        for item in evidence.items
        if item.measurement is not None
    }


def _windows(intent: SatQueryIntent) -> list[TimeRange]:
    """Flatten the intent's temporal windows, whichever shape they take."""

    windows = intent.time_windows
    if isinstance(windows, TemporalComparison):
        return [windows.baseline, windows.target]
    return list(windows)


def _acquisition_dates(evidence: AgentEvidence) -> set[str]:
    """``YYYY-MM-DD`` for every acquisition the evidence actually carries.

    A requested window and an acquired scene are different dates - Phase 12's
    whole point. Allowlisting only the request meant a *correct* answer citing
    the date a scene was really acquired was withheld as ungrounded.

    These come from the evidence, never from the answer: only a date the
    system itself established is accepted, so an invented one still fails.
    """

    if evidence.execution is None:
        return set()

    dates: set[str] = set()
    for window in evidence.execution.windows:
        for scene in window.scenes:
            match = _ISO_DATE_PREFIX.match(scene.datetime or "")
            if match is not None:
                dates.add(match.group(0))
    return dates


def _allowed_dates(evidence: AgentEvidence) -> tuple[set[str], set[str], set[str]]:
    """ISO dates, ``month year`` phrases and years the evidence supports.

    Two sources, both established by the system: the *validated intent* that was
    executed (what was asked for) and the *acquisitions* the execution returned
    (what was actually obtained). Both are taken from
    ``evidence.execution`` rather than accepted as arguments, so the allowlist
    cannot describe a query that never ran. With no execution there is no
    intent and no acquisition, so nothing is allowlisted.
    """

    if evidence.execution is None:
        return set(), set(), set()

    iso: set[str] = set()
    month_years: set[str] = set()
    years: set[str] = set()

    def allow(year: int, month: int, iso_date: str) -> None:
        iso.add(iso_date)
        years.add(f"{year:04d}")
        month_years.add(f"{_MONTH_NAMES[month - 1]} {year:04d}")

    for window in _windows(evidence.execution.plan.intent):
        for moment in (window.start_date, window.end_date):
            allow(moment.year, moment.month, moment.isoformat())

    for acquired in _acquisition_dates(evidence):
        year, month, _ = acquired.split("-")
        allow(int(year), int(month), acquired)

    return iso, month_years, years


def _scene_ids(evidence: AgentEvidence) -> set[str]:
    """Scene identifiers the execution actually produced.

    Only these are masked. An identifier-shaped token that survives the mask
    was never produced by the execution, so it is reported as a fabricated
    claim rather than waved through.
    """

    if evidence.execution is None:
        return set()

    ids: set[str] = set()
    for window in evidence.execution.windows:
        if window.selected_scene_id is not None:
            ids.add(window.selected_scene_id)
        ids.update(scene.id for scene in window.scenes)
    return ids


# --------------------------------------------------------------------------- #
# Numeric grounding
# --------------------------------------------------------------------------- #


def _decimals(literal: str) -> int:
    """Decimal places implied by the literal, exponent included.

    ``0.28`` shows 2; ``1.5e10`` shows one mantissa decimal shifted ten places
    left, so it claims precision to the nearest 10^9 and yields -9. Python's
    ``round`` accepts a negative ndigits, so the same comparison works for
    both without a separate code path.
    """

    mantissa, _, exponent = literal.lower().partition("e")
    _, _, fraction = mantissa.partition(".")
    shift = int(exponent) if exponent else 0
    return len(fraction) - shift


def _is_grounded(literal: str, allowed: Iterable[float]) -> bool:
    """Whether ``literal`` is some allowed value shown at its own precision.

    Agreement is decided by rounding rather than by a tolerance band: an answer
    writing ``0.28`` is making a claim about a value that rounds to 0.28 at two
    decimals, so 0.2777 qualifies and 0.27 does not. This adapts to however
    precisely the answer chose to speak, without inventing a similarity
    threshold that would let a materially different number through.
    """

    cleaned = literal.replace(",", "").lstrip("+")
    try:
        stated = float(cleaned)
    except ValueError:  # pragma: no cover - the regex cannot produce this
        return False

    places = _decimals(cleaned)
    return any(
        abs(round(candidate, places) - stated) <= _FLOAT_EPSILON
        for candidate in allowed
    )


def _mask(text: str, literals: Iterable[str]) -> str:
    """Blank out exact substrings so their digits are not read as claims."""

    masked = text
    for literal in sorted(literals, key=len, reverse=True):
        if literal:
            masked = masked.replace(literal, " ")
    return masked


def _ungrounded_claims(summary: str, evidence: AgentEvidence) -> list[str]:
    """Every literal in ``summary`` that the evidence cannot account for.

    Order of operations matters. Real scene identifiers and intent-derived
    dates are removed first, because both are legitimately full of digits that
    are not measurements. What remains is treated as claims: an
    identifier-shaped token the execution never produced is a fabricated scene
    id, and every surviving number must match a value the system computed.
    """

    iso_dates, month_years, years = _allowed_dates(evidence)

    text = _mask(summary, _scene_ids(evidence))

    # Anything still shaped like a scene id was not one the execution returned.
    invented = [
        match.group(0)
        for match in _IDENTIFIER.finditer(text)
        if any(c.isdigit() for c in match.group(0))
        and any(c.isalpha() for c in match.group(0))
    ]
    text = _mask(text, invented)

    # Allowed date forms are removed; unrecognised ones are deliberately left
    # in place so their digits surface as unaccounted numbers.
    text = _mask(text, iso_dates)
    text = _MONTH_YEAR.sub(
        lambda m: " " if m.group(0).lower() in month_years else m.group(0), text
    )
    text = _YEAR.sub(lambda m: " " if m.group(1) in years else m.group(0), text)

    # A platform name is not a measurement. This masks the identifier only, so
    # units, ordinary numbers and scientific notation beside it stay claims.
    text = _PLATFORM_IDENTIFIER.sub(" ", text)

    allowed = _allowed_values(evidence)
    return invented + [
        match.group(0)
        for match in _NUMBER.finditer(text)
        if not _is_grounded(match.group(0), allowed)
    ]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def find_forbidden_phrases(text: str) -> list[str]:
    """Forbidden phrases present in ``text``, case-insensitively."""

    lowered = text.lower()
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in lowered]


def unresolved_references(draft: DraftAnswer, evidence: AgentEvidence) -> list[str]:
    """Cited ids that no evidence item carries.

    Relies on ``AgentEvidence`` guaranteeing unique ids; this does not re-check
    uniqueness, which is the contract's job.
    """

    return sorted(set(draft.evidence_refs) - evidence.ids())


def validate_answer(
    draft: DraftAnswer, evidence: AgentEvidence
) -> AnswerValidation:
    """Run every mechanical check and report each outcome independently.

    The three checks do not short-circuit one another: a forbidden phrase does
    not hide an ungrounded number, and vice versa, so a caller sees everything
    that is wrong rather than only the first thing.

    Pure: no I/O, no clock, no randomness, and neither argument is mutated.
    """

    numeric = "fail" if _ungrounded_claims(draft.summary, evidence) else "pass"
    forbidden = "fail" if find_forbidden_phrases(draft.summary) else "pass"
    references = "fail" if unresolved_references(draft, evidence) else "pass"

    return AnswerValidation(
        numeric_grounding=numeric,
        forbidden_terms=forbidden,
        evidence_refs=references,
    )
