"""Deterministic completion of a plan for explicitly named spectral indices.

    question + validated AgentPlan -> AgentPlan (the same object when unchanged)

**Why this exists.** A planner may return a structurally valid plan that omits
the one analysis the user asked for by name. Observed live with both providers:
for "What is the NDWI of <a place> in <a month>?" a planner returned
``execute_query`` ALONE. Discovery then ran, nothing was measured, and
the synthesizer - correctly, given what it was handed - answered "Insufficient
evidence". The pipeline behaved honestly and the product still looked broken,
because the analysis the user named outright was never run.

**What it does.** When the question names NDVI, NDWI or NDBI by its acronym and
the plan is a single-window Sentinel-2 plan that does not already compute that
index, the index is added - by extending the plan's ``spectral_indices`` step,
or by appending one. Nothing else is ever changed.

**Why this is not fabrication.** It adds a request to COMPUTE, never a result.
The index is still measured from real pixels by the same deterministic engine,
under the same validation; a missing band still fails honestly. It is the same
kind of server-owned invariant as ``AgentPlan`` enabling ``include_imagery`` for
an analysis plan: the user's intent was explicit, only the plan was incomplete.

**Deliberately narrow.** Only the three acronyms trigger it - "water" or
"vegetation" remain the planner's judgement, because mapping prose to an index
is exactly the decision a planner exists to make. Only a ``single`` window with
``sentinel-2-optical`` qualifies: a comparison has its own tool, and an index
cannot be computed from SAR. And ``AgentPlan`` stays the authority - the result
is re-validated, and anything it would refuse (the step budget, a repeated tool)
returns the planner's plan untouched rather than dropping a step it chose.

Transparency: the caller keeps the planner's ORIGINAL plan in ``trace.plan`` and
executes this one, so ``trace.steps`` shows the added step as having run. The
trace already allows the requested plan and the executed steps to differ.

Purity: no provider, no network, no clock. Neither input is mutated.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from pydantic import ValidationError

from app.services.agent.schemas import (
    AgentPlan,
    ExecuteQueryParams,
    NdwiParams,
    RsModelParams,
    SpectralIndicesParams,
    TemporalNdwiParams,
)

#: The acronyms, as whole words. "NDWI" and "ndwi" count; "ndwis" or an
#: identifier containing the letters does not.
_NAMED_INDEX = re.compile(r"(?<![A-Za-z0-9_])(ndvi|ndwi|ndbi)(?![A-Za-z0-9_])", re.IGNORECASE)


def named_indices(question: str) -> list[str]:
    """The indices the question names by acronym, in order of first mention.

    Recognition only - it reports what the text CONTAINS, which is not the same
    as what it asks for. :func:`requested_indices` is what completion consults.
    """

    found: list[str] = []
    for match in _NAMED_INDEX.finditer(question):
        key = match.group(1).lower()
        if key not in found:
            found.append(key)
    return found


# --------------------------------------------------------------------------- #
# Reading the request, rather than scanning it
#
# Substring presence is not intent. "Show imagery only. Do not calculate NDVI."
# names NDVI in order to REFUSE it, and completing on the name alone is not a
# harmless extra: it spends two more band reads, puts an unrequested measurement
# in the evidence, and lets a synthesized answer discuss an index the user ruled
# out. The same holds for a definition ("NDVI is a vegetation index"), a
# quotation, and a supposition ("if I wanted NDVI ...").
#
# So a mention is classified before it is honoured. The rules are deliberately
# shallow - clause polarity plus a few framing patterns, no parser and no model
# - and they are biased in ONE direction: unless a mention is clearly a request,
# nothing is added. That asymmetry is the whole design. A missed completion
# leaves the planner's own judgement in place, which is the status quo ante; an
# invented one executes work the user explicitly refused.
# --------------------------------------------------------------------------- #

#: Sentence terminators. A refusal never reaches across one.
_SENTENCE_END = re.compile(r"[.!?;\n]+")

#: Within a sentence, a comma or a contrastive conjunction opens a new clause.
_CLAUSE_BOUNDARY = re.compile(
    r",|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b|\s[-–—]\s",
    re.IGNORECASE,
)

#: A contrastive conjunction CANCELS the preceding refusal ("do not bother with
#: the imagery, but do compute NDWI"). A bare comma does not.
_CONTRAST = re.compile(r"(?:but|however|although|though|yet)", re.IGNORECASE)

#: A clause opening with an imperative is a fresh request, so a refusal in the
#: clause before it does not govern it ("there is no NDVI here, compute NDWI").
_AFFIRMATIVE_OPENING = re.compile(
    r"^\s*(?:and|then|also|please)?\s*(?:do\s+)?"
    r"(?:compute|calculate|show|display|include|add|give|measure|report|"
    r"run|produce|return|provide|plot|map)\b",
    re.IGNORECASE,
)

#: Refusal cues. Each scopes FORWARD to the end of its clause, which is why
#: "Show NDWI, not NDVI" keeps the first and drops the second.
_NEGATION = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"do\s+not|don['’]?t|does\s+not|doesn['’]?t|did\s+not|"
    r"didn['’]?t|cannot|can['’]?t|won['’]?t|"
    r"not|no|never|without|"
    r"skip|exclude|excluding|omit|omitting|avoid|avoiding|ignore|ignoring|"
    r"rather\s+than|instead\s+of|leave\s+out"
    r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

#: A supposition frames what follows as not-yet-asked-for. Counted only BEFORE
#: the mention: "compute NDWI if the scene is clear" is a request with a
#: condition attached, not a hypothetical.
_HYPOTHETICAL = re.compile(
    r"\b(?:if|suppose|supposing|imagine|hypothetically)\b", re.IGNORECASE
)

#: "What is NDWI?" asks for a definition. "What is the NDWI of Chennai?" and
#: "What is the NDWI, and is water visible?" both ask for a measurement.
#:
#: The determiner separates them. A definition asks about the TERM, so the
#: phrase runs straight into the acronym; anything in between - "the NDWI" -
#: makes it a property of something, which is a measurement. An earlier rule
#: here looked for an object AFTER the acronym instead, and refused to compute
#: "What is the NDWI, and is water visible in the image?" - a real request,
#: caught by the existing suite.
_DEFINITION_LEAD = re.compile(
    r"(?:what\s+(?:is|are)|define[sd]?|definition\s+of|explain|meaning\s+of)\s*$",
    re.IGNORECASE,
)
_IS_DEFINED = re.compile(
    r"^\s*(?:is|are|stands\s+for|means|refers\s+to|measures)\b", re.IGNORECASE
)
#: "How leafy is Chennai?" asks HOW MUCH, not what "leafy" means: a lowercase
#: word straight after "how" is a degree, so "<word> is" does not define it.
#: An acronym ("How NDVI is computed") keeps the definition reading.
_DEGREE_QUESTION = re.compile(r"(?<![A-Za-z0-9_])how\s+$", re.IGNORECASE)

#: Quoted text is being discussed, not issued as an instruction.
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'|“[^”]*”")


def _clauses(sentence: str) -> Iterator[tuple[int, str, bool]]:
    """Each clause as ``(offset in the sentence, text, cancels_refusal)``."""

    position = 0
    cancels = False
    for boundary in _CLAUSE_BOUNDARY.finditer(sentence):
        yield position, sentence[position : boundary.start()], cancels
        cancels = _CONTRAST.fullmatch(boundary.group().strip()) is not None
        position = boundary.end()
    yield position, sentence[position:], cancels


def requested_matches(
    question: str, pattern: re.Pattern[str]
) -> list[re.Match[str]]:
    """Matches of ``pattern`` that the question actually ASKS FOR.

    A match is dropped when it is refused, quoted, supposed, or defined rather
    than requested. Results are in document order, so a caller can report them
    by first mention.
    """

    quoted = [(m.start(), m.end()) for m in _QUOTED.finditer(question)]
    kept: list[re.Match[str]] = []
    start = 0

    for terminator in [*_SENTENCE_END.finditer(question), None]:
        end = terminator.start() if terminator is not None else len(question)
        sentence = question[start:end]
        supposed = _HYPOTHETICAL.search(sentence)
        refused = False

        for offset, clause, cancels in _clauses(sentence):
            if cancels or _AFFIRMATIVE_OPENING.match(clause):
                refused = False
            cue = _NEGATION.search(clause)

            for match in pattern.finditer(clause):
                at = offset + match.start()
                if any(lo <= start + at < hi for lo, hi in quoted):
                    continue
                if supposed is not None and supposed.start() < at:
                    continue
                if refused or (cue is not None and match.start() >= cue.start()):
                    continue
                tail = sentence[offset + match.end() :]
                if _IS_DEFINED.match(tail) and not (
                    match.group(0).islower()
                    and _DEGREE_QUESTION.search(sentence[: offset + match.start()])
                ):
                    continue
                if _DEFINITION_LEAD.search(sentence[:at]):
                    continue
                kept.append(match)

            if cue is not None:
                refused = True

        start = end + (len(terminator.group()) if terminator is not None else 0)

    return kept


def requested_indices(question: str) -> list[str]:
    """The indices the question asks to COMPUTE, in order of first request.

    Differs from :func:`named_indices` exactly where it matters: an acronym that
    is refused, quoted, supposed or defined is named but not requested.
    """

    found: list[str] = []
    for match in requested_matches(question, _NAMED_INDEX):
        key = match.group(1).lower()
        if key not in found:
            found.append(key)
    return found


def ensure_requested_indices(question: str, plan: AgentPlan) -> AgentPlan:
    """Return a plan that computes every index the question names by acronym.

    Returns ``plan`` itself - the same object - whenever nothing needs adding or
    the addition would not validate, so ``result is plan`` tells a caller
    whether anything changed.
    """

    named = requested_indices(question)
    if not named:
        return plan

    discovery = plan.steps[0]
    if not isinstance(discovery, ExecuteQueryParams):
        return plan
    intent = discovery.intent
    if intent.temporal_mode != "single" or "sentinel-2-optical" not in intent.modalities:
        return plan

    covered: set[str] = set()
    spectral_at: int | None = None
    for position, step in enumerate(plan.steps):
        if isinstance(step, SpectralIndicesParams):
            covered.update(step.indices)
            spectral_at = position
        elif isinstance(step, NdwiParams):
            covered.add("ndwi")

    missing = [key for key in named if key not in covered]
    if not missing:
        return plan

    steps = list(plan.steps)
    if spectral_at is not None:
        existing = steps[spectral_at]
        assert isinstance(existing, SpectralIndicesParams)
        steps[spectral_at] = SpectralIndicesParams(indices=[*existing.indices, *missing])
    else:
        steps.append(SpectralIndicesParams(indices=missing))

    try:
        return AgentPlan(steps=steps)
    except ValidationError:
        # The step budget or another structural rule would be broken. The
        # planner's own plan runs as it was; a step it chose is never dropped
        # to make room.
        return plan


# --------------------------------------------------------------------------- #
# Visual observation
#
# Observed live: "Is there visible water in the Sentinel-2 image of Marina
# Beach, Chennai in January 2025?" - the workspace's own first example - was
# planned as execute_query ALONE, so no image reached the vision model and the
# run ended "Insufficient evidence" beside a perfectly good scene.
#
# The same rule as the indices, applied to looking: a question that asks what is
# VISIBLE gets the observation step, carrying the user's own question verbatim.
# The model then looks at the image the server retrieved and its reply is
# recorded as an attributed observation - never a measurement, and never able to
# authorise a number. Deliberately narrow: "image" or "imagery" alone does not
# trigger it, because merely viewing imagery needs discovery only.
# --------------------------------------------------------------------------- #

_VISUAL_REQUEST = re.compile(
    r"\b(?:visible|visibly|visually|looks?\s+like|can\s+(?:you\s+)?see|can\s+be\s+seen)\b",
    re.IGNORECASE,
)

#: ``RsModelParams.question`` is capped; a longer question is not truncated,
#: because a cut question could ask the model something the user did not.
_VISUAL_QUESTION_LIMIT = 500


def asks_what_is_visible(question: str) -> bool:
    """Whether the question mentions seeing at all. Recognition only."""

    return _VISUAL_REQUEST.search(question) is not None


def requests_visual_observation(question: str) -> bool:
    """Whether it ASKS to be told what is visible.

    "Do not describe what is visible" mentions it and refuses it; the same
    polarity rules that govern the indices govern the observation.
    """

    return bool(requested_matches(question, _VISUAL_REQUEST))


def ensure_requested_observation(question: str, plan: AgentPlan) -> AgentPlan:
    """Add the visual observation a question explicitly asks for.

    Same contract as :func:`ensure_requested_indices`: ``plan`` itself is
    returned whenever nothing is added or the addition would not validate.
    """

    text = question.strip()
    if not requests_visual_observation(text) or len(text) > _VISUAL_QUESTION_LIMIT:
        return plan
    if any(isinstance(step, RsModelParams) for step in plan.steps):
        return plan

    discovery = plan.steps[0]
    if not isinstance(discovery, ExecuteQueryParams):
        return plan
    intent = discovery.intent
    # The observation is of ONE Sentinel-2 true-colour image; the planner
    # instruction already rules it out for SAR.
    if intent.temporal_mode != "single" or intent.modalities != ["sentinel-2-optical"]:
        return plan

    try:
        return AgentPlan(steps=[*plan.steps, RsModelParams(question=text)])
    except ValidationError:
        return plan


# --------------------------------------------------------------------------- #
# Temporal NDWI
#
# Observed live: "How has water changed at <a place> between January 2024 and
# January 2025?" was planned correctly as a two-window COMPARISON over
# Sentinel-2 - and then with no analysis step, so two scenes were found,
# nothing was measured, and the run ended "Insufficient evidence".
#
# The planner had already committed to the comparison; only the one tool that
# interprets a comparison was missing. This adds it when the question is about
# water (or names NDWI). A comparison about vegetation or anything else is left
# alone: temporal NDWI would answer a different question from the one asked.
# --------------------------------------------------------------------------- #

_WATER_REQUEST = re.compile(r"(?<![A-Za-z0-9_])(?:water|ndwi)(?![A-Za-z0-9_])", re.IGNORECASE)


def ensure_requested_comparison(question: str, plan: AgentPlan) -> AgentPlan:
    """Add temporal NDWI to an optical comparison that asks about water."""

    if not requested_matches(question, _WATER_REQUEST):
        return plan
    if any(isinstance(step, TemporalNdwiParams) for step in plan.steps):
        return plan

    discovery = plan.steps[0]
    if not isinstance(discovery, ExecuteQueryParams):
        return plan
    intent = discovery.intent
    if intent.temporal_mode != "compare" or "sentinel-2-optical" not in intent.modalities:
        return plan

    try:
        return AgentPlan(steps=[*plan.steps, TemporalNdwiParams()])
    except ValidationError:
        return plan


def complete_plan(question: str, plan: AgentPlan) -> AgentPlan:
    """Apply every completion. ``result is plan`` when nothing changed."""

    completed = ensure_requested_indices(question, plan)
    completed = ensure_requested_comparison(question, completed)
    return ensure_requested_observation(question, completed)
