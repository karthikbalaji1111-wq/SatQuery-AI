"""Mechanical validation of a generated answer against deterministic evidence.

    DraftAnswer + AgentEvidence -> AnswerValidation

**This is containment, not proof.** The validator can establish three narrow,
checkable properties and nothing more:

* every number is supported by a cited measurement with the same identity,
  unit and available observation context, at the precision stated - a
  sentence repeating a cited engine caveat verbatim is that engine's own
  statement, not a claim;
* factual prose is a supported measurement statement or repeats cited evidence;
* the answer uses none of the phrases that would mischaracterise the system's
  output.

It does not prove the truth of a model observation. Qualitative evidence may
be repeated, not freely extrapolated or paraphrased: unmatched prose fails
closed. Narrow, whole-sentence abstentions need no citation. This deliberately
trades recall for containment without introducing a semantic model judge.

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

from app.services.agent.schemas import AgentEvidence, AnswerValidation, EvidenceItem
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
#: validator exists to prevent. Unit agreement is checked separately below.
#:
#: The remaining guards stop a partial match inside a longer number: a
#: following digit, or a following ``.digit``. A sentence-final period is
#: allowed, so "the mean was 0.85." still yields the claim 0.85.
#:
#: Scientific notation is matched too, for the same reason: without it,
#: "1.5e10" produced no match and went unchecked.
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])[-+−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
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
_PLATFORM_IDENTIFIER = re.compile(r"\bsentinel-[12][A-Za-z]?(?:-[A-Za-z]+)*\b", re.IGNORECASE)

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
_MONTH_YEAR = re.compile(r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(\d{4})\b", re.IGNORECASE)

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


#: Evidence sources that may authorise a number, i.e. those produced by a
#: deterministic engine over real pixels or real metadata. Declared as an
#: allowlist rather than a denylist so a NEW source is untrusted by default: a
#: future producer has to be named here to be believed.
_NUMERIC_AUTHORITIES: frozenset[str] = frozenset(
    # Every deterministic index computes its own numbers from pixels, so each
    # is authoritative for the values it reports. Adding an index WITHOUT
    # listing it here would withhold answers quoting perfectly real
    # measurements, so the two must move together.
    {
        "execution",
        "ndvi",
        "ndwi",
        "ndbi",
        "temporal_ndwi",
        "sar_backscatter",
        "compatibility",
    }
)


def _allowed_values(evidence: AgentEvidence) -> set[float]:
    """Raw deterministic values for inspection, NOT claim authorization.

    Only DETERMINISTIC evidence counts. A model-sourced item contributes
    nothing, whatever it carries: a vision-language model that says "about 42
    percent" must not thereby make 42 citable, or numeric grounding becomes
    circular and passes exactly the invented figures it exists to catch.

    The filter is on the SOURCE, not on the absence of a measurement. Visual
    evidence carries no measurement today, so keying on that would make the
    guarantee an accident of the current shape rather than a rule.
    """

    return {
        item.measurement.value
        for item in evidence.items
        if item.measurement is not None and item.source in _NUMERIC_AUTHORITIES
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

    dates: set[str] = set()
    sar = evidence.analysis.sar_backscatter if evidence.analysis else None
    if sar is not None and sar.acquired_at is not None:
        dates.add(sar.acquired_at.isoformat()[:10])
    comparison = evidence.analysis.temporal_comparison if evidence.analysis else None
    if comparison is not None:
        dates.update(
            o.acquired_at.isoformat()[:10]
            for o in (comparison.first, comparison.second)
            if o.acquired_at is not None
        )
    for window in evidence.execution.windows if evidence.execution else []:
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

    iso: set[str] = set()
    month_years: set[str] = set()
    years: set[str] = set()

    def allow(year: int, month: int, iso_date: str) -> None:
        iso.add(iso_date)
        years.add(f"{year:04d}")
        month_years.add(f"{_MONTH_NAMES[month - 1]} {year:04d}")

    for window in _windows(evidence.execution.plan.intent) if evidence.execution else []:
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

    ids: set[str] = set()
    sar = evidence.analysis.sar_backscatter if evidence.analysis else None
    if sar is not None:
        ids.add(sar.scene_id)
    comparison = evidence.analysis.temporal_comparison if evidence.analysis else None
    if comparison is not None:
        ids.update((comparison.first.scene_id, comparison.second.scene_id))
    for window in evidence.execution.windows if evidence.execution else []:
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

    literal = literal.replace("−", "-")
    cleaned = literal.replace(",", "").lstrip("+")
    try:
        stated = float(cleaned)
    except ValueError:  # pragma: no cover - the regex cannot produce this
        return False

    # Malformed grouping and enormous exponents must not become a partial
    # number, infinity, or an unbounded precision request.
    if not re.fullmatch(
        r"[-+]?(?:\d+|\d{1,3}(?:,\d{3})+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d{1,3})?",
        literal,
    ):
        return False
    places = _decimals(cleaned)
    if not -308 <= places <= 308:
        return False
    return any(abs(round(candidate, places) - stated) <= _FLOAT_EPSILON for candidate in allowed)


def _mask(text: str, literals: Iterable[str]) -> str:
    """Blank out exact substrings so their digits are not read as claims."""

    masked = text
    for literal in sorted(literals, key=len, reverse=True):
        if literal:
            masked = re.sub(
                r"(?<![\w.])" + re.escape(literal) + r"(?!\w|\.\d)",
                lambda match: " " * len(match.group()),
                masked,
            )
    return masked


_METRIC = re.compile(r"\b(?:ndvi|ndwi|ndbi|vv|vh)\b", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?!\d)|\n+")
_CLAUSES = re.compile(r";|\b(?:and|with|over|across|but|while)\b", re.IGNORECASE)
_UNITS = {
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "index": "index",
    "db": "dB",
    "pixels": "pixels",
    "pixel": "pixels",
    "scenes": "count",
    "scene": "count",
    "count": "count",
    "m": "m",
    "metres": "m",
    "meters": "m",
    "km2": "km2",
    "km²": "km2",
    "hectares": "ha",
    "ha": "ha",
    "x": "x",
    "days": "days",
    "day": "days",
}
_ABSTENTIONS = frozenset(
    {
        "insufficient evidence to answer the question",
        "there is insufficient evidence to answer the question",
        "the available evidence does not answer the question",
        "i cannot determine this from the available evidence",
    }
)


_MEASUREMENT_WORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "are",
        "of",
        "for",
        "on",
        "in",
        "at",
        "from",
        "to",
        "about",
        "approximately",
        "mean",
        "average",
        "averaged",
        "minimum",
        "min",
        "maximum",
        "max",
        "ndvi",
        "ndwi",
        "ndbi",
        "vv",
        "vh",
        "db",
        "minus",
        "backscatter",
        "index",
        "value",
        "valid",
        "paired",
        "pixel",
        "pixels",
        "percent",
        "percentage",
        "%",
        "imagery",
        "gives",
        "scene",
        "scenes",
        "count",
        "difference",
        "change",
        "first",
        "second",
        "earlier",
        "later",
        "baseline",
        "target",
        "selected",
        "measured",
        "computed",
        "reported",
        "analysed",
        "analyzed",
    ]
)


def _words(text: str) -> str:
    """Punctuation/case normalization only; never discard negation or qualifiers."""
    return " ".join(re.findall(r"[\w%]+", text.lower()))


def _is_date_year(text: str, match: re.Match[str]) -> bool:
    """A known year is not an exemption for a measurement equal to that year."""
    return bool(
        re.search(r"\b(?:in|from|during|year|for)\s*$", text[: match.start()], re.IGNORECASE)
        or re.match(r"\s+(?:acquisition|imagery|scene)\b", text[match.end() :], re.IGNORECASE)
    )


def _kind(text: str) -> str | None:
    text = text.lower().replace("_", " ")
    if re.search(r"percent|%", text):
        return "percent"
    if "pixel" in text:
        return "paired_pixels" if "paired" in text else "pixels"
    if re.search(r"scene\s+count|\bscenes\b", text):
        return "scenes"
    if "difference" in text or re.search(r"\bvv\s*(?:minus|[-−])\s*vh\b", text):
        return "difference"
    stats = {
        key
        for key, pattern in (
            ("min", r"\bmin(?:imum)?\b"),
            ("max", r"\bmax(?:imum)?\b"),
            ("mean", r"\bmean\b|\baveraged?\b"),
        )
        if re.search(pattern, text)
    }
    if len(stats) > 1:
        return None
    stat = next(iter(stats), None)
    if "change" in text:
        return f"change_{stat or 'mean'}"
    return stat or ("mean" if _METRIC.search(text) else None)


def _observations(item: EvidenceItem, evidence: AgentEvidence) -> list[tuple[str, str, str]]:
    """Existing provenance only: (scene id, acquisition date, requested role)."""
    comparison = evidence.analysis.temporal_comparison if evidence.analysis else None
    if item.id.startswith("temporal_ndwi."):
        if comparison is None:
            return []
        observations = [comparison.first, comparison.second]
        if item.id.startswith("temporal_ndwi.first."):
            observations = observations[:1]
        elif item.id.startswith("temporal_ndwi.second."):
            observations = observations[1:]
        return [
            (o.scene_id, o.acquired_at.isoformat()[:10] if o.acquired_at else "", o.window_label)
            for o in observations
        ]
    if item.source == "sar_backscatter":
        sar = evidence.analysis.sar_backscatter if evidence.analysis else None
        if sar is None:
            return []
        return [(sar.scene_id, sar.acquired_at.isoformat()[:10] if sar.acquired_at else "",
                 sar.window_label)]
    execution = evidence.execution
    if execution is None:
        return []
    windows = execution.windows
    if item.source in {"ndvi", "ndwi", "ndbi"}:
        windows = [
            w for w in windows if w.modality == "sentinel-2-optical" and w.selected_scene_id
        ][:1]
    elif item.source == "execution":
        windows = [w for w in windows if item.id == f"execution.{w.modality}.{w.label}.scene_count"]
    else:
        return []
    return [
        (s.id, (s.datetime or "")[:10], w.label)
        for w in windows
        for s in w.scenes
        if s.id == w.selected_scene_id
    ]


def _scope_matches(item: EvidenceItem, clause: str, evidence: AgentEvidence) -> bool:
    observations = _observations(item, evidence)
    for platform in _PLATFORM_IDENTIFIER.finditer(clause):
        sar = platform.group().lower().startswith("sentinel-1")
        if sar and item.source in {"ndvi", "ndwi", "ndbi", "temporal_ndwi"}:
            return False
        if not sar and item.source == "sar_backscatter":
            return False
        if item.source == "execution":
            modality = "sentinel-1-sar" if sar else "sentinel-2-optical"
            if not item.id.startswith(f"execution.{modality}."):
                return False
    scenes = {
        scene
        for scene in _scene_ids(evidence)
        if re.search(r"(?<!\w)" + re.escape(scene) + r"(?!\w)", clause)
    }
    dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", clause))
    if scenes and not scenes <= {scene for scene, _, _ in observations}:
        return False
    if dates and not dates <= {date for _, date, _ in observations}:
        return False
    years = {m.group(1) for m in _YEAR.finditer(clause) if _is_date_year(clause, m)}
    if years and not years <= {date[:4] for _, date, _ in observations}:
        return False
    for match in _MONTH_YEAR.finditer(clause):
        prefix = f"{match.group(2)}-{_MONTH_NAMES.index(match.group(1).lower()) + 1:02d}"
        if not any(date.startswith(prefix) for _, date, _ in observations):
            return False
    for role in ("first", "second", "earlier", "later", "baseline", "target"):
        if not re.search(r"\b" + role + r"\b", clause, re.IGNORECASE):
            continue
        position = {"earlier": "first", "later": "second"}.get(role, role)
        if position in {"first", "second"}:
            if not item.id.startswith(f"temporal_ndwi.{position}."):
                return False
        elif role not in {label for _, _, label in observations}:
            return False
    return True


#: A unit stated BEFORE the number and bound to it by the phrasing, as in
#: "the mean NDWI in dB is 0.2777". Only text between the unit and the number
#: that is pure linking prose ("is", "was", "of", "about") counts as binding, so
#: a "dB" mentioned elsewhere in an explanatory sentence is NOT read as a claim
#: about this measurement.
_LEADING_UNIT = re.compile(
    r"\b(?:in|expressed\s+in|measured\s+in|given\s+in|as)\s+(%|[A-Za-z]+[\u00b22]?)\s*"
    r"(?:\b(?:is|was|are|were|of|at|the|a|an|about|approximately|roughly|"
    r"around|equal|to|value|values)\b\s*)*$",
    re.IGNORECASE,
)


def _leading_unit(before: str) -> str | None:
    """The unit a clause claims BEFORE stating the number, if it states one.

    `_supporting_measurements` used to read only the text AFTER the literal, so
    "the mean NDWI in dB is 0.2777" bound to an `index` measurement and passed:
    every number was real, and the sentence still asserted a decibel figure the
    evidence never contained. Reading only one side of the number checked half
    the claim.

    Returns None for an unrecognised word, which keeps ordinary prose
    unaffected - "NDWI in the bay is 0.2777" states no unit, and inventing one
    from "bay" would reject a sentence that is perfectly well grounded.
    """

    match = _LEADING_UNIT.search(before)
    if match is None:
        return None
    return _UNITS.get(match.group(1).lower())


def _supporting_measurements(
    clause: str, sentence: str, literal: str, evidence: AgentEvidence
) -> list[EvidenceItem]:
    """Match identity BEFORE value. Ambiguous implicit identities fail closed."""
    metrics = {m.group().lower() for m in _METRIC.finditer(clause)}
    if not metrics:
        metrics = {m.group().lower() for m in _METRIC.finditer(sentence)}
    _, numeric_text = _numeric_text(clause, evidence)
    remainder = _NUMBER.sub(" ", numeric_text)
    if evidence.execution is not None:
        remainder = re.sub(
            re.escape(evidence.execution.plan.intent.location_query),
            " ",
            remainder,
            flags=re.IGNORECASE,
        )
    if set(_words(remainder).split()) - _MEASUREMENT_WORDS:
        return []
    if len(list(_NUMBER.finditer(numeric_text))) != 1:
        return []  # use separate clauses for separate measurements
    before, after = clause.split(literal, 1)
    kind = _kind(clause)
    # Read an explicit unit adjacent to this number, including unsupported
    # attached suffixes. An index value can never authorize metres or ratios.
    suffix = re.match(r"(\s*)(%|[A-Za-z]+[²2]?)", after)
    unit = None
    if suffix:
        token = suffix.group(2).lower()
        unit = _UNITS.get(token)
        if not suffix.group(1) and unit is None:
            return []
    # A unit may also be stated ahead of the number. Both sides are checked
    # against the same measurement, and a clause that names two different units
    # for one value contradicts itself and is refused outright.
    leading = _leading_unit(before)
    if leading is not None:
        if unit is not None and unit != leading:
            return []
        unit = leading
    if kind is None and re.search(r"\bvalid\s+pixels?\b", after, re.IGNORECASE):
        kind = "pixels"
    candidates = []
    for item in evidence.items:
        measurement = item.measurement
        if (
            measurement is None
            or item.source not in _NUMERIC_AUTHORITIES
            or not item.id.startswith(f"{item.source}.")
        ):
            continue
        item_metrics = {
            m.group().lower() for m in _METRIC.finditer(measurement.name.replace("_", " "))
        }
        if (
            item_metrics
            and item.source not in item_metrics
            and not (item.source == "temporal_ndwi" and item_metrics == {"ndwi"})
            and not (item.source == "sar_backscatter" and item_metrics <= {"vv", "vh"})
        ):
            continue
        if measurement.name == "paired_valid_pixel_count" and item.source == "temporal_ndwi":
            item_metrics = {"ndwi"}
        if item.source == "sar_backscatter" and (
            not metrics or measurement.name not in {
                "vv_mean_db", "vv_min_db", "vv_max_db", "vv_valid_pixel_count",
                "vh_mean_db", "vh_min_db", "vh_max_db", "vh_valid_pixel_count",
                "vv_minus_vh_mean_db",
            }
        ):
            continue
        if metrics and metrics != item_metrics:
            continue
        if kind is None or kind != _kind(measurement.name):
            continue
        expected_unit = {
            "percent": "%",
            "pixels": "pixels",
            "paired_pixels": "pixels",
            "scenes": "count",
        }.get(kind, "dB" if item.source == "sar_backscatter" else "index")
        if measurement.unit != expected_unit:
            continue
        if unit is not None and unit != measurement.unit:
            continue
        if not _scope_matches(item, clause, evidence):
            continue
        candidates.append(item)
    # Flat references do not say which of two same-kind observations a bare
    # "mean" refers to. Do not use the claimed value to guess its identity.
    if len(candidates) != 1:
        return []
    item = candidates[0]
    assert item.measurement is not None
    return [item] if _is_grounded(literal, [item.measurement.value]) else []


def _parts(text: str) -> Iterable[tuple[str, str]]:
    for sentence in _SENTENCES.split(text):
        for clause in _CLAUSES.split(sentence):
            if clause.strip():
                yield sentence, clause


def _prose_supported(summary: str, evidence: AgentEvidence) -> bool:
    """Closed descriptive vocabulary; qualitative claims must repeat a citation.

    This check is reported through the existing evidence_refs outcome. It is
    support checking, not a new AgentEvidence shape or a fourth API check.
    """
    quoted = {
        _words(sentence)
        for item in evidence.items
        if item.source in _NUMERIC_AUTHORITIES | {"model"}
        for sentence in _SENTENCES.split(item.visual.statement if item.visual else item.text or "")
        if sentence.strip()
    }
    # Exact metadata statements are supported only by the cited item's own
    # observation, never by some other candidate scene in the execution.
    for item in evidence.items:
        for scene, acquired, _ in _observations(item, evidence):
            quoted.add(_words(f"Scene {scene} was selected."))
            if acquired:
                quoted.add(_words(f"The scene was acquired on {acquired}."))
    for sentence in _SENTENCES.split(summary):
        normalized = _words(sentence)
        if not normalized:
            continue
        if normalized in quoted or normalized in _ABSTENTIONS:
            continue
        # Permit only this introduction to an otherwise exact cited sentence.
        # Keep numeric validation independent; model numbers gain no authority.
        introduced = re.match(r"^\s*yes,\s+(.+)$", sentence, re.IGNORECASE)
        if introduced and _words(introduced.group(1)) in quoted:
            continue
        for _, clause in _parts(sentence):
            _, numeric_text = _numeric_text(clause, evidence)
            numbers = list(_NUMBER.finditer(numeric_text))
            if not numbers or not all(
                _supporting_measurements(clause, sentence, number.group(), evidence)
                for number in numbers
            ):
                return False
    return bool(summary.strip())


def _numeric_text(summary: str, evidence: AgentEvidence) -> tuple[list[str], str]:
    """Separate identifiers/date context from quantitative literals.

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
        if any(c.isdigit() for c in match.group(0)) and any(c.isalpha() for c in match.group(0))
    ]
    text = _mask(text, invented)

    # Allowed date forms are removed; unrecognised ones are deliberately left
    # in place so their digits surface as unaccounted numbers.
    text = _mask(text, iso_dates)
    text = _MONTH_YEAR.sub(lambda m: " " if m.group(0).lower() in month_years else m.group(0), text)
    text = _YEAR.sub(
        lambda m: " " if m.group(1) in years and _is_date_year(text, m) else m.group(0), text
    )

    # A platform name is not a measurement. This masks the identifier only, so
    # units, ordinary numbers and scientific notation beside it stay claims.
    text = _PLATFORM_IDENTIFIER.sub(" ", text)

    return invented, text


#: Evidence ids that carry a deterministic engine's own caveat: a warning or a
#: limitation it attached to its result.
_CAVEAT_ID = re.compile(r"\.(?:warning|limitation)\.\d+$")


def _verbatim(text: str) -> str:
    """Whitespace, case and closing punctuation only.

    Digits, signs and decimal points are kept exactly - unlike ``_words`` - so
    a copy with any figure altered is a different sentence.
    """

    return " ".join(text.split()).casefold().rstrip(".!?")


def _repeated_caveats(evidence: AgentEvidence) -> set[str]:
    """Sentences of the cited evidence's own caveats, for verbatim repetition.

    An engine writes these, never a model and never from the question or the
    plan, so repeating one exactly asserts nothing the system did not already
    assert. Observed live: the NDBI caveat "NDBI uses a 20 m band, so it is
    sampled on the 10 m grid but resolves detail no finer than 20 m." quoted
    beside a correct NDBI value withheld the whole answer - 20 and 10 matched
    no measurement - while ``_prose_supported`` accepted the same sentence as a
    repetition of cited evidence.

    Only warnings and limitations from a numeric authority qualify. Model
    observations are excluded by source, and the executor's failure notes by
    id: those relay an error message rather than an engine's caveat.
    """

    return {
        _verbatim(sentence)
        for item in evidence.items
        if item.source in _NUMERIC_AUTHORITIES and item.text and _CAVEAT_ID.search(item.id)
        for sentence in _SENTENCES.split(item.text)
        if sentence.strip()
    }


def _ungrounded_claims(summary: str, evidence: AgentEvidence) -> list[str]:
    """Unmatched numbers, checked against identity-bound evidence, not a pool.

    A sentence repeating a cited caveat exactly is not read for claims: its
    figures are the engine's own, cited and unaltered. They authorise nothing
    elsewhere - the whole sentence must match, every digit included.
    """
    caveats = _repeated_caveats(evidence)
    failures: list[str] = []
    for sentence, clause in _parts(summary):
        if _verbatim(sentence) in caveats:
            continue
        invented, text = _numeric_text(clause, evidence)
        failures.extend(invented)
        failures.extend(
            match.group()
            for match in _NUMBER.finditer(text)
            if not _supporting_measurements(clause, sentence, match.group(), evidence)
        )
    return failures


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


def validate_answer(draft: DraftAnswer, evidence: AgentEvidence) -> AnswerValidation:
    """Run every mechanical check and report each outcome independently.

    The three checks do not short-circuit one another: a forbidden phrase does
    not hide an ungrounded number, and vice versa, so a caller sees everything
    that is wrong rather than only the first thing.

    ``visual_claims`` is NOT a fourth check. It is a statement of provenance:
    when the evidence includes a model observation, the answer rests partly on
    something no mechanical rule can validate, and saying so is the honest
    alternative to letting a reader infer verification from three passes. The
    numeric guarantee is unaffected - a model still cannot authorise a number.

    Pure: no I/O, no clock, no randomness, and neither argument is mutated.
    """

    cited = evidence.model_copy(
        update={"items": [item for item in evidence.items if item.id in set(draft.evidence_refs)]}
    )
    numeric = "fail" if _ungrounded_claims(draft.summary, cited) else "pass"
    forbidden = "fail" if find_forbidden_phrases(draft.summary) else "pass"
    references = (
        "fail"
        if unresolved_references(draft, evidence) or not _prose_supported(draft.summary, cited)
        else "pass"
    )
    visual = (
        "attributed"
        if any(item.source == "model" and item.visual is not None for item in cited.items)
        else "not_run"
    )

    return AnswerValidation(
        numeric_grounding=numeric,
        forbidden_terms=forbidden,
        evidence_refs=references,
        visual_claims=visual,
    )
