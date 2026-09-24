"""Deterministic interpretation of a natural-language satellite question.

    question -> QueryInterpretation            (every fact the workflow needs)
             -> ClarificationRequiredError          (a fact is missing or unsupported)

This is the provider-independent half of natural-language querying. It turns a
supported question into the SAME typed plan an AI planner would propose - a
``SatQueryIntent`` inside an ``execute_query`` step, plus the analysis tools -
so everything downstream (geocoding, discovery, scene/pixel/radiometric/
geometric validation, the engines, grounding, evidence) runs unchanged.

What it decides, and nothing more:

* **which analysis**: NDVI, NDWI, NDBI, SAR backscatter, a two-period NDWI
  comparison, or true-colour imagery - from the words the question uses;
* **where**: the place text exactly as the question names it. It is NEVER
  geocoded here; the geospatial service remains the authority on coordinates;
* **when**: explicit dates only - a day, a month, a year, or a range or pair of
  them. Nothing is assumed: no "today", no default window, no season boundaries;
* **which sensor**: follows from the analysis (optical for the indices and for
  imagery, Sentinel-1 RTC for backscatter). The catalogs remain authoritative
  for what actually exists.

What it never does: compute, estimate, geocode, call a model, or fill a gap by
guessing. Every gap becomes a :class:`ClarificationRequiredError` naming the missing
fact and the choices that ARE supported. A question it cannot map confidently
is refused, never approximated - an approximated question executes work the
user did not ask for, and the result would then be presented as an answer.

The reading is shallow on purpose: fixed vocabularies plus the clause-polarity
rules :func:`~app.services.agent.plan_completion.requested_matches` already
applies to plan completion, so "show water, not vegetation" asks for water
alone. Anything outside the vocabularies is outside the supported set, and
saying so is the correct answer.

Purity: no network, no provider, no service handle. The only clock read is the
observability bound shared with ``SatQueryIntent`` (no window may start in the
future), and a caller may pin it.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

from app.core.errors import InvalidInputError
from app.services.agent.plan_completion import requested_matches
from app.services.agent.schemas import (
    AgentClarification,
    AgentPlan,
    ClarificationReason,
    ExecuteQueryParams,
    SarBackscatterParams,
    SpectralIndicesParams,
    TemporalNdwiParams,
    ToolCall,
)
from app.services.query.schemas import (
    SatQueryIntent,
    TemporalComparison,
    TimeRange,
    observation_period_problem,
)

#: The analyses this interpreter can route to - each one a real, deterministic
#: capability of the backend. Nothing is listed that the engines cannot do.
AnalysisKey = Literal["ndvi", "ndwi", "ndbi", "sar_backscatter", "temporal_ndwi", "imagery"]

#: How each analysis is named to a reader, in clarifications and options.
ANALYSIS_LABELS: dict[str, str] = {
    "ndvi": "vegetation (NDVI)",
    "ndwi": "water (NDWI)",
    "ndbi": "built-up area (NDBI)",
    "sar_backscatter": "SAR backscatter (Sentinel-1 VV/VH)",
    "temporal_ndwi": "water compared between two periods (NDWI)",
    "imagery": "true-colour imagery (Sentinel-2)",
}

#: What a reader may ask for when their question named no analysis. Listed in
#: the order the workspace examples use.
SUPPORTED_OPTIONS: tuple[str, ...] = (
    ANALYSIS_LABELS["ndvi"],
    ANALYSIS_LABELS["ndwi"],
    ANALYSIS_LABELS["ndbi"],
    ANALYSIS_LABELS["sar_backscatter"],
    ANALYSIS_LABELS["temporal_ndwi"],
)

_OPTICAL_INDICES: tuple[str, ...] = ("ndvi", "ndwi", "ndbi")


class ClarificationRequiredError(InvalidInputError):
    """The question cannot be executed as asked; this says what is missing.

    An :class:`InvalidInputError` (422) because it is exactly that: the request
    is well formed and cannot be acted on. The structured
    :class:`AgentClarification` travels with it, so the agent route can return
    it as a result and the parse route as an error, from one source.
    """

    code = "clarification_required"

    def __init__(self, clarification: AgentClarification) -> None:
        super().__init__(clarification.message)
        self.clarification = clarification


# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #


def _terms(body: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive alternation."""

    return re.compile(rf"(?<![A-Za-z0-9_])(?:{body})(?![A-Za-z0-9_])", re.IGNORECASE)


#: Words that ask for each analysis. Deliberately conservative: a word that is
#: commonly part of something else ("green" in a place name, "buildings" in a
#: request to count them) is left out, because a false route executes work
#: nobody asked for.
_ANALYSIS_TERMS: dict[str, re.Pattern[str]] = {
    "ndvi": _terms(
        r"ndvi|vegetation|vegetated|greenery|greenness|green\s+cover|plant\s+cover|"
        r"crops?|cropland|farmland|forests?|forest\s+cover|tree\s+cover"
    ),
    "ndwi": _terms(r"ndwi|water|waters|water\s*bod(?:y|ies)|surface\s+water|open\s+water"),
    "ndbi": _terms(
        r"ndbi|built[-\s]?up|urban|urbani[sz]ation|urbani[sz]ed|impervious|"
        r"settlements?|bare\s+(?:soil|land|ground)"
    ),
    "sar_backscatter": _terms(
        r"sar|radar|backscatter|backscattering|sentinel[-\s]?1[ab]?|s1|vv|vh|"
        r"polari[sz]ations?"
    ),
    "imagery": _terms(
        r"imagery|images?|pictures?|photos?|true[-\s]colou?r|rgb|satellite\s+view"
    ),
}

#: A request to compare two periods. "Change" alone is enough: a question about
#: how something changed IS a comparison, and it needs two periods to answer.
_COMPARISON = _terms(
    r"compare[sd]?|comparing|comparison|changes?|changed|changing|difference|"
    r"differ(?:s|ed)?|versus|vs\.?|before\s+and\s+after|increase[sd]?|"
    r"decrease[sd]?|grew|grown|growth|shr[ai]nk|shrunk|expan(?:d|ded|sion)|"
    r"loss|lost|gain(?:ed)?|trends?"
)

#: Recognised requests the backend does not implement, refused even beside a
#: supported one: running the supported half would present a partial answer to
#: a question about something SatQuery cannot measure.
_UNSUPPORTED: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        _terms(r"flood(?:s|ed|ing)?|inundat\w*"),
        "SatQuery does not map or classify floods. It can measure the "
        "water-like response as NDWI index statistics - ask about 'water' or "
        "'NDWI' for a place and a period.",
    ),
    (
        _terms(
            r"count(?:s|ing)?|how\s+many|classif\w*|segment\w*|ships?|vessels?|"
            r"vehicles?|cars?|aircraft|planes?|land[-\s]?cover|land[-\s]?use|objects?"
        ),
        "Object identification and land-cover classification are not "
        "implemented. SatQuery measures spectral indices and SAR backscatter "
        "over an area.",
    ),
)

#: Verbs that ask for a detection. With a supported analysis beside them
#: ("detect water") the measurement is what can be delivered - and it is
#: reported as index statistics, never as a detection. Alone, they ask for
#: something SatQuery does not do.
_DETECTION = _terms(r"detect(?:s|ed|ion|ing)?|identify|identification|locate|find\s+all")
_DETECTION_MESSAGE = (
    "Detection and identification are not implemented. SatQuery measures "
    "spectral indices and SAR backscatter over an area - for example the "
    "water index (NDWI) of a place in a month."
)

#: Asking what an image SHOWS needs a model that can look at it.
_VISUAL = _terms(
    r"visible|visibly|visually|looks?\s+like|can\s+(?:you\s+)?see|can\s+be\s+seen|"
    r"describe|description|appear(?:s|ance)?"
)

#: A sensor named as the SOURCE of an analysis ("vegetation using radar"). Only
#: meaningful when it contradicts the analysis asked for.
_SAR_SOURCE = re.compile(
    r"\b(?:using|with|from|via|by|on)\s+(?:the\s+)?(?:sar|radar|sentinel[-\s]?1[ab]?|s1)\b",
    re.IGNORECASE,
)
_OPTICAL_SOURCE = re.compile(
    r"\b(?:using|with|from|via|by|on)\s+(?:the\s+)?(?:sentinel[-\s]?2[abc]?|s2|optical|"
    r"multispectral)\b",
    re.IGNORECASE,
)

#: A VH request changes which polarization is DISPLAYED; both are measured.
_VH_ONLY = _terms(r"vh")
_VV = _terms(r"vv")


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

_MONTH_NUMBERS: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_ORD = r"(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)\d{2}"

#: Most specific first; an earlier pattern's span wins over a later one's.
_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("iso_day", re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])")),
    ("iso_month", re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})(?![\d-])")),
    (
        "day_span",
        re.compile(
            rf"(?<![\d.])(\d{{1,2}}){_ORD}\s*(?:-|–|to|until|till|through)\s*"
            rf"(\d{{1,2}}){_ORD}\s+(?:of\s+)?({_MONTH})\.?,?\s+({_YEAR})(?!\d|\.\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "dmy",
        re.compile(
            rf"(?<![\d.])(\d{{1,2}}){_ORD}\s+(?:of\s+)?({_MONTH})\b\.?(?:,?\s+({_YEAR}))?(?!\d|\.\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "mdy",
        re.compile(
            rf"\b({_MONTH})\.?\s+(\d{{1,2}}){_ORD}(?:,?\s+({_YEAR}))?(?!\d|\.\d)",
            re.IGNORECASE,
        ),
    ),
    ("my", re.compile(rf"\b({_MONTH})\b\.?,?\s+({_YEAR})(?!\d|\.\d)", re.IGNORECASE)),
    ("month", re.compile(rf"\b({_MONTH})\b", re.IGNORECASE)),
    ("year", re.compile(rf"(?<![\d.,-])({_YEAR})(?!\d|\.\d)")),
)

#: "may" is a month only when something says it is a date - otherwise "water
#: may be present" would ask which year May is.
_MAY_AS_MONTH_BEFORE = re.compile(
    r"\b(?:in|during|of|between|from|to|until|till|through|since|and|early|mid|late)\s*$",
    re.IGNORECASE,
)

#: Expressions whose dates this interpreter will not invent.
_RELATIVE = _terms(
    r"today|yesterday|tomorrow|tonight|now|currently|current|latest|recent(?:ly)?|"
    r"last\s+(?:week|month|year|summer|winter|monsoon|spring|autumn)|"
    r"this\s+(?:week|month|year|summer|winter|monsoon|spring|autumn)|"
    r"next\s+(?:week|month|year)|past\s+(?:\d+\s+)?(?:days?|weeks?|months?|years?)|ago"
)
_SEASON = _terms(r"summer|winter|monsoon|spring|autumn|pre[-\s]?monsoon|post[-\s]?monsoon")
_QUALIFIED = re.compile(
    rf"\b(?:early|mid|middle\s+of|late|end\s+of|beginning\s+of|start\s+of|"
    rf"first\s+half\s+of|second\s+half\s+of)\s*-?\s*(?={_MONTH}\b|{_YEAR})",
    re.IGNORECASE,
)

#: What may sit between two dates that belong together.
_RANGE_LINK = re.compile(r"^(?:to|till|until|through|thru|-|–|—)$", re.IGNORECASE)
_PAIR_LINK = re.compile(
    r"^(?:and|vs\.?|versus|compared\s+(?:to|with)|against|to|till|until|through|thru|-|–|—)$",
    re.IGNORECASE,
)
_RANGE_FRAME = re.compile(r"\b(?:between|from)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class _DateExpr:
    start: int
    end: int
    kind: Literal["day", "day_span", "month", "year"]
    year: int | None
    month: int | None = None
    day: int | None = None
    day_to: int | None = None
    text: str = ""


def _month_number(token: str) -> int:
    return _MONTH_NUMBERS[token.lower()[:3]]


def _find_dates(text: str) -> list[_DateExpr]:
    """Every explicit date expression, in document order, without overlaps."""

    taken: list[tuple[int, int]] = []
    found: list[_DateExpr] = []

    def free(start: int, end: int) -> bool:
        return all(end <= lo or start >= hi for lo, hi in taken)

    for kind, pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if not free(start, end):
                continue
            groups = match.groups()
            expr: _DateExpr | None
            if kind == "iso_day":
                expr = _DateExpr(start, end, "day", int(groups[0]), int(groups[1]), int(groups[2]))
            elif kind == "iso_month":
                expr = _DateExpr(start, end, "month", int(groups[0]), int(groups[1]))
            elif kind == "day_span":
                expr = _DateExpr(
                    start, end, "day_span", int(groups[3]), _month_number(groups[2]),
                    int(groups[0]), int(groups[1]),
                )
            elif kind == "dmy":
                expr = _DateExpr(
                    start, end, "day", int(groups[2]) if groups[2] else None,
                    _month_number(groups[1]), int(groups[0]),
                )
            elif kind == "mdy":
                expr = _DateExpr(
                    start, end, "day", int(groups[2]) if groups[2] else None,
                    _month_number(groups[0]), int(groups[1]),
                )
            elif kind == "my":
                expr = _DateExpr(start, end, "month", int(groups[1]), _month_number(groups[0]))
            elif kind == "month":
                if groups[0].lower() == "may" and not _MAY_AS_MONTH_BEFORE.search(
                    text[:start]
                ):
                    continue
                expr = _DateExpr(start, end, "month", None, _month_number(groups[0]))
            else:
                expr = _DateExpr(start, end, "year", int(groups[0]))
            taken.append((start, end))
            found.append(_DateExpr(**{**expr.__dict__, "text": match.group(0)}))

    return sorted(found, key=lambda expr: expr.start)


def _link(text: str, first: _DateExpr, second: _DateExpr) -> str:
    return " ".join(text[first.end : second.start].split())


def _inherit_years(text: str, exprs: list[_DateExpr]) -> list[_DateExpr]:
    """A yearless day or month takes the year of the LINKED date after it.

    "between January and March 2025" states 2025 once, for both. Only a direct
    link carries it - two unrelated dates never share a year by proximity.
    """

    result = list(exprs)
    for index in range(len(result) - 2, -1, -1):
        current, following = result[index], result[index + 1]
        if current.year is None and following.year is not None and _PAIR_LINK.match(
            _link(text, current, following)
        ):
            result[index] = _DateExpr(**{**current.__dict__, "year": following.year})
    return result


def _to_range(expr: _DateExpr) -> TimeRange:
    """The closed interval an expression names. Raises ``ValueError`` if impossible."""

    assert expr.year is not None
    if expr.kind == "year":
        return TimeRange(start_date=date(expr.year, 1, 1), end_date=date(expr.year, 12, 31))
    assert expr.month is not None
    if expr.kind == "month":
        last = calendar.monthrange(expr.year, expr.month)[1]
        return TimeRange(
            start_date=date(expr.year, expr.month, 1),
            end_date=date(expr.year, expr.month, last),
        )
    assert expr.day is not None
    if expr.kind == "day_span":
        assert expr.day_to is not None
        return TimeRange(
            start_date=date(expr.year, expr.month, expr.day),
            end_date=date(expr.year, expr.month, expr.day_to),
        )
    day = date(expr.year, expr.month, expr.day)
    return TimeRange(start_date=day, end_date=day)


# --------------------------------------------------------------------------- #
# Location
# --------------------------------------------------------------------------- #

#: Words that introduce the place a question is about.
_LOCATION_LEAD = re.compile(
    r"\b(?:around|in|at|over|near|of|for|across|within|inside|surrounding|"
    r"covering|along|about|nearby)\s+",
    re.IGNORECASE,
)

#: Where a place phrase ends. The marker ``|`` stands for a removed date or
#: sensor phrase.
_LOCATION_END = re.compile(
    r"\s*\|"
    r"|[?!;\n]"
    r"|\.(?!\d)"
    r"|,\s*(?=(?:and|then|please|compute|calculate|show|measure|find|analy[sz]e|compare|"
    r"what|how|is|are|can|could|using|with|over|for)\b)"
    r"|\s+(?:using|with|via|please|then|so\s+that|"
    r"and\s+(?:compute|calculate|show|measure|find|analy[sz]e|compare|map|plot|display|"
    r"then|also)|"
    r"to\s+(?:compute|calculate|show|measure|find|see|analy[sz]e|compare|check|map|"
    r"detect|identify)|"
    r"today|yesterday|now|currently|latest|recently|last\s+\w+|this\s+(?:week|month|"
    r"year)|next\s+\w+|past\s+(?:\d+\s+)?\w+|\d+\s+\w+\s+ago)\b",
    re.IGNORECASE,
)

#: Connectors a place phrase may have been left ending with.
_TRAILING_CONNECTOR = re.compile(
    r"(?:\s+|^)(?:in|on|at|and|or|for|to|during|from|between|since|of|by|over|around|"
    r"near|the|a|an|please|now)\s*$",
    re.IGNORECASE,
)

#: A reference to a place the question does not name.
_DEICTIC = re.compile(
    r"^(?:this|that|these|those|here|there|it|same|my|our|the\s+(?:same|current|"
    r"selected|shown|above|given)\b)",
    re.IGNORECASE,
)

#: A phrase whose head is a generic noun ("the area near ...", "the Sentinel-2
#: image of ..."): the place, if any, comes after it.
_GENERIC_HEAD = re.compile(
    r"^(?:(?:the|a|an)\s+)?(?:(?:whole|entire|surrounding|nearby|same|latest|"
    r"satellite|optical|sentinel[-\s]?[12][abc]?)\s+)*"
    r"(?:area|region|vicinity|surroundings|neighbou?rhood|image|imagery|scene|"
    r"picture|photo|map|data|tile|place|location|spot)s?\b",
    re.IGNORECASE,
)

#: Words that can be a whole "phrase" without naming anywhere.
_NOT_A_PLACE = _terms(
    r"me|us|it|them|general|detail|details|now|please|time|period|date|dates|"
    r"something|anything|everything|information|analysis|index|indices|"
    r"statistics|stats|values?|results?|measurements?"
)

#: Sensor phrases that read like a place ("in the Sentinel-2 image").
_SENSOR_PHRASE = re.compile(
    r"\b(?:using|with|from|via|by|on|in)\s+(?:the\s+)?(?:latest\s+)?"
    r"(?:sentinel[-\s]?[12][abc]?|s[12]|optical|sar|radar|landsat|satellite)"
    r"(?:\s+(?:data|imagery|images?|scenes?|products?))?\b",
    re.IGNORECASE,
)

#: Coordinates as the geocoder accepts them: "lat, lon".
_COORDINATES = re.compile(r"^[-+]?\d{1,2}(?:\.\d+)?\s*,\s*[-+]?\d{1,3}(?:\.\d+)?$")

#: A framing word left in front of a removed date.
_DATE_FRAME = re.compile(
    r"\b(?:in|on|during|for|from|between|since|by|throughout|until|till|to|and|"
    r"through|vs\.?|versus|compared\s+(?:to|with)|of)\s*\|",
    re.IGNORECASE,
)


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace each span with the phrase marker, keeping every other offset."""

    chars = list(text)
    for start, end in spans:
        chars[start:end] = ["|"] + [" "] * (end - start - 1)
    return "".join(chars)


def _starts_with_term(phrase: str) -> bool:
    """Whether the phrase opens with an analysis or sensor word."""

    head = phrase.lstrip()
    for pattern in (*_ANALYSIS_TERMS.values(), _COMPARISON):
        match = pattern.match(head)
        if match is not None:
            return True
    return re.match(r"(?:the\s+)?(?:sentinel|landsat|optical|satellite)\b", head, re.I) is not None


@dataclass(frozen=True)
class _Place:
    text: str | None
    #: The raw phrase that pointed at a place without naming one, if any.
    deictic: str | None = None
    #: Where the chosen phrase sits in the masked text, so its words are not
    #: read as analysis requests ("Forest Hill" is a place, not NDVI).
    span: tuple[int, int] | None = None


def _find_place(masked: str) -> _Place:
    """The first phrase that names a place.

    Two passes. The first skips a phrase opening with an analysis word, because
    "the NDVI of vegetation in <a city>" names its place later. Only when no
    other phrase names one does the second pass accept such a phrase - and then
    only a Capitalised name with a word beyond the analysis vocabulary, so
    "Forest Hill" is a place while "of water" is not.
    """

    first = _scan_places(masked, allow_term_start=False)
    if first.text is not None:
        return first
    second = _scan_places(masked, allow_term_start=True)
    return second if second.text is not None else first


def _names_a_place_despite_terms(phrase: str) -> bool:
    words = re.findall(r"[A-Za-z][\w'-]*", phrase)
    leftover = [
        word for word in words
        if not any(p.fullmatch(word) for p in (*_ANALYSIS_TERMS.values(), _COMPARISON))
    ]
    return bool(leftover) and all(word[0].isupper() for word in words[:2])


def _scan_places(masked: str, *, allow_term_start: bool) -> _Place:
    deictic: str | None = None
    for lead in _LOCATION_LEAD.finditer(masked):
        begin = lead.end()
        stop = _LOCATION_END.search(masked, begin)
        end = stop.start() if stop is not None else len(masked)
        phrase = masked[begin:end]

        # Drop connectors left behind by a removed date ("<place> in |").
        previous = None
        while previous != phrase:
            previous = phrase
            phrase = _TRAILING_CONNECTOR.sub("", phrase.rstrip(" ,"))
        phrase = " ".join(phrase.split())

        if not phrase or _NOT_A_PLACE.fullmatch(phrase):
            continue
        if _DEICTIC.match(phrase):
            deictic = deictic or phrase
            continue
        if _GENERIC_HEAD.match(phrase):
            continue
        if _starts_with_term(phrase) and not (
            allow_term_start and _names_a_place_despite_terms(phrase)
        ):
            continue

        place = re.sub(r"^(?:the)\s+", "", phrase, flags=re.IGNORECASE)
        if not _COORDINATES.match(place):
            # "<landmark> in <city>" is one place, written nested.
            place = re.sub(r"\s+in\s+", ", ", place, flags=re.IGNORECASE)
        place = place.strip(" '\"“”‘’")
        if place:
            return _Place(text=place[:300], span=(begin, begin + len(masked[begin:end])))
    return _Place(text=None, deictic=deictic)


# --------------------------------------------------------------------------- #
# The interpretation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QueryInterpretation:
    """Every fact the existing workflow needs, and no prose.

    ``analyses`` is ordered by first mention. ``windows`` holds one range for a
    single-period question and two - earlier-stated first - for a comparison.
    """

    analyses: tuple[AnalysisKey, ...]
    location_query: str
    windows: tuple[TimeRange, ...]
    comparison: bool
    sar_polarization: Literal["vv", "vh"] = "vv"

    @property
    def modalities(self) -> list[Literal["sentinel-2-optical", "sentinel-1-sar"]]:
        modalities: list[Literal["sentinel-2-optical", "sentinel-1-sar"]] = []
        if any(key != "sar_backscatter" for key in self.analyses):
            modalities.append("sentinel-2-optical")
        if "sar_backscatter" in self.analyses:
            modalities.append("sentinel-1-sar")
        return modalities

    def intent(self) -> SatQueryIntent:
        """The existing intent contract, validated by its own rules."""

        if self.comparison:
            baseline, target = self.windows
            return SatQueryIntent(
                location_query=self.location_query,
                temporal_mode="compare",
                time_windows=TemporalComparison(baseline=baseline, target=target),
                modalities=self.modalities,
                task="visualize",
            )
        return SatQueryIntent(
            location_query=self.location_query,
            temporal_mode="single",
            time_windows=list(self.windows),
            modalities=self.modalities,
            # Every analysis here is a measurement or a view, never change
            # detection or object identification - both of which the backend
            # reports as not implemented.
            task="visualize",
        )

    def plan(self) -> AgentPlan:
        """The same validated plan an AI planner would have to propose."""

        steps: list[ToolCall] = [
            ExecuteQueryParams(
                intent=self.intent(),
                include_imagery=True,
                sar_polarization=self.sar_polarization,
            )
        ]
        indices = [key for key in self.analyses if key in _OPTICAL_INDICES]
        if indices:
            steps.append(SpectralIndicesParams(indices=indices))  # type: ignore[arg-type]
        if "sar_backscatter" in self.analyses:
            steps.append(SarBackscatterParams())
        if "temporal_ndwi" in self.analyses:
            steps.append(TemporalNdwiParams())
        return AgentPlan(steps=steps)


def _clarify(
    reason: ClarificationReason,
    message: str,
    *,
    options: tuple[str, ...] = (),
    analyses: list[str] | None = None,
    location: str | None = None,
    periods: list[TimeRange] | None = None,
) -> ClarificationRequiredError:
    return ClarificationRequiredError(
        AgentClarification(
            reason=reason,
            message=message,
            options=list(options),
            understood_analyses=[ANALYSIS_LABELS[key] for key in analyses or []],
            understood_location=location,
            understood_periods=list(periods or []),
        )
    )


#: Stands in for an intra-word apostrophe while offsets must stay fixed.
_APOSTROPHE = "\u02bc"


def _protect_apostrophes(question: str) -> str:
    """Swap intra-word apostrophes for a same-width stand-in.

    The polarity reader treats text between apostrophes as quoted, so
    "<city>'s ... don't" would hide everything between them. The stand-in
    keeps every offset, and the place is restored with its real apostrophe.
    """

    return re.sub(r"(?<=\w)['’](?=\w)", _APOSTROPHE, question)


def _for_polarity(text: str) -> str:
    """Contractions without apostrophes: the negation cues match "dont"."""

    return text.replace(_APOSTROPHE, "")


def _requested(text: str, pattern: re.Pattern[str]) -> bool:
    return bool(requested_matches(text, pattern))


def interpret(question: str, *, today: date | None = None) -> QueryInterpretation:
    """Map ``question`` to a :class:`QueryInterpretation`, or say what is missing.

    Raises :class:`ClarificationRequiredError` whenever the question does not state
    something the workflow needs, or asks for something it does not do.
    """

    today = today or datetime.now(UTC).date()
    original = " ".join(question.split())
    text = _protect_apostrophes(original)

    # -- dates: found first, so their words are never read as a place --------
    exprs = _inherit_years(text, _find_dates(text))
    date_spans = [(expr.start, expr.end) for expr in exprs]
    dated = _mask(text, date_spans)
    # Sensor phrases ("in the Sentinel-2 image") look like places; they are
    # hidden from the place search only - the analysis reading keeps them,
    # because "in Sentinel-1 imagery" is how some questions ask for radar.
    masked = _mask(text, sorted(date_spans + [m.span() for m in _SENSOR_PHRASE.finditer(text)]))
    previous = None
    while previous != masked:
        previous = masked
        masked = _DATE_FRAME.sub(lambda m: " " * (len(m.group(0)) - 1) + "|", masked)

    place = _find_place(masked)
    # The place's own words are not requests ("Forest Hill" is not NDVI).
    reading = _for_polarity(dated if place.span is None else (
        dated[: place.span[0]] + " " * (place.span[1] - place.span[0]) + dated[place.span[1] :]
    ))
    location = place.text.replace(_APOSTROPHE, "'") if place.text is not None else None
    place = _Place(text=location, deictic=place.deictic, span=place.span)

    # -- what is asked for ------------------------------------------------------
    mentions: list[tuple[int, str]] = []
    for key, pattern in _ANALYSIS_TERMS.items():
        for match in requested_matches(reading, pattern):
            mentions.append((match.start(), key))
    analyses: list[str] = list(dict.fromkeys(key for _, key in sorted(mentions)))
    measured = [key for key in analyses if key != "imagery"]
    comparison = _requested(reading, _COMPARISON)

    unsupported = next(
        (message for pattern, message in _UNSUPPORTED if _requested(reading, pattern)),
        None,
    )
    if unsupported is None and not measured and _requested(reading, _DETECTION):
        unsupported = _DETECTION_MESSAGE
    if unsupported is not None:
        raise _clarify(
            "analysis_unsupported", unsupported, options=SUPPORTED_OPTIONS,
            analyses=measured, location=place.text,
        )

    if _requested(reading, _VISUAL):
        raise _clarify(
            "requires_ai_model",
            "Describing what is visible in an image needs an AI model. Select one "
            "in the AI menu, or ask for a measurement instead - for example the "
            "water index (NDWI) or the vegetation index (NDVI) of a place in a "
            "month.",
            options=SUPPORTED_OPTIONS,
            analyses=measured,
            location=place.text,
        )

    # -- sensor contradictions --------------------------------------------------
    optical_asked = [key for key in measured if key in _OPTICAL_INDICES]
    if optical_asked and _SAR_SOURCE.search(original):
        raise _clarify(
            "conflicting_request",
            "NDVI, NDWI and NDBI are computed from Sentinel-2 optical imagery; "
            "from Sentinel-1 radar SatQuery measures backscatter. Which should "
            "be computed?",
            options=(*[ANALYSIS_LABELS[k] for k in optical_asked],
                     ANALYSIS_LABELS["sar_backscatter"]),
            analyses=measured,
            location=place.text,
        )
    if "sar_backscatter" in measured and _OPTICAL_SOURCE.search(original) and not optical_asked:
        raise _clarify(
            "conflicting_request",
            "SAR backscatter is measured from Sentinel-1 radar, not Sentinel-2 "
            "optical imagery. Which should be computed?",
            options=(ANALYSIS_LABELS["sar_backscatter"], *[
                ANALYSIS_LABELS[k] for k in _OPTICAL_INDICES
            ]),
            location=place.text,
        )

    # -- the analysis, resolved against the comparison request ------------------
    if comparison:
        others = [key for key in measured if key != "ndwi"]
        if not measured:
            raise _clarify(
                "analysis_missing",
                "What should be compared? Comparison between two periods is "
                "available for water (NDWI) - for example 'Compare water at "
                "<place> between January 2024 and January 2025'.",
                options=(ANALYSIS_LABELS["temporal_ndwi"],),
                location=place.text,
            )
        if others:
            raise _clarify(
                "analysis_unsupported",
                "Comparison between two periods is available for water (NDWI) "
                "only. "
                + ", ".join(ANALYSIS_LABELS[k] for k in others)
                + " can be measured for one period at a time.",
                options=(ANALYSIS_LABELS["temporal_ndwi"],
                         *[ANALYSIS_LABELS[k] for k in others]),
                analyses=measured,
                location=place.text,
            )
        chosen: list[str] = ["temporal_ndwi"]
    else:
        if not analyses:
            raise _clarify(
                "analysis_missing",
                "What would you like to analyse - vegetation (NDVI), water "
                "(NDWI), built-up area (NDBI) or SAR backscatter?",
                options=SUPPORTED_OPTIONS,
                location=place.text,
            )
        # Imagery rides along with every analysis; asked for alone, it is the
        # analysis.
        chosen = measured or ["imagery"]

    # -- where --------------------------------------------------------------------
    if place.text is None:
        message = (
            f"Which place should be analysed? The question refers to "
            f"'{place.deictic}' but does not name it. "
            if place.deictic
            else "Which place should be analysed? "
        ) + (
            "Name a city, district, landmark or 'lat, lon' after 'around', "
            "'in' or 'at' - for example 'around <city>' or 'at <landmark>, <city>'."
        )
        raise _clarify("location_missing", message, analyses=chosen)

    # -- when ---------------------------------------------------------------------
    periods = _periods(text, exprs, comparison=comparison, analyses=chosen, place=place.text)
    for window in periods:
        problem = observation_period_problem(window, today=today)
        if problem is not None:
            raise _clarify(
                "date_invalid",
                f"That period cannot be observed: {problem}.",
                analyses=chosen, location=place.text, periods=periods,
            )
    if comparison and _overlaps(*periods):
        raise _clarify(
            "date_invalid",
            "The two periods overlap, so they cannot be compared. Give two "
            "separate periods - for example January 2024 and January 2025.",
            analyses=chosen, location=place.text, periods=periods,
        )

    polarization: Literal["vv", "vh"] = (
        "vh" if _requested(reading, _VH_ONLY) and not _requested(reading, _VV) else "vv"
    )
    return QueryInterpretation(
        analyses=tuple(chosen),  # type: ignore[arg-type]
        location_query=place.text,
        windows=tuple(periods),
        comparison=comparison,
        sar_polarization=polarization,
    )


def _overlaps(first: TimeRange, second: TimeRange) -> bool:
    return first.start_date <= second.end_date and second.start_date <= first.end_date


def _periods(
    text: str,
    exprs: list[_DateExpr],
    *,
    comparison: bool,
    analyses: list[str],
    place: str,
) -> list[TimeRange]:
    """The one or two periods the question states, or a clarification."""

    def clarify(reason: ClarificationReason, message: str) -> ClarificationRequiredError:
        return _clarify(reason, message, analyses=analyses, location=place)

    example = (
        "for example 'between January 2024 and January 2025'"
        if comparison
        else "for example 'in January 2025' or 'on 15 January 2025'"
    )
    if _SEASON.search(text) or _QUALIFIED.search(text):
        raise clarify(
            "date_ambiguous",
            "Seasons and partial periods such as 'early 2024' are not converted "
            f"into dates. Give explicit months or days - {example}.",
        )
    if not exprs:
        relative = _RELATIVE.search(text)
        raise clarify(
            "date_ambiguous" if relative else "date_missing",
            (
                f"'{relative.group(0)}' is not resolved to a date automatically. "
                if relative
                else "For which date or period? "
            )
            + f"Give a month and year or a date - {example}.",
        )
    yearless = [expr.text for expr in exprs if expr.year is None]
    if yearless:
        raise clarify(
            "date_ambiguous",
            f"Which year is meant for '{yearless[0]}'? Give the year - {example}.",
        )
    if len(exprs) > 2:
        raise clarify(
            "date_ambiguous",
            "At most two periods can be analysed in one question: one period, "
            f"or two periods to compare - {example}.",
        )

    try:
        ranges = [_to_range(expr) for expr in exprs]
    except ValueError:
        raise clarify(
            "date_invalid", f"That date does not exist. Check the day and month - {example}."
        ) from None

    if comparison:
        if len(ranges) != 2:
            raise clarify(
                "comparison_incomplete",
                f"A comparison needs two periods; only one was given - {example}.",
            )
        return ranges

    if len(ranges) == 1:
        return ranges

    first, second = exprs
    link = _link(text, first, second)
    framed = _RANGE_FRAME.search(text[: first.start]) is not None
    if _RANGE_LINK.match(link) or (framed and _PAIR_LINK.match(link)):
        # "from January to March 2025" / "between January and March 2025": one
        # period spanning both.
        if ranges[1].end_date < ranges[0].start_date:
            raise clarify(
                "date_invalid",
                "The period ends before it starts. Give the earlier date first.",
            )
        return [TimeRange(start_date=ranges[0].start_date, end_date=ranges[1].end_date)]
    raise clarify(
        "date_ambiguous",
        f"Should {first.text} and {second.text} be compared, or treated as one "
        f"period? Say 'compare ... between {first.text} and {second.text}', or "
        f"'from {first.text} to {second.text}'.",
    )
