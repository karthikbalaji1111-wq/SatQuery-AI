"""Provider-neutral prompt text for the three AI roles.

Extracted from the Gemini provider **verbatim** so every provider issues the
same instructions. Two copies of a prompt drift apart; one copy cannot. Nothing
here imports an SDK and nothing here is provider-specific - a provider decides
how to *send* these strings, never what they say.

The wording is unchanged from the Gemini implementation already in production
use, which keeps that behaviour the reference for any new provider.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

from app.services.agent.registry import TOOL_REGISTRY
from app.services.agent.schemas import AgentEvidence


def _tool_catalogue() -> str:
    """Describe the permitted tools, straight from the allowlist.

    Generated from :data:`TOOL_REGISTRY` rather than written out by hand, so the
    model can never be told about a capability the executor would refuse - or
    left ignorant of one it would accept.
    """

    return "\n".join(
        f"- {spec.name}: {spec.description}" for spec in TOOL_REGISTRY.values()
    )

#: How far back the product looks when the user names no date at all.
#: A PRODUCT DEFAULT, not a user intent - stated here once so every provider is
#: given the same rule and cannot drift from one another. Thirty days is chosen
#: to be long enough that Sentinel-2's five-day revisit yields several
#: candidates even through cloud, and short enough that "recent" stays honest.
DEFAULT_LOOKBACK_DAYS = 30


def default_time_window() -> tuple[date, date]:
    """``(today, today - DEFAULT_LOOKBACK_DAYS)`` in UTC.

    One function so the policy has a single definition. It used to be computed
    inline in the instruction string, which meant the only statement of the
    product's date default lived inside a prompt - unreachable to a test, and
    impossible to assert was the same for both providers.

    Returns the pair in the order the instruction renders them.
    """

    today = datetime.now(UTC).date()
    return today, today - timedelta(days=DEFAULT_LOOKBACK_DAYS)


def _system_instruction() -> str:
    """The planning instruction. Deliberately asks for a plan and nothing else.

    It requests no explanation, no justification and no account of how the plan
    was arrived at, because none of that has anywhere to go: ``AgentPlan`` has
    no field for it, and the trace shown to a user records decisions and
    outcomes only.

    It also describes only what a planner needs - the tools and the shape of a
    plan - not the repository's architecture.
    """

    today, default_start = default_time_window()
    return f"""\
You select which remote-sensing analyses to run for a user's question, and
return ONLY a JSON object matching the provided response schema. No prose, no
explanation, no commentary.

AVAILABLE TOOLS

{_tool_catalogue()}

PLAN RULES

- A plan has 1 to 3 steps.
- The first step MUST be "execute_query"; it appears exactly once. The analysis
  tools interpret its result, so nothing can run before it.
- Do not repeat a tool.
- Choose an analysis tool only when the question calls for it. A request merely
  to find or view imagery needs "execute_query" alone.
- "temporal_ndwi_statistics" compares two Sentinel-2 acquisitions, so use it
  only with a "compare" temporal mode carrying a baseline and a target window.
- "ndwi_statistics" is optical-only; it needs "sentinel-2-optical" among the
  modalities.
- For Sentinel-1 imagery use execute_query with include_imagery=true. The
  public source supplies provider terrain-corrected RTC VV/VH imagery.
  Use "sar_backscatter_statistics" for SAR analysis, VV, VH or backscatter
  questions with "sentinel-1-sar". Do not request optical indices or
  rs_model_analysis for SAR. Calibration is supplied by the provider.
  Set sar_polarization to "vh" when VH display is requested; otherwise "vv".
- "spectral_indices" is the tool for any question about vegetation, water or
  built-up/urban extent. Choose the indices that answer the question: "ndvi"
  for vegetation, "ndwi" for water, "ndbi" for built-up or bare ground. Ask for
  several in ONE step when the question spans several - never plan the tool
  twice. It is optical-only and needs "sentinel-2-optical".

EXECUTE_QUERY PARAMETERS

- intent.location_query: the place named by the user, verbatim. NEVER invent or
  output coordinates.
- intent.temporal_mode: "single" (one window), "compare" (a baseline and a
  target window), or "timeseries" (three or more windows).
- intent.time_windows: for "single" a list of exactly one
  {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}; for "timeseries" a
  list of two or more; for "compare" an object with "baseline" and "target".
  Resolve unambiguous relative expressions into explicit ISO ranges.
  Today is {today.isoformat()} UTC. If no dates are supplied, use the bounded
  default window {default_start.isoformat()} through {today.isoformat()}; this
  is a product default, not a user-specified date, and the effective window
  is displayed with the result. Never apply this default over explicit dates.
- intent.modalities: a non-empty list from "sentinel-2-optical" and
  "sentinel-1-sar", with no duplicates. Default to ["sentinel-2-optical"] when
  the user names no sensor.
- intent.task: "visualize" for viewing imagery or for measuring an index or
  SAR backscatter - this covers every analysis tool above. "change_detection"
  only with a "compare" temporal mode. "object_identification" only when the
  user asks to identify, detect, count or locate specific objects; it is not
  implemented and is reported as such, so never choose it for a measurement.
- include_imagery: true when the user asks to see the imagery, and ALWAYS true
  when the plan uses a spectral, temporal NDWI or rs_model_analysis tool, so
  the workspace can display the same scene used for analysis. The visual tool
  observes that retrieved image. (The server enforces this either way.)
- max_cloud_cover: 0-100, only when the user asks for cloud-free imagery.

RULES

- Use only the tools listed above. Never invent a tool name.
- Emit only the fields described here; extra fields are rejected.
- Extract only what the user's question supports. Never invent a location, a
  date range beyond the explicit bounded default above, a scene, or a measurement.
- You are planning only. You do not run the tools and you never report results.
"""

_SYNTHESIS_INSTRUCTION = """\
You write one short, factual answer to the user's question using ONLY the
evidence supplied below, and return it as JSON matching the response schema.
No prose outside the schema, no commentary.

RULES

- Use only the supplied evidence. Never introduce an observation, a place, a
  date, a scene or a measurement that does not appear in it.
- State no number that is not present in the evidence. Quote a value as given,
  or round it; never estimate, extrapolate or infer one.
- Cite the evidence you used in "evidence_refs", using the exact ids shown.
  Cite nothing by returning an empty list - never omit the field.
- Use one measurement per sentence, naming its metric, statistic and unit:
  "The mean NDWI was <value> index." Substitute only the cited measurement's
  actual identity, value and unit; preserve any scene/date context needed to
  distinguish it from other measurements. For SAR use "The mean VV was <value> dB."
  or "The VV minus VH difference was <value> dB." Name VV or VH explicitly;
  never substitute one polarization or a spectral index for another.
- For a comparison of two acquisitions, write exactly these sentences, adding
  no other words: "The earlier mean NDWI was <value> index.", "The later mean
  NDWI was <value> index." and "The mean NDWI difference was <value> index."
  Cite each value by its full id exactly as listed (for example
  "temporal_ndwi.first.ndwi_mean"), never a shortened prefix. Never put two
  values in one sentence, and never state a bare "mean NDWI" when two exist.
- Copy qualitative observations and warning sentences verbatim from cited
  evidence, including uncertainty. Avoid added conclusions or paraphrases.
  Prefer direct statements without conversational introductions.
- When the evidence carries limitations or warnings, say so plainly rather than
  presenting a result as more certain than it is.
- Describe what was measured. Do not claim detection, classification,
  co-registration, alignment, or comparison between individual pixels.
- If the evidence does not answer the question, use exactly:
  "Insufficient evidence to answer the question."
- Return only "summary" and "evidence_refs". No other field is accepted.
"""

#: Significant digits a measurement is SHOWN to the synthesiser with.
_DISPLAY_SIGNIFICANT_DIGITS = 4


def _display_value(value: float) -> str:
    """A measurement as a reader should see it: rounded, never estimated.

    DISPLAY ONLY. The evidence item keeps the full-precision value, and so do
    the API response and the evidence export; only the line the synthesiser
    reads is rounded. Observed live: rendering the raw float made the answer
    read "The mean NDWI was 0.1463908465206975 index." - sixteen digits of
    binary-float noise presented to a user as though they were measured.

    Always groundable. Grounding decides agreement by rounding the evidence
    value to the number of decimals the answer states, so a value shown here at
    fewer decimals is, by construction, a value that check accepts. Integral
    values (counts, pixel totals) are shown as integers.
    """

    if not math.isfinite(value):
        return str(value)
    if value == int(value):
        return str(int(value))
    magnitude = math.floor(math.log10(abs(value)))
    places = _DISPLAY_SIGNIFICANT_DIGITS - 1 - magnitude
    if places <= 0:
        return str(int(round(value)))
    return f"{round(value, places):.{places}f}".rstrip("0").rstrip(".")


def _render_context(evidence: AgentEvidence) -> list[str]:
    """What was queried and which scene each number was measured on.

    Observed live, with both providers and identical evidence: the synthesiser
    answered a direct "What is the NDWI of <place> in <month>?" correctly only
    about half the time, and otherwise replied "Insufficient evidence" - once
    while citing every relevant id. The citable lines named a metric and a
    value but not WHERE or WHEN, so a careful model could not tell that they
    answered a question about a particular place and month. With this block the
    same evidence was answered and grounded 6/6.

    Every fact here comes from ``evidence.execution`` and is one grounding
    already accepts when repeated: the query's own location text (masked in a
    measurement sentence), the requested window's ISO dates, the selected scene
    id and its ISO acquisition date. Nothing is citable - there is no id to
    cite - so this block can explain the evidence but never extend it.
    """

    execution = evidence.execution
    if execution is None:
        return []
    lines = [
        "CONTEXT (what was queried and which scene was measured; not citable)",
        f"- location queried: {execution.plan.intent.location_query}",
    ]
    for window in execution.windows:
        span = (
            f"{window.time_range.start_date.isoformat()} to "
            f"{window.time_range.end_date.isoformat()}"
        )
        chosen = next(
            (scene for scene in window.scenes if scene.id == window.selected_scene_id),
            None,
        )
        if chosen is None:
            lines.append(f"- {window.modality} window {window.label}: {span}; no scene selected")
            continue
        acquired = (chosen.datetime or "")[:10] or "unknown"
        lines.append(
            f"- {window.modality} window {window.label}: {span}; "
            f"measured scene {chosen.id}, acquired {acquired}"
        )
    lines.append("")
    return lines


def _render_evidence(evidence: AgentEvidence) -> str:
    """Present the evidence as citable lines: ``id | source | content``.

    Only the flattened, citable ``items`` are shown - each with the id the
    answer must cite. The full execution and analysis results are deliberately
    not dumped in: everything a sentence may legitimately quote is already an
    item, and a smaller prompt is a smaller surface for the model to wander off
    into.
    """

    if not evidence.items:
        return "(no evidence was collected)"

    lines = [*_render_context(evidence), "CITABLE ITEMS"]
    for item in evidence.items:
        if item.measurement is not None:
            content = (
                f"{item.measurement.name} = "
                f"{_display_value(item.measurement.value)} "
                f"{item.measurement.unit}"
            )
        elif item.visual is not None:
            # A model observation, labelled as one in the prompt itself so the
            # synthesiser can use it without mistaking it for a measurement.
            # The synthesiser NEVER sees the image - it reports what the
            # observer said, which is why the witness/narrator split holds.
            content = (
                f'observed by {item.visual.model} in scene '
                f'{item.visual.scene_id}: "{item.visual.statement}"'
            )
        else:
            content = item.text or ""
        lines.append(f"- {item.id} | {item.source} | {content}")
    return "\n".join(lines)

_VISUAL_INSTRUCTION = (
    "You are looking at one Sentinel-2 true-colour satellite image.\n\n"
    "Answer the question from WHAT YOU CAN SEE in this image, in one or two "
    "plain sentences.\n\n"
    "- Describe only what is visible. If the image does not show enough to "
    "answer, say so plainly.\n"
    "- Do NOT state precise quantities, areas, percentages or counts as if "
    "measured. If you estimate, say clearly that you are estimating by eye.\n"
    "- Do NOT claim to have run any index, classifier or detector. You are "
    "looking at a picture.\n"
    "- Do not describe the image as evidence of flooding, disaster or damage "
    "unless that is plainly visible.\n"
    "- Write for someone with no remote-sensing background: plain everyday "
    "words, no technical terms.\n"
    "- You are not told where or when the image was taken. Do not name or "
    "guess the place; describe only what the picture shows."
)
