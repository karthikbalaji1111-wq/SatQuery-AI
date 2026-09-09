"""Provider-neutral prompt text for the three AI roles.

Extracted from the Gemini provider **verbatim** so every provider issues the
same instructions. Two copies of a prompt drift apart; one copy cannot. Nothing
here imports an SDK and nothing here is provider-specific - a provider decides
how to *send* these strings, never what they say.

The wording is unchanged from the Gemini implementation already in production
use, which keeps that behaviour the reference for any new provider.
"""

from __future__ import annotations

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

def _system_instruction() -> str:
    """The planning instruction. Deliberately asks for a plan and nothing else.

    It requests no explanation, no justification and no account of how the plan
    was arrived at, because none of that has anywhere to go: ``AgentPlan`` has
    no field for it, and the trace shown to a user records decisions and
    outcomes only.

    It also describes only what a planner needs - the tools and the shape of a
    plan - not the repository's architecture.
    """

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
  default public source supplies provider terrain-corrected RTC VV imagery;
  do not request optical indices or rs_model_analysis for SAR. SatQuery does
  not perform SAR calibration or quantitative backscatter analysis.
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
- intent.modalities: a non-empty list from "sentinel-2-optical" and
  "sentinel-1-sar", with no duplicates. Default to ["sentinel-2-optical"] when
  the user names no sensor.
- intent.task: "visualize", "change_detection" or "object_identification".
- include_imagery: true when the user asks to see the imagery, and ALWAYS true
  when the plan uses a spectral, temporal NDWI or rs_model_analysis tool, so
  the workspace can display the same scene used for analysis. The visual tool
  observes that retrieved image. (The server enforces this either way.)
- max_cloud_cover: 0-100, only when the user asks for cloud-free imagery.

RULES

- Use only the tools listed above. Never invent a tool name.
- Emit only the fields described here; extra fields are rejected.
- Extract only what the user's question supports. Never invent a location, a
  date range, a scene, or a measurement.
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
  distinguish it from other measurements.
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

    lines = []
    for item in evidence.items:
        if item.measurement is not None:
            content = (
                f"{item.measurement.name} = {item.measurement.value} "
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
    "unless that is plainly visible."
)
