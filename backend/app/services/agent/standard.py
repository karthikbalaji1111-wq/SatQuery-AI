"""The standard, provider-independent natural-language workflow.

    question -> StandardPlanner   -> AgentPlan        (deterministic interpretation)
             -> AgentExecutor     -> evidence         (unchanged, real data)
             -> StandardReport    -> DraftAnswer      (fixed sentences from evidence)
             -> validate_answer                       (unchanged grounding)

The same three roles an AI provider fills, filled without one. Nothing here
calls a model, and nothing here computes: the planner maps words to the closed
tool set through :mod:`~app.services.agent.interpretation`, and the report
reads values the engines already produced into the sentence templates the
grounding validator was built around. The report is still validated like any
other answer - a report that marked its own homework would establish nothing.

Why this is the default and AI is not: a supported question - an index, a
backscatter measurement or a water comparison for a named place and period -
has exactly one correct plan, and producing it needs no language model. A model
adds latency, quota and a failure mode to a decision that is not in doubt. An
AI provider remains available for questions outside the vocabulary, but only
when a request names one.
"""

from __future__ import annotations

from app.services.agent.grounding import DraftAnswer
from app.services.agent.interpretation import interpret
from app.services.agent.planner import AgentPlanner
from app.services.agent.prompts import _display_value
from app.services.agent.schemas import AgentEvidence, AgentPlan, EvidenceItem
from app.services.agent.synthesizer import AnswerSynthesizer
from app.services.ai.ports import IntentParser
from app.services.query.schemas import SatQueryIntent

#: How a run by the standard workflow is attributed. Not a model: nothing was
#: generated, so there is no model to name.
STANDARD_INTERPRETER = "standard"

#: The one sentence grounding accepts when nothing answers the question.
ABSTENTION = "Insufficient evidence to answer the question."

#: Evidence id -> the sentence template grounding was designed to accept.
#: Ordered as a reader would expect: optical indices, then radar, then a
#: comparison. Only ids an engine produces appear here.
_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("ndvi.ndvi_mean", "The mean NDVI was {value} index."),
    ("ndwi.ndwi_mean", "The mean NDWI was {value} index."),
    ("ndbi.ndbi_mean", "The mean NDBI was {value} index."),
    ("sar_backscatter.vv_mean_db", "The mean VV was {value} dB."),
    ("sar_backscatter.vh_mean_db", "The mean VH was {value} dB."),
    ("sar_backscatter.vv_minus_vh_mean_db", "The VV minus VH difference was {value} dB."),
    ("temporal_ndwi.first.ndwi_mean", "The earlier mean NDWI was {value} index."),
    ("temporal_ndwi.second.ndwi_mean", "The later mean NDWI was {value} index."),
    (
        "temporal_ndwi.difference.mean_ndwi_difference",
        "The mean NDWI difference was {value} index.",
    ),
)


class StandardPlanner(AgentPlanner):
    """Proposes the plan a supported question implies, with no model.

    Raises :class:`~app.services.agent.interpretation.ClarificationRequiredError`
    rather than proposing a plan it is not sure of.
    """

    async def plan(self, question: str) -> AgentPlan:
        return interpret(question).plan()


class StandardIntentParser(IntentParser):
    """``/query/parse`` without a provider: the same interpretation, as an intent."""

    async def parse_intent(self, prompt: str) -> SatQueryIntent:
        return interpret(prompt).intent()


def _scene_sentences(item: EvidenceItem, evidence: AgentEvidence) -> list[str]:
    """Which scene a cited value was measured on, and when.

    Both sentences are metadata the grounding validator accepts only from the
    cited item's own observation, so the scene named is the one measured.
    """

    execution = evidence.execution
    analysis = evidence.analysis
    if item.source == "sar_backscatter" and analysis and analysis.sar_backscatter:
        sar = analysis.sar_backscatter
        acquired = sar.acquired_at.isoformat()[:10] if sar.acquired_at else None
        scene = sar.scene_id
    elif execution is not None:
        if item.source == "execution":
            windows = [
                w for w in execution.windows
                if item.id == f"execution.{w.modality}.{w.label}.scene_count"
            ]
        else:
            windows = [
                w for w in execution.windows
                if w.modality == "sentinel-2-optical" and w.selected_scene_id
            ][:1]
        selected = [s for w in windows for s in w.scenes if s.id == w.selected_scene_id]
        if not selected:
            return []
        scene = selected[0].id
        acquired = (selected[0].datetime or "")[:10] or None
    else:
        return []
    sentences = [f"Scene {scene} was selected."]
    if acquired:
        sentences.append(f"The scene was acquired on {acquired}.")
    return sentences


class StandardReport(AnswerSynthesizer):
    """States the measured values in fixed sentences. Generates nothing.

    Each value appears exactly as the evidence holds it, rounded for display
    only, in the sentence form the grounding validator checks. When a run
    measured nothing - no scene matched, an area was refused, a band failed -
    the answer is the validator's abstention, and the evidence panel carries
    the reason the executor recorded.
    """

    async def synthesize(self, question: str, evidence: AgentEvidence) -> DraftAnswer:
        del question  # the plan already encoded it; the evidence is the answer
        by_id = {item.id: item for item in evidence.items}
        sentences: list[str] = []
        refs: list[str] = []

        for evidence_id, template in _TEMPLATES:
            item = by_id.get(evidence_id)
            if item is None or item.measurement is None:
                continue
            sentences.append(template.format(value=_display_value(item.measurement.value)))
            refs.append(evidence_id)

        # Provenance for a single-scene measurement. A comparison names two
        # scenes in its own evidence and needs no more words here.
        anchor = next(
            (by_id[ref] for ref in refs if not ref.startswith("temporal_ndwi.")), None
        )
        if anchor is None and not refs:
            # Imagery alone: the selected scene IS the result.
            anchor = next(
                (
                    item for item in evidence.items
                    if item.id.endswith(".scene_count")
                    and item.measurement is not None
                    and item.measurement.value > 0
                ),
                None,
            )
            if anchor is not None:
                refs.append(anchor.id)
        if anchor is not None:
            sentences.extend(_scene_sentences(anchor, evidence))

        if not sentences:
            return DraftAnswer(summary=ABSTENTION, evidence_refs=[])
        return DraftAnswer(summary=" ".join(sentences), evidence_refs=refs)
