"""Deciding the OPERATION: the local intent model, checked against the rules.

    question -> read_question()        slots + the rules' operation   (deterministic)
             -> IntentClassifier       one label + confidence          (local model)
             -> decide_operation()     which operation, or a question
             -> resolve()              guards, place, dates            (deterministic)
             -> QueryInterpretation    -> the unchanged M5.5 plan

The model adds recall: it recognises operations the fixed vocabularies miss
("how green are the tea gardens", "radar reflectivity"). The rules add
precision: an explicit "NDVI" or "compare" is not a guess. So the model is
allowed to decide only where the two cannot contradict each other, and the
system prefers a question to a wrong measurement.

The decision rule, in full:

1. **No usable prediction** (no artifact, a refused artifact, a prediction that
   fails validation) or **confidence below the model's own threshold** -> the
   rule-based interpreter decides alone. This is M5.5, unchanged.
2. **UNSUPPORTED** -> refused. If the rules read a supported analysis in the
   same words, the question is ambiguous and is asked back instead.
3. **CLARIFICATION** -> asked back. If the rules read an analysis, the model
   and the rules disagree, and it is still asked back - as an ambiguity.
4. **An analysis label** -> executed only when the rules found NO analysis (the
   model adds what the vocabulary missed) or found the SAME one (agreement; the
   rules' full set is kept, so "NDVI and SAR" still runs both). A different
   explicit analysis is a disagreement, asked back.
5. **TEMPORAL_NDWI** -> a comparison of water; a rules-read comparison with the
   water label is the same thing. An explicit other analysis beside it is a
   disagreement.

Whatever decided the operation, the rules' guards still apply (unsupported
requests, visual questions, sensor contradictions) and the rules alone supply
the place and the dates. The model cannot add a slot, remove a guard, or reach
anything but the closed operation list.

The threshold is not chosen here. It is the value the training script chose on
the validation split for THIS artifact, and it travels inside the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import ValidationError

from app.core.logging import get_logger
from app.services.agent.intent_model import IntentClassifier, IntentPrediction
from app.services.agent.interpretation import (
    ANALYSIS_LABELS,
    SUPPORTED_OPTIONS,
    ClarificationRequiredError,
    QueryInterpretation,
    QuestionReading,
    _clarify,
    read_question,
    resolve,
    resolve_operation,
)

logger = get_logger("agent.intent_router")

#: The model labels that name one single-period analysis.
_LABEL_ANALYSIS: dict[str, str] = {
    "NDVI": "ndvi",
    "NDWI": "ndwi",
    "NDBI": "ndbi",
    "SAR_BACKSCATTER": "sar_backscatter",
    "TRUE_COLOR": "imagery",
}

#: The refusal for a request the model recognises as unsupported when the rules
#: have no more specific message for it.
_UNSUPPORTED_MESSAGE = (
    "That request is outside what SatQuery measures. It computes vegetation "
    "(NDVI), water (NDWI), built-up area (NDBI), SAR backscatter, a two-period "
    "water comparison, or shows true-colour imagery - for a named place and "
    "period."
)

DecisionSource = Literal["intent_model", "rules"]
DecisionReason = Literal[
    "model_unavailable",
    "low_confidence",
    "model_added_operation",
    "model_agrees_with_rules",
    "model_unsupported",
    "model_clarification",
    "disagreement",
]


@dataclass(frozen=True)
class OperationDecision:
    """Who chose the operation, and why. Logged; never executed."""

    source: DecisionSource
    reason: DecisionReason
    prediction: IntentPrediction | None
    #: The override handed to the resolver, or ``None`` to let the rules decide.
    analyses: tuple[str, ...] | None = None
    comparison: bool | None = None


def _safe_predict(
    classifier: IntentClassifier | None, text: str
) -> IntentPrediction | None:
    """A validated prediction, or ``None`` - never an exception.

    A model that raises, or returns something outside the closed contract, is
    treated as absent: the rules then decide, exactly as without a model.
    """

    if classifier is None:
        return None
    try:
        prediction = classifier.predict(text)
        # Re-validated: the classifier is replaceable, the contract is not.
        return IntentPrediction.model_validate(
            prediction.model_dump() if isinstance(prediction, IntentPrediction)
            else prediction
        )
    except (ValidationError, ValueError, TypeError, KeyError, IndexError) as exc:
        logger.warning("Intent model output refused (%s)", type(exc).__name__)
        return None


def _ambiguous(
    reading: QuestionReading, rules: tuple[str, ...], model_label: str
) -> ClarificationRequiredError:
    rule_labels = [ANALYSIS_LABELS[key] for key in rules]
    model_side = (
        ANALYSIS_LABELS[_LABEL_ANALYSIS[model_label]]
        if model_label in _LABEL_ANALYSIS
        else ANALYSIS_LABELS["temporal_ndwi"]
        if model_label == "TEMPORAL_NDWI"
        else None
    )
    choices = list(dict.fromkeys([*rule_labels, *([model_side] if model_side else [])]))
    message = (
        "The question could mean "
        + " or ".join(choices)
        + ". Which should be computed?"
        if len(choices) > 1
        else f"The question mentions {choices[0]}, but does not clearly ask for it. "
        "Should it be computed?"
        if choices
        else "Which analysis should be computed?"
    )
    return _clarify(
        "analysis_ambiguous",
        message,
        options=tuple(choices) or SUPPORTED_OPTIONS,
        analyses=list(rules),
        location=reading.place.text,
    )


def decide_operation(
    reading: QuestionReading, classifier: IntentClassifier | None
) -> OperationDecision:
    """Apply the decision rule. Raises a clarification where the rule asks one."""

    prediction = _safe_predict(classifier, reading.classifier_text)
    if prediction is None:
        return OperationDecision("rules", "model_unavailable", None)
    assert classifier is not None
    if prediction.confidence < classifier.threshold:
        return OperationDecision("rules", "low_confidence", prediction)

    label = prediction.label
    rules = reading.measured
    rules_imagery = "imagery" in reading.analyses

    if label == "UNSUPPORTED":
        if rules and reading.unsupported is None:
            raise _ambiguous(reading, rules, label)
        if reading.unsupported is not None or reading.visual:
            # The rules have the specific refusal; let them state it.
            return OperationDecision("rules", "model_unsupported", prediction)
        raise _clarify(
            "analysis_unsupported", _UNSUPPORTED_MESSAGE, options=SUPPORTED_OPTIONS,
            location=reading.place.text,
        )

    if label == "CLARIFICATION":
        if rules or rules_imagery:
            raise _ambiguous(reading, rules, label)
        # The rules ask the specific question (analysis, or what to compare).
        return OperationDecision("rules", "model_clarification", prediction)

    if label == "TEMPORAL_NDWI":
        if any(key != "ndwi" for key in rules):
            raise _ambiguous(reading, rules, label)
        return OperationDecision(
            "intent_model",
            "model_agrees_with_rules" if rules else "model_added_operation",
            prediction,
            analyses=("ndwi",),
            comparison=True,
        )

    analysis = _LABEL_ANALYSIS[label]
    if analysis == "imagery":
        # Imagery alone. A measurement the rules read explicitly is not.
        if rules:
            raise _ambiguous(reading, rules, label)
        return OperationDecision(
            "intent_model",
            "model_agrees_with_rules" if rules_imagery else "model_added_operation",
            prediction,
            analyses=("imagery",),
            comparison=reading.comparison,
        )
    if not rules:
        return OperationDecision(
            "intent_model", "model_added_operation", prediction,
            analyses=(analysis,), comparison=reading.comparison,
        )
    if analysis in rules:
        return OperationDecision(
            "intent_model", "model_agrees_with_rules", prediction,
            analyses=rules, comparison=reading.comparison,
        )
    raise _ambiguous(reading, rules, label)


def route_operation(
    question: str, classifier: IntentClassifier | None
) -> tuple[list[str], bool, OperationDecision]:
    """The operation alone - guards applied, slots not examined.

    What the evaluation measures: whether the OPERATION chosen for a question
    is the one it asks for, separately from whether it named a place and date.
    """

    reading = read_question(question)
    decision = decide_operation(reading, classifier)
    chosen, comparison = resolve_operation(
        reading, analyses=decision.analyses, comparison=decision.comparison
    )
    return chosen, comparison, decision


def route(
    question: str,
    classifier: IntentClassifier | None,
    *,
    today: date | None = None,
) -> tuple[QueryInterpretation, OperationDecision]:
    """The full interpretation: the decided operation, then the rules' slots."""

    reading = read_question(question)
    decision = decide_operation(reading, classifier)
    interpretation = resolve(
        reading,
        analyses=decision.analyses,
        comparison=decision.comparison,
        today=today,
    )
    return interpretation, decision
