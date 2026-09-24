"""M5.6 - the local intent model and the rule that decides when to trust it.

A. The shipped artifact: loads, is small, matches its evaluation report, and
   its numpy inference reproduces scikit-learn exactly.
B. The shipped model on clear questions - one per operation.
C. Untrusted output: a prediction that is not one of the eight labels, or not
   a finite probability, never routes anything.
D. Untrusted artifact: every malformed or smuggled field refuses the model.
E. The decision rule, branch by branch, with controlled predictions.
F. Integration: the real model through the real AgentService and executor,
   with recording fakes in place of the catalog and the engines.
G. Provider independence: no key, no Ollama, no socket.

Hand-written fakes, as elsewhere in this suite. No test contacts a model
provider, Nominatim, STAC or imagery.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import gzip
import hashlib
import json
import math
import pathlib
import socket
import time
from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from app.api.routes import query as query_routes
from app.main import create_app
from app.services.agent.executor import AgentExecutor
from app.services.agent.intent_model import (
    DEFAULT_ARTIFACT,
    INTENT_LABELS,
    IntentModelError,
    IntentPrediction,
    load_intent_classifier,
    parse_artifact,
    read_artifact,
)
from app.services.agent.intent_router import decide_operation, route, route_operation
from app.services.agent.interpretation import (
    ClarificationRequiredError,
    classifier_text,
    read_question,
)
from app.services.agent.schemas import AgentQuestionRequest
from app.services.agent.service import AgentService
from app.services.agent.standard import StandardPlanner, StandardReport
from app.services.query.schemas import TemporalComparison, TimeRange
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.test_standard_workflow import EngineAnalysis, RecordingQueryExecution

BACKEND = pathlib.Path(__file__).resolve().parents[1]
REPORT = BACKEND / "data" / "intent" / "evaluation_v2.json"
DATASET = BACKEND / "data" / "intent" / "satquery_intents_v2.jsonl"
#: The shipped threshold never goes below this (scripts/train_intent_model.py).
THRESHOLD_FLOOR = 0.90
TODAY = date(2026, 9, 24)


@pytest.fixture(scope="module")
def model() -> Any:
    classifier = read_artifact(DEFAULT_ARTIFACT)
    assert classifier is not None
    return classifier


def artifact_document() -> dict:
    return json.loads(gzip.decompress(DEFAULT_ARTIFACT.read_bytes()))


# =========================================================================== #
# A. The shipped artifact
# =========================================================================== #


def test_the_shipped_artifact_loads_with_exactly_the_supported_labels(model: Any) -> None:
    assert sorted(model.labels) == sorted(INTENT_LABELS)
    assert len(INTENT_LABELS) == 8
    assert model.version == "satquery-intent-v2"
    assert 0.0 < model.threshold < 1.0


def test_the_artifact_is_small_and_is_data_not_code() -> None:
    raw = DEFAULT_ARTIFACT.read_bytes()
    assert len(raw) < 1_000_000  # ~192 KB today
    assert raw[:2] == b"\x1f\x8b"  # gzip...
    assert isinstance(json.loads(gzip.decompress(raw)), dict)  # ...of plain JSON


def test_the_evaluation_report_describes_this_exact_artifact() -> None:
    report = json.loads(REPORT.read_text())
    assert report["artifact"]["sha256"] == hashlib.sha256(DEFAULT_ARTIFACT.read_bytes()).hexdigest()
    assert report["threshold"] == artifact_document()["threshold"]
    dataset = artifact_document()["metadata"]["dataset"]
    assert dataset["sha256"] == hashlib.sha256(DATASET.read_bytes()).hexdigest()


def test_the_reported_quality_does_not_regress() -> None:
    """A retrained model must still clear these, on held-out data only."""

    report = json.loads(REPORT.read_text())
    assert report["test"]["macro_f1"] >= 0.85
    # What a user gets: no wrong operation EXECUTED by the full pipeline.
    assert report["system_on_test"]["full_pipeline_model_plus_rules"][
        "wrong_operation_executed"
    ] == 0
    # And no acted-on prediction was wrong at the shipped threshold.
    assert report["test_at_threshold"]["accepted_errors"] == 0
    assert set(report["challenges"]) == {"challenge_v2", "challenge_v1"}
    for name, challenge in report["challenges"].items():
        assert challenge["model"]["macro_f1"] >= 0.85, name
        assert challenge["system"]["full_pipeline_model_plus_rules"][
            "wrong_operation_executed"
        ] == 0, name
        assert challenge["at_threshold"]["accepted_errors"] == 0, name


def test_coverage_rose_without_loosening_the_threshold() -> None:
    """v2 against v1 on the SAME held-out set, through the same rules."""

    report = json.loads(REPORT.read_text())
    assert report["threshold"] >= THRESHOLD_FLOOR
    baseline = report["baseline"]
    assert baseline["model_version"] == "satquery-intent-v1"
    before = baseline["challenges"]["challenge_v2"]
    after = report["challenges"]["challenge_v2"]
    assert after["role"].startswith("held out")
    assert after["at_threshold"]["accepted"] > before["at_threshold"]["accepted"]
    assert after["at_threshold"]["accepted_errors"] == 0
    full = "full_pipeline_model_plus_rules"
    assert after["system"][full]["correct"] >= before["system"][full]["correct"]


def test_numpy_inference_reproduces_scikit_learn_exactly(model: Any) -> None:
    """The exported parameters, run through scikit-learn itself."""

    sklearn = pytest.importorskip("sklearn")
    del sklearn
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    document = artifact_document()
    blocks = []
    for name, analyzer in (("word", "word"), ("char", "char_wb")):
        spec = document["features"][name]
        vectorizer = TfidfVectorizer(
            analyzer=analyzer, lowercase=True, sublinear_tf=True,
            ngram_range=tuple(spec["ngram_range"]), vocabulary=spec["vocabulary"],
        )
        vectorizer.fit(["placeholder"])  # vocabulary is fixed; fit only sets state
        vectorizer.idf_ = np.asarray(spec["idf"])
        blocks.append(vectorizer)
    classifier = LogisticRegression()
    classifier.classes_ = np.asarray(document["labels"])
    classifier.coef_ = np.asarray(document["classifier"]["coef"])
    classifier.intercept_ = np.asarray(document["classifier"]["intercept"])

    texts = [
        classifier_text(json.loads(line)["text"])
        for line in DATASET.read_text().splitlines()[::7]
    ]
    features = hstack([block.transform(texts) for block in blocks]).tocsr()
    expected = classifier.predict_proba(features)
    actual = np.vstack([model.probabilities(text) for text in texts])
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_inference_is_fast(model: Any) -> None:
    text = classifier_text("Show vegetation around Marina Beach, Chennai in January 2025")
    start = time.perf_counter()
    for _ in range(200):
        model.predict(text)
    assert (time.perf_counter() - start) / 200 < 0.005  # < 5 ms; ~0.05 ms measured


def test_the_model_sees_placeholders_not_places() -> None:
    assert classifier_text(
        "Show water around Forest Hill in January 2025"
    ) == "Show water around PLACE in DATE"


# =========================================================================== #
# B. The shipped model on clear questions
# =========================================================================== #


@pytest.mark.parametrize(
    ("question", "label"),
    [
        ("Show vegetation around Marina Beach, Chennai in January 2025", "NDVI"),
        ("How healthy are the crops near Karnal in 2024", "NDVI"),
        ("Show water around Marina Beach, Chennai in January 2025", "NDWI"),
        ("How much water is in Powai Lake in February 2025?", "NDWI"),
        ("Show built-up area around Marina Beach, Chennai in January 2025", "NDBI"),
        ("Analyze SAR backscatter around Marina Beach, Chennai in January 2025",
         "SAR_BACKSCATTER"),
        ("Radar echoes over Jakarta in January 2025", "SAR_BACKSCATTER"),
        ("Compare water at Marina Beach, Chennai between January 2024 and January 2025",
         "TEMPORAL_NDWI"),
        ("Show the satellite image of Marina Beach, Chennai in January 2025", "TRUE_COLOR"),
        ("Natural colour snapshot of Kolleru lake in 2024", "TRUE_COLOR"),
        ("Analyze Chennai", "CLARIFICATION"),
        ("Tell me about Pune", "CLARIFICATION"),
        ("Count ships in Chennai harbor", "UNSUPPORTED"),
        ("Predict the rice yield near Thanjavur in 2025", "UNSUPPORTED"),
    ],
)
def test_the_shipped_model_names_each_operation(model: Any, question: str, label: str) -> None:
    prediction = model.predict(classifier_text(question))
    assert prediction.label == label
    assert prediction.confidence >= model.threshold


def test_the_model_adds_what_the_vocabulary_misses(model: Any) -> None:
    """ "Lush tea gardens" names no index; the rules alone ask what to analyse."""

    question = "How green are the tea gardens around Darjeeling in June 2024"
    with pytest.raises(ClarificationRequiredError):
        route_operation(question, None)
    chosen, comparison, decision = route_operation(question, model)
    assert (chosen, comparison) == (["ndvi"], False)
    assert decision.reason == "model_added_operation"


# =========================================================================== #
# C. Untrusted output
# =========================================================================== #


class FakeClassifier:
    """Returns a fixed prediction and records exactly what it was shown."""

    def __init__(self, label: Any, confidence: Any = 0.99, threshold: float = 0.9) -> None:
        self._label, self._confidence = label, confidence
        self.threshold = threshold
        self.seen: list[str] = []

    def predict(self, text: str) -> Any:
        self.seen.append(text)
        return IntentPrediction(label=self._label, confidence=self._confidence)


class RawClassifier(FakeClassifier):
    """Returns whatever object it is given - the prediction is NOT validated."""

    def __init__(self, raw: Any, threshold: float = 0.9) -> None:
        super().__init__("NDVI", threshold=threshold)
        self._raw = raw

    def predict(self, text: str) -> Any:
        self.seen.append(text)
        return self._raw


@pytest.mark.parametrize(
    "bad",
    [
        {"label": "DELETE_DATABASE", "confidence": 0.99},
        {"label": "https://evil.example/route", "confidence": 0.99},
        {"label": "rm -rf /", "confidence": 0.99},
        {"label": "NDVI", "confidence": 1.5},
        {"label": "NDVI", "confidence": float("nan")},
        {"label": "NDVI", "confidence": 0.99, "url": "https://evil.example"},
        {"confidence": 0.99},
        {"label": "NDVI"},
    ],
)
def test_a_prediction_outside_the_contract_is_refused(bad: dict) -> None:
    with pytest.raises(ValidationError):
        IntentPrediction.model_validate(bad)


@pytest.mark.parametrize(
    "raw",
    [
        SimpleNamespace(label="rm -rf /", confidence=0.99),
        {"label": "SHELL", "confidence": 0.99},
        {"label": "NDBI", "confidence": math.inf},
        {"label": "NDBI"},
        None,
        "NDBI",
        42,
    ],
)
def test_malformed_model_output_falls_back_to_the_rules(raw: Any) -> None:
    """A model that answers outside the contract is treated as absent."""

    reading = read_question("Show vegetation around Chennai in January 2025")
    decision = decide_operation(reading, RawClassifier(raw))  # type: ignore[arg-type]
    assert decision.source == "rules"
    assert decision.reason == "model_unavailable"
    chosen, comparison, _ = route_operation(
        "Show vegetation around Chennai in January 2025", RawClassifier(raw)  # type: ignore[arg-type]
    )
    assert (chosen, comparison) == (["ndvi"], False)


def test_a_classifier_that_raises_falls_back_to_the_rules() -> None:
    class Broken(FakeClassifier):
        def predict(self, text: str) -> Any:
            raise ValueError("broken")

    _, _, decision = route_operation("Show water around Pune in 2024", Broken("NDVI"))
    assert decision.reason == "model_unavailable"


# =========================================================================== #
# D. Untrusted artifact
# =========================================================================== #


def tampered(mutate) -> dict:
    document = copy.deepcopy(artifact_document())
    mutate(document)
    return document


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("format", "something-else"),
        lambda d: d.__setitem__("format_version", 2),
        lambda d: d["labels"].__setitem__(0, "RUN_SHELL"),
        lambda d: d["labels"].append("EXTRA"),
        lambda d: d.__setitem__("threshold", 0.0),
        lambda d: d.__setitem__("threshold", 1.5),
        lambda d: d.__setitem__("threshold", "0.9"),
        lambda d: d.__setitem__("model_version", "../../etc/passwd"),
        lambda d: d["classifier"]["coef"].pop(),
        lambda d: d["classifier"]["coef"][0].append(1.0),
        lambda d: d["classifier"]["intercept"].__setitem__(0, float("nan")),
        lambda d: d["features"]["word"]["idf"].__setitem__(0, -1.0),
        lambda d: d["features"]["char"]["vocabulary"].append(
            d["features"]["char"]["vocabulary"][0]
        ),
        lambda d: d["features"]["word"].__setitem__("analyzer", "char"),
        lambda d: d["features"]["word"].__setitem__("ngram_range", [1, 99]),
        lambda d: d.__setitem__("metadata", "not an object"),
    ],
)
def test_a_tampered_artifact_is_refused(mutate) -> None:
    with pytest.raises(IntentModelError):
        parse_artifact(tampered(mutate))


def test_an_unreadable_artifact_means_rules_only(tmp_path: pathlib.Path) -> None:
    """Not an outage: the loader returns None and the rules decide."""

    load_intent_classifier.cache_clear()
    try:
        garbage = tmp_path / "model.json.gz"
        garbage.write_bytes(b"\x80\x04\x95 not json - and never unpickled")
        assert load_intent_classifier(garbage) is None
        assert load_intent_classifier(tmp_path / "missing.json.gz") is None
    finally:
        load_intent_classifier.cache_clear()


# =========================================================================== #
# E. The decision rule
# =========================================================================== #


def operation(question: str, classifier: Any) -> tuple[list[str], bool, str]:
    chosen, comparison, decision = route_operation(question, classifier)
    return chosen, comparison, decision.reason


def asked(question: str, classifier: Any) -> str:
    with pytest.raises(ClarificationRequiredError) as raised:
        route_operation(question, classifier)
    return raised.value.clarification.reason


def test_a_low_confidence_prediction_is_never_acted_on() -> None:
    """Mutation target 6: the rules decide, whatever the model said."""

    # Rules see NDVI; a weak NDBI must not turn this into a disagreement.
    assert operation(
        "Show vegetation around Pune in 2024", FakeClassifier("NDBI", 0.5)
    ) == (["ndvi"], False, "low_confidence")
    # Rules see nothing; a weak NDVI must not execute - the rules ask.
    assert asked(
        "How green are the tea gardens around Darjeeling in June 2024",
        FakeClassifier("NDVI", 0.89),
    ) == "analysis_missing"


def test_the_threshold_is_the_artifacts_own(model: Any) -> None:
    assert model.threshold == artifact_document()["threshold"]
    assert model.threshold >= THRESHOLD_FLOOR


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("NDVI", ["ndvi"]),
        ("NDWI", ["ndwi"]),
        ("NDBI", ["ndbi"]),
        ("SAR_BACKSCATTER", ["sar_backscatter"]),
        ("TRUE_COLOR", ["imagery"]),
    ],
)
def test_a_confident_label_adds_the_operation_the_rules_missed(
    label: str, expected: list[str]
) -> None:
    """Mutation targets 1-3: each label maps to its own engine only."""

    chosen, comparison, reason = operation(
        "Please look into the fields near Anand for me in 2024", FakeClassifier(label)
    )
    assert (chosen, comparison, reason) == (expected, False, "model_added_operation")


def test_a_confident_temporal_label_compares_two_periods() -> None:
    """Mutation target 4: a comparison is never run as a single period.

    "evolve" is outside the rules' comparison words, so without the model the
    two dates are read as ONE period.
    """

    question = "How did the reservoir level at Idukki evolve from 2023 to 2024"
    interpretation, decision = route(question, FakeClassifier("TEMPORAL_NDWI"), today=TODAY)
    assert decision.reason == "model_added_operation"
    assert interpretation.comparison is True
    assert interpretation.analyses == ("temporal_ndwi",)
    assert interpretation.intent().time_windows == TemporalComparison(
        baseline=TimeRange(start_date=date(2023, 1, 1), end_date=date(2023, 12, 31)),
        target=TimeRange(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)),
    )


def test_agreement_keeps_every_analysis_the_rules_read() -> None:
    chosen, _, reason = operation(
        "Show NDVI and SAR backscatter around Pune in 2024", FakeClassifier("NDVI")
    )
    assert (chosen, reason) == (["ndvi", "sar_backscatter"], "model_agrees_with_rules")


def test_disagreement_with_an_explicit_analysis_is_asked_back() -> None:
    assert asked("Show NDVI around Pune in 2024", FakeClassifier("NDBI")) == "analysis_ambiguous"
    assert asked("Show NDVI around Pune in 2024", FakeClassifier("TRUE_COLOR")) == (
        "analysis_ambiguous"
    )
    assert asked("Show NDVI around Pune in 2024", FakeClassifier("CLARIFICATION")) == (
        "analysis_ambiguous"
    )
    assert asked(
        "Show NDVI around Pune in 2024", FakeClassifier("TEMPORAL_NDWI")
    ) == "analysis_ambiguous"


def test_unsupported_is_refused_even_when_the_rules_see_nothing() -> None:
    """Mutation target 5: an unsupported request never becomes an analysis."""

    assert asked(
        "Estimate the kappa coefficient around Pune in 2024", FakeClassifier("UNSUPPORTED")
    ) == "analysis_unsupported"


def test_unsupported_beside_an_explicit_analysis_is_asked_back() -> None:
    assert asked(
        "Show NDVI around Pune in 2024", FakeClassifier("UNSUPPORTED")
    ) == "analysis_ambiguous"


def test_a_clarification_label_lets_the_rules_ask_their_specific_question() -> None:
    assert asked("Analyze Chennai", FakeClassifier("CLARIFICATION")) == "analysis_missing"


def test_the_model_cannot_override_a_safety_guard() -> None:
    """Visual questions, unsupported requests and sensor contradictions are the
    rules' - a confident label cannot run past them."""

    assert asked(
        "Is there visible water at Marina Beach, Chennai in January 2025?",
        FakeClassifier("NDWI"),
    ) == "requires_ai_model"
    assert asked(
        "Count ships and show NDVI near Chennai port in 2024", FakeClassifier("NDVI")
    ) == "analysis_unsupported"
    assert asked(
        "Show vegetation using radar around Chennai in 2024", FakeClassifier("NDVI")
    ) == "conflicting_request"


def test_the_model_cannot_change_the_place_or_the_dates() -> None:
    question = "How green are the tea gardens around Darjeeling in June 2024"
    interpretation, _ = route(question, FakeClassifier("NDVI"), today=TODAY)
    assert interpretation.location_query == "Darjeeling"
    assert interpretation.windows == (
        TimeRange(start_date=date(2024, 6, 1), end_date=date(2024, 6, 30)),
    )


def test_a_missing_slot_is_still_asked_whoever_chose_the_operation() -> None:
    with pytest.raises(ClarificationRequiredError) as raised:
        route("How green are the tea gardens around Darjeeling", FakeClassifier("NDVI"),
              today=TODAY)
    assert raised.value.clarification.reason == "date_missing"


def test_the_classifier_is_shown_placeholders_only() -> None:
    fake = FakeClassifier("NDVI")
    route_operation("Show vegetation around Marina Beach, Chennai in January 2025", fake)
    assert fake.seen == ["Show vegetation around PLACE in DATE"]


# =========================================================================== #
# F. Integration - the real model, the real service, fake engines
# =========================================================================== #


def answer(question: str, classifier: Any) -> tuple[Any, Any, Any]:
    query, analysis = RecordingQueryExecution(), EngineAnalysis()
    service = AgentService(
        planner=StandardPlanner(classifier=classifier),
        executor=AgentExecutor(query_execution_service=query, analysis_service=analysis),
        synthesizer=StandardReport(),
    )
    result = asyncio.run(service.answer(AgentQuestionRequest(question=question)))
    return result, query, analysis


@pytest.mark.parametrize(
    ("question", "indices", "sar", "temporal", "modalities"),
    [
        ("Show vegetation around Marina Beach, Chennai in January 2025",
         ["ndvi"], False, False, ["sentinel-2-optical"]),
        ("Show water around Marina Beach, Chennai in January 2025",
         ["ndwi"], False, False, ["sentinel-2-optical"]),
        ("Show built-up area around Marina Beach, Chennai in January 2025",
         ["ndbi"], False, False, ["sentinel-2-optical"]),
        ("Analyze SAR backscatter around Marina Beach, Chennai in January 2025",
         [], True, False, ["sentinel-1-sar"]),
        ("Compare water at Marina Beach, Chennai between January 2024 and January 2025",
         [], False, True, ["sentinel-2-optical"]),
        ("How green are the tea gardens around Darjeeling in June 2024",
         ["ndvi"], False, False, ["sentinel-2-optical"]),
    ],
)
def test_the_model_routes_to_the_deterministic_engines(
    model: Any, question: str, indices: list[str], sar: bool, temporal: bool,
    modalities: list[str],
) -> None:
    """Mutation target 9: the engine runs, and the answer is its value."""

    result, query, analysis = answer(question, model)

    assert query.calls[0].intent.modalities == modalities
    [request] = analysis.calls
    assert request.indices == indices
    assert request.include_sar_backscatter is sar
    assert request.include_temporal_ndwi is temporal
    assert result.status == "ok"
    if indices == ["ndvi"]:
        assert "The mean NDVI was -0.06136 index." in (result.answer or "")
    if sar:
        assert "The mean VV was -5.444 dB." in (result.answer or "")


def test_the_model_routes_true_colour_to_imagery_only(model: Any) -> None:
    result, query, analysis = answer(
        "Show the satellite image of Marina Beach, Chennai in January 2025", model
    )
    assert query.calls[0].include_imagery is True
    assert analysis.calls == []
    assert result.status == "ok"
    assert result.answer and result.answer.startswith("Scene ")


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("Analyze Chennai", "analysis_missing"),
        ("Count ships in Chennai harbor", "analysis_unsupported"),
    ],
)
def test_the_model_routes_clarification_and_unsupported_to_no_execution(
    model: Any, question: str, reason: str
) -> None:
    result, query, analysis = answer(question, model)
    assert result.status == "needs_clarification"
    assert result.clarification is not None
    assert result.clarification.reason == reason
    assert query.calls == [] and analysis.calls == []


# =========================================================================== #
# G. Provider independence
# =========================================================================== #


def test_inference_needs_no_network(model: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation target 8: every socket connection fails, inference still works."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the intent model tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    prediction = model.predict(classifier_text("Show water around Pune in 2024"))
    assert prediction.label == "NDWI"
    chosen, _, _ = route_operation("Show water around Pune in 2024", model)
    assert chosen == ["ndwi"]


def test_the_intent_modules_import_no_provider_or_transport() -> None:
    root = BACKEND / "app" / "services" / "agent"
    for name in ("intent_model.py", "intent_router.py", "interpretation.py", "standard.py"):
        tree = ast.parse((root / name).read_text())
        modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names
        }
        for banned in ("httpx", "requests", "urllib", "socket", "subprocess", "providers",
                       "google", "anthropic", "openai", "sklearn", "pickle", "joblib"):
            assert not any(banned in module for module in modules), (name, banned)


def test_the_standard_route_uses_the_model_without_any_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []

    def forbid(name: str) -> Any:
        def boom(*args: Any, **kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(name)

        return boom

    monkeypatch.setattr(query_routes, "get_agent_providers", forbid("get_agent_providers"))
    monkeypatch.setattr(query_routes, "get_intent_parser", forbid("get_intent_parser"))
    monkeypatch.setattr(query_routes, "QueryExecutionService", RecordingQueryExecution)
    monkeypatch.setattr(query_routes, "AnalysisService", EngineAnalysis)
    # No key is configured (conftest) and Ollama is reported not running
    # (conftest's probe stub); the intent model needs neither.

    client = TestClient(create_app(), raise_server_exceptions=False)
    response = client.post(
        "/api/v1/query/agent",
        json={"question": "How green are the tea gardens around Darjeeling in June 2024"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert "The mean NDVI was" in body["answer"]
    assert touched == []


def test_readiness_names_the_loaded_model() -> None:
    body = TestClient(create_app()).get("/ready").json()
    detail = {c["name"]: c for c in body["capabilities"]}["interpretation"]["detail"]
    assert "satquery-intent-v2" in detail
    assert f"{read_artifact(DEFAULT_ARTIFACT).threshold:.2f}" in detail
    assert "No external AI provider" in detail
