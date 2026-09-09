"""Honest reporting of a provider failure, and the bounded retry beneath it.

Two properties are under test, and they exist for the same reason: when the
language-model provider is degraded, the deterministic half of this system is
still perfectly good, and the response must say so accurately.

**The distinction that matters.** ``synthesis_unavailable`` means the tools ran,
the measurements are valid, and only the prose could not be generated. It is
NOT the same as a synthesizer reporting that the evidence does not answer the
question - that is a successful run whose ``answer`` is an abstention. Reading
the first as the second tells the user their measurements are worthless when
they are not, so the contract must let a caller tell them apart without
guessing.

**The retry.** Measured against the live endpoint (2026-09), the free tier
allows 5 requests per minute per model and returns 503/504 under load. A short
bounded retry recovers the overload case; a quota exhaustion is reported with
the delay the server itself named, because sleeping tens of seconds inside a
request would hang it rather than help.

No test here makes a live provider call. The Gemini classes are exercised
through the repository's established fake-client pattern.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from app.core.config import Settings
from app.core.errors import IntentParsingError, UpstreamServiceError
from app.services.agent.grounding import DraftAnswer
from app.services.agent.providers import gemini as gemini_mod
from app.services.agent.providers.gemini import (
    GeminiAgentPlanner,
    GeminiAnswerSynthesizer,
)
from app.services.agent.schemas import (
    AgentEvidence,
    AgentFailure,
    AgentPlan,
    AgentQuestionRequest,
    AgentResult,
    AgentToolStep,
    AgentTrace,
)
from app.services.agent.service import AgentService
from google.genai import errors as genai_errors
from google.genai._transformers import t_schema
from pydantic import ValidationError

from tests.test_agent_service import (  # reuse the established fakes verbatim
    RecordingExecutor,
    RecordingPlanner,
    RecordingSynthesizer,
    make_evidence,
    make_plan,
)

FAKE_KEY = "test-key-not-real"
QUESTION = "What is the NDWI of Chennai in January 2024?"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def quota_error(retry_delay: str | None = "36.36s") -> genai_errors.APIError:
    """A 429 shaped like the real one, RetryInfo included.

    The body mirrors what the live endpoint returned when the free-tier quota
    was exhausted, so the parser is tested against the real shape rather than
    one invented to suit it.
    """

    details: list[dict[str, Any]] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}],
        }
    ]
    if retry_delay is not None:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        )
    return genai_errors.APIError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded for metric: generate_content_free_tier",
                "details": details,
            }
        },
    )


class SequencedAioModels:
    """Raises a scripted sequence of failures, then returns ``text``.

    Counts its calls, which is how "retried exactly N times" is established as
    an observation rather than an assumption.
    """

    def __init__(self, *, errors: list[Exception], text: str | None = None) -> None:
        self.errors = list(errors)
        self.text = text
        self.calls = 0

    async def generate_content(self, **kwargs: Any) -> Any:
        # Same faithfulness step the other provider fakes perform: the real SDK
        # translates the response schema when building the request, so an
        # untranslatable schema must fail here too.
        config = kwargs.get("config")
        if config is not None and config.response_schema is not None:
            t_schema(None, config.response_schema)
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(text=self.text)


class SequencedClient:
    def __init__(self, models: SequencedAioModels) -> None:
        self.aio = SimpleNamespace(models=models)


VALID_PLAN_JSON = json.dumps(
    {
        "steps": [
            {
                "tool": "execute_query",
                "intent": {
                    "location_query": "Chennai",
                    "temporal_mode": "single",
                    "time_windows": [
                        {"start_date": "2024-01-01", "end_date": "2024-01-31"}
                    ],
                    "modalities": ["sentinel-2-optical"],
                    "task": "visualize",
                },
            }
        ]
    }
)


def planner_over(models: SequencedAioModels) -> GeminiAgentPlanner:
    return GeminiAgentPlanner(
        settings=Settings(gemini_api_key=FAKE_KEY), client=SequencedClient(models)
    )


def plan_now(planner: GeminiAgentPlanner) -> AgentPlan:
    return asyncio.run(planner.plan("anything"))


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff sleeps instead of serving them.

    The delays under test are real seconds. Sleeping them would make the suite
    slow for no added confidence, so the durations are captured and asserted
    on directly - which checks MORE than a wall-clock test would.
    """

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(gemini_mod.asyncio, "sleep", fake_sleep)
    return slept


# =========================================================================== #
# A. The retry policy - what is retried, and what is not
# =========================================================================== #


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_a_transient_upstream_failure_is_retried_and_can_succeed(status: int) -> None:
    """The overload case: a second attempt moments later often works."""

    models = SequencedAioModels(
        errors=[genai_errors.APIError(status, {})], text=VALID_PLAN_JSON
    )
    plan = plan_now(planner_over(models))

    assert models.calls == 2
    assert [step.tool for step in plan.steps] == ["execute_query"]


def test_a_transport_timeout_is_retried_and_can_succeed() -> None:
    models = SequencedAioModels(errors=[TimeoutError("read")], text=VALID_PLAN_JSON)
    plan_now(planner_over(models))

    assert models.calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_client_error_is_never_retried(status: int) -> None:
    """Repeating a request the server has already rejected only wastes quota."""

    models = SequencedAioModels(
        errors=[genai_errors.APIError(status, {})], text=VALID_PLAN_JSON
    )
    with pytest.raises(UpstreamServiceError):
        plan_now(planner_over(models))

    assert models.calls == 1


def test_an_unknown_failure_is_never_retried() -> None:
    """An uncharacterised failure may have a side effect; do not multiply it."""

    models = SequencedAioModels(errors=[RuntimeError("something odd")])
    with pytest.raises(UpstreamServiceError):
        plan_now(planner_over(models))

    assert models.calls == 1


def test_retrying_is_bounded_and_then_gives_up_honestly() -> None:
    """It must stop, and it must report the real failure when it does."""

    models = SequencedAioModels(
        errors=[genai_errors.APIError(503, {}) for _ in range(10)],
        text=VALID_PLAN_JSON,
    )
    with pytest.raises(UpstreamServiceError):
        plan_now(planner_over(models))

    assert models.calls == gemini_mod._MAX_ATTEMPTS


def test_the_total_backoff_never_exceeds_the_declared_budget(
    _no_real_sleeping: list[float],
) -> None:
    models = SequencedAioModels(
        errors=[genai_errors.APIError(503, {}) for _ in range(10)],
        text=VALID_PLAN_JSON,
    )
    with pytest.raises(UpstreamServiceError):
        plan_now(planner_over(models))

    assert sum(_no_real_sleeping) <= gemini_mod._MAX_RETRY_WAIT_SECONDS


def test_a_successful_first_attempt_never_sleeps(
    _no_real_sleeping: list[float],
) -> None:
    models = SequencedAioModels(errors=[], text=VALID_PLAN_JSON)
    plan_now(planner_over(models))

    assert models.calls == 1
    assert _no_real_sleeping == []


# =========================================================================== #
# B. Rate limiting is reported as itself, not as an outage
# =========================================================================== #


def test_a_rate_limit_is_reported_under_its_own_code() -> None:
    """A quota that clears in seconds is not the same as a service being down."""

    models = SequencedAioModels(errors=[quota_error()], text=VALID_PLAN_JSON)
    with pytest.raises(UpstreamServiceError) as caught:
        plan_now(planner_over(models))

    assert caught.value.code == "rate_limited"
    assert "rate limited" in caught.value.message.lower()


def test_a_rate_limit_carries_the_delay_the_server_asked_for() -> None:
    models = SequencedAioModels(errors=[quota_error("36.36s")], text=VALID_PLAN_JSON)
    with pytest.raises(UpstreamServiceError) as caught:
        plan_now(planner_over(models))

    assert caught.value.retry_after_seconds == pytest.approx(36.36)


def test_a_long_retry_delay_is_reported_rather_than_waited_out(
    _no_real_sleeping: list[float],
) -> None:
    """Holding the request open for 36 s would hang it, not help it."""

    models = SequencedAioModels(errors=[quota_error("36.36s")], text=VALID_PLAN_JSON)
    with pytest.raises(UpstreamServiceError):
        plan_now(planner_over(models))

    assert models.calls == 1, "must fail fast, not retry after a long delay"
    assert _no_real_sleeping == []


def test_a_short_retry_delay_is_honoured_and_retried(
    _no_real_sleeping: list[float],
) -> None:
    """Inside the budget, the server's own number is what gets waited."""

    models = SequencedAioModels(errors=[quota_error("2s")], text=VALID_PLAN_JSON)
    plan_now(planner_over(models))

    assert models.calls == 2
    assert _no_real_sleeping == [2.0]


def test_a_rate_limit_without_a_stated_delay_reports_none_not_zero() -> None:
    """Unknown must never be rendered as 'retry immediately'."""

    models = SequencedAioModels(
        errors=[quota_error(None) for _ in range(10)], text=VALID_PLAN_JSON
    )
    with pytest.raises(UpstreamServiceError) as caught:
        plan_now(planner_over(models))

    assert caught.value.retry_after_seconds is None


def test_no_upstream_error_text_ever_reaches_the_message() -> None:
    """Provider payloads do not reach responses; only the duration is taken.

    An upstream error body is third-party text that may echo the request, so
    the message is written from the status code alone.
    """

    error = quota_error()
    error.details["error"]["message"] = "SECRET-CANARY-abc123 leaked from upstream"
    models = SequencedAioModels(errors=[error], text=VALID_PLAN_JSON)

    with pytest.raises(UpstreamServiceError) as caught:
        plan_now(planner_over(models))

    assert "SECRET-CANARY" not in caught.value.message


@pytest.mark.parametrize(
    "details",
    [
        None,
        "not a dict",
        {"error": "not a dict"},
        {"error": {"details": "not a list"}},
        {"error": {"details": [{"@type": "RetryInfo", "retryDelay": "soon"}]}},
        {"error": {"details": [{"@type": "RetryInfo"}]}},
        {"error": {"details": ["not a dict"]}},
    ],
)
def test_a_malformed_error_body_yields_no_delay_rather_than_crashing(
    details: Any,
) -> None:
    """A surprising payload must stay a handled failure."""

    assert gemini_mod._retry_delay_seconds(genai_errors.APIError(429, details)) is None


# =========================================================================== #
# C. Every role shares one policy
# =========================================================================== #


def test_the_synthesizer_retries_on_the_same_terms_as_the_planner() -> None:
    """One policy, one implementation - three copies drifted apart before."""

    models = SequencedAioModels(
        errors=[genai_errors.APIError(503, {})],
        text=json.dumps({"summary": "The scene was retrieved.", "evidence_refs": []}),
    )
    synthesizer = GeminiAnswerSynthesizer(
        settings=Settings(gemini_api_key=FAKE_KEY), client=SequencedClient(models)
    )
    answer = asyncio.run(synthesizer.synthesize(QUESTION, AgentEvidence()))

    assert models.calls == 2
    assert answer.summary == "The scene was retrieved."


def test_the_synthesizer_reports_a_rate_limit_as_a_rate_limit() -> None:
    models = SequencedAioModels(errors=[quota_error()])
    synthesizer = GeminiAnswerSynthesizer(
        settings=Settings(gemini_api_key=FAKE_KEY), client=SequencedClient(models)
    )
    with pytest.raises(UpstreamServiceError) as caught:
        asyncio.run(synthesizer.synthesize(QUESTION, AgentEvidence()))

    assert caught.value.code == "rate_limited"


def test_unusable_model_output_is_still_not_retried() -> None:
    """A well-formed HTTP response carrying nonsense is not a transport blip."""

    models = SequencedAioModels(errors=[], text="not json at all")
    with pytest.raises(IntentParsingError):
        plan_now(planner_over(models))

    assert models.calls == 1


# =========================================================================== #
# D. The AgentFailure contract
# =========================================================================== #


def make_result(**overrides: Any) -> AgentResult:
    payload: dict[str, Any] = {
        "status": "planner_unavailable",
        "answer": None,
        "trace": AgentTrace(),
        "evidence": AgentEvidence(),
    }
    payload.update(overrides)
    return AgentResult.model_validate(payload)


def failure(**overrides: Any) -> AgentFailure:
    payload: dict[str, Any] = {
        "stage": "planning",
        "code": "rate_limited",
        "message": "The language-model service is rate limited.",
    }
    payload.update(overrides)
    return AgentFailure.model_validate(payload)


def test_a_failure_is_optional_so_existing_callers_stay_valid() -> None:
    assert make_result().failure is None


def test_a_failure_may_accompany_the_two_provider_failure_statuses() -> None:
    assert make_result(failure=failure()).failure is not None
    assert (
        make_result(
            status="synthesis_unavailable", failure=failure(stage="synthesis")
        ).failure
        is not None
    )


def test_a_delivered_answer_may_not_carry_a_failure() -> None:
    """Nothing failed, so claiming something did would misdescribe the run."""

    with pytest.raises(ValidationError):
        make_result(status="ok", answer="A scene was selected.", failure=failure())


def test_a_withheld_answer_may_not_carry_a_provider_failure() -> None:
    """The providers worked; validation is what rejected the answer.

    That outcome is reported through ``answer_validation``, and duplicating it
    here would offer two accounts of one event.
    """

    with pytest.raises(ValidationError):
        make_result(status="answer_withheld", failure=failure())


def test_the_stage_and_the_status_must_agree() -> None:
    with pytest.raises(ValidationError):
        make_result(status="planner_unavailable", failure=failure(stage="synthesis"))
    with pytest.raises(ValidationError):
        make_result(status="synthesis_unavailable", failure=failure(stage="planning"))


def test_a_failure_stage_outside_the_two_provider_stages_is_rejected() -> None:
    """Tool failures are reported per-step in the trace, never as a stage."""

    with pytest.raises(ValidationError):
        failure(stage="execution")


def test_a_negative_retry_delay_is_rejected() -> None:
    with pytest.raises(ValidationError):
        failure(retry_after_seconds=-1)


def test_an_unknown_retry_delay_is_none_not_zero() -> None:
    assert failure().retry_after_seconds is None


# =========================================================================== #
# E. The service reports which stage failed, and why
# =========================================================================== #


def build(*, planner: Any = None, synthesizer: Any = None) -> AgentService:
    return AgentService(
        planner=planner if planner is not None else RecordingPlanner(),
        executor=RecordingExecutor(),  # type: ignore[arg-type]
        synthesizer=synthesizer if synthesizer is not None else RecordingSynthesizer(),
    )


def ask(service: AgentService) -> AgentResult:
    return asyncio.run(service.answer(AgentQuestionRequest(question=QUESTION)))


def test_a_planner_failure_reports_the_planning_stage_and_its_code() -> None:
    error = UpstreamServiceError("Rate limited.", code="rate_limited")
    error.retry_after_seconds = 36.36  # type: ignore[attr-defined]
    result = ask(build(planner=RecordingPlanner(error=error)))

    assert result.status == "planner_unavailable"
    assert result.failure is not None
    assert result.failure.stage == "planning"
    assert result.failure.code == "rate_limited"
    assert result.failure.retry_after_seconds == pytest.approx(36.36)


def test_an_actionable_planner_message_survives_to_the_caller() -> None:
    """This was being logged and then discarded, leaving a bare status.

    A misconfigured deployment gets exactly one useful sentence, and it must
    not be the one thing the response drops.
    """

    result = ask(
        build(
            planner=RecordingPlanner(
                error=UpstreamServiceError(
                    "GEMINI_API_KEY is not configured; agent planning is "
                    "unavailable."
                )
            )
        )
    )

    assert result.failure is not None
    assert "GEMINI_API_KEY" in result.failure.message


def test_a_synthesis_failure_reports_the_synthesis_stage() -> None:
    result = ask(
        build(synthesizer=RecordingSynthesizer(error=UpstreamServiceError("down")))
    )

    assert result.status == "synthesis_unavailable"
    assert result.failure is not None
    assert result.failure.stage == "synthesis"


def test_a_synthesis_failure_keeps_the_evidence_and_says_prose_was_what_broke() -> None:
    """THE distinction. The measurements are valid; only the sentence is gone.

    Rendering this as "insufficient evidence" would tell the user their real,
    pixel-derived numbers answered nothing.
    """

    result = ask(
        build(
            synthesizer=RecordingSynthesizer(
                error=UpstreamServiceError("The language-model service is unavailable.")
            )
        )
    )

    assert result.status == "synthesis_unavailable"
    assert result.answer is None
    assert result.failure is not None
    assert result.evidence.items, "the deterministic evidence must survive"
    assert result.evidence.items[0].measurement is not None


def test_an_honest_abstention_is_a_success_carrying_no_failure() -> None:
    """The case a synthesis failure must never be confused with.

    Here the synthesizer ran perfectly and reported that the evidence does not
    answer the question. That is an answer, and the run succeeded.
    """

    result = ask(
        build(
            synthesizer=RecordingSynthesizer(
                answer=DraftAnswer(
                    summary="Insufficient evidence to answer the question.",
                    evidence_refs=[],
                )
            )
        )
    )

    assert result.status == "ok"
    assert result.answer == "Insufficient evidence to answer the question."
    assert result.failure is None


def test_the_two_cases_are_distinguishable_without_reading_the_prose() -> None:
    """A caller branches on 'failure is None', never on the answer's wording."""

    abstained = ask(
        build(
            synthesizer=RecordingSynthesizer(
                answer=DraftAnswer(
                    summary="Insufficient evidence to answer the question.",
                    evidence_refs=[],
                )
            )
        )
    )
    broke = ask(build(synthesizer=RecordingSynthesizer(error=UpstreamServiceError("x"))))

    assert (abstained.failure is None) != (broke.failure is None)
    assert abstained.status != broke.status


def test_a_planner_failure_still_fabricates_no_evidence() -> None:
    """The failure field explains the absence; it does not fill it."""

    result = ask(build(planner=RecordingPlanner(error=UpstreamServiceError("x"))))

    assert result.evidence.items == []
    assert result.trace.plan is None
    assert result.trace.answer_validation is None


def test_a_provider_that_names_no_delay_yields_no_delay() -> None:
    """``retry_after_seconds`` is an optional hint, not a required field."""

    result = ask(build(planner=RecordingPlanner(error=UpstreamServiceError("down"))))

    assert result.failure is not None
    assert result.failure.retry_after_seconds is None


def test_a_nonsense_retry_hint_is_ignored_rather_than_trusted() -> None:
    """The hint is read reflectively, so it must be type-checked on arrival."""

    error = UpstreamServiceError("down")
    error.retry_after_seconds = "soon"  # type: ignore[attr-defined]
    result = ask(build(planner=RecordingPlanner(error=error)))

    assert result.failure is not None
    assert result.failure.retry_after_seconds is None


def test_the_failure_reaches_the_serialized_api_body() -> None:
    """Subagent-facing contract: the field must survive model_dump."""

    body = ask(
        build(planner=RecordingPlanner(error=UpstreamServiceError("down")))
    ).model_dump(mode="json")

    assert body["failure"]["stage"] == "planning"
    assert body["failure"]["code"] == "upstream_error"
    assert body["failure"]["retry_after_seconds"] is None


def test_a_successful_answer_serializes_a_null_failure() -> None:
    """Backward compatible: the field is always present, null on success."""

    body = ask(build()).model_dump(mode="json")

    assert body["status"] == "ok"
    assert body["failure"] is None


def test_the_evidence_and_plan_are_unchanged_by_the_new_field() -> None:
    """Nothing else about a successful run moved."""

    result = ask(build())

    assert result.status == "ok"
    assert result.evidence == make_evidence()
    assert result.trace.plan == make_plan()
    assert [step.tool for step in result.trace.steps] == [
        "execute_query",
        "ndwi_statistics",
    ]


def test_a_step_level_tool_failure_is_not_a_stage_failure() -> None:
    """A tool that failed is reported in the trace, and the run still answers."""

    from app.services.agent.executor import ExecutionOutcome

    plan = make_plan()
    result = ask(AgentService(planner=RecordingPlanner(), synthesizer=RecordingSynthesizer(),
                             executor=RecordingExecutor(outcome=ExecutionOutcome(
        steps=[AgentToolStep(status="failed", parameters=plan.steps[0],
                             error_message="catalog unavailable")],
        evidence=make_evidence(),
    ))))
    assert result.status == "ok"
    assert result.failure is None
    assert result.trace.steps[0].error_message == "catalog unavailable"


def test_tool_failures_stay_in_the_trace_not_in_failure() -> None:
    """The two channels report different things and must not be conflated."""

    from app.services.agent.executor import ExecutionOutcome

    plan = make_plan()
    executor = RecordingExecutor(
        outcome=ExecutionOutcome(
            steps=[
                AgentToolStep(status="ok", parameters=plan.steps[0]),
                AgentToolStep(
                    status="failed",
                    parameters=plan.steps[1],
                    error_message="the band could not be read",
                ),
            ],
            evidence=make_evidence(),
        )
    )
    service = AgentService(
        planner=RecordingPlanner(),
        executor=executor,  # type: ignore[arg-type]
        synthesizer=RecordingSynthesizer(),
    )
    result = asyncio.run(service.answer(AgentQuestionRequest(question=QUESTION)))

    assert result.failure is None, "a tool failure is not a provider-stage failure"
    assert result.trace.steps[1].status == "failed"
    assert result.trace.steps[1].error_message == "the band could not be read"
