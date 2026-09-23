"""Live smoke test for the local Ollama provider. NOT part of CI.

Needs Ollama running with the model installed, and network access for the one
real Sentinel-2 scene it retrieves through SatQuery's own pipeline:

    cd backend
    uv run python scripts/local_model_smoke.py                  # LOCAL_AI_MODEL
    uv run python scripts/local_model_smoke.py --model qwen3-vl:2b

    TEST 1  text        - location and period from a natural-language question
    TEST 2  planning    - output validates against the AgentPlan contract
    TEST 3  image       - the real retrieved PNG, described by the model
    TEST 4  synthesis   - a concise answer from deterministic NDWI evidence
    TEST 5  grounding   - an unsupported number can never reach the user

Every model call prints its wall-clock latency; the provider's own log line
adds Ollama's load time and token counts. Exits non-zero if a check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.agent.executor import AgentExecutor  # noqa: E402
from app.services.agent.grounding import DraftAnswer, validate_answer  # noqa: E402
from app.services.agent.plan_completion import complete_plan  # noqa: E402
from app.services.agent.providers.local import (  # noqa: E402
    LocalAgentPlanner,
    LocalAnswerSynthesizer,
    LocalIntentParser,
    LocalVisualAnalyst,
    ProbeFailure,
    installed_models,
)
from app.services.agent.schemas import AgentPlan  # noqa: E402
from app.services.analysis import AnalysisService  # noqa: E402
from app.services.query import QueryExecutionService  # noqa: E402

QUESTION = "What is the NDWI of Marina Beach, Chennai in January 2025?"
IMAGE_QUESTION = "Describe the major visible land, water and vegetation patterns."
TEMPTING = "What percentage of Marina Beach is under water, and how many ships are visible?"
FIXED_PLAN = {
    "steps": [
        {
            "tool": "execute_query",
            "intent": {
                "location_query": "Marina Beach, Chennai",
                "temporal_mode": "single",
                "time_windows": [{"start_date": "2025-01-01", "end_date": "2025-01-31"}],
                "modalities": ["sentinel-2-optical"],
                "task": "visualize",
            },
            "include_imagery": True,
        },
        {"tool": "spectral_indices", "indices": ["ndwi"]},
    ]
}

failures: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"   {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


async def timed(label: str, coro):  # type: ignore[no-untyped-def]
    started = time.perf_counter()
    result = await coro
    print(f"   [{label}: {time.perf_counter() - started:.1f}s]")
    return result


def loaded_models(base_url: str) -> str:
    """What Ollama has resident right now, from GET /api/ps."""

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/ps", timeout=3) as r:
            listed = json.load(r).get("models", [])
    except OSError as exc:
        return f"unavailable ({type(exc).__name__})"
    return (
        ", ".join(
            f"{m.get('name')} {m.get('size', 0) / 1e9:.2f} GB resident "
            f"({m.get('size_vram', 0) / 1e9:.2f} GB on GPU)"
            for m in listed
        )
        or "none loaded"
    )


async def main(model: str | None) -> int:
    settings = get_settings()
    chosen = model or settings.local_ai_model
    print(f"Ollama at {settings.local_ai_base_url}, model {chosen}")
    installed = await installed_models(settings)
    if installed is None:
        print("Ollama is not running. Start it (open the Ollama app, or `ollama serve`).")
        return 2
    if installed is ProbeFailure.MODELS_UNREADABLE:
        print("Ollama is running but cannot read its models. Is the drive holding them connected?")
        return 2
    if chosen not in installed:
        print(f"{chosen} is not installed. Run: ollama pull {chosen}")
        return 2

    print("\nTEST 1 - text: location and period")
    intent = await timed(
        "intent, cold", LocalIntentParser(settings=settings, model=chosen).parse_intent(QUESTION)
    )
    print(f"   location={intent.location_query!r} mode={intent.temporal_mode}")
    print(f"   windows={intent.time_windows}")
    check("marina" in intent.location_query.lower(), "location names Marina Beach")
    first = intent.time_windows[0] if isinstance(intent.time_windows, list) else None
    check(
        first is not None
        and first.start_date.isoformat() == "2025-01-01"
        and first.end_date.isoformat() == "2025-01-31",
        "period is January 2025",
    )
    print(f"   memory: {loaded_models(settings.local_ai_base_url)}")

    print("\nTEST 2 - planning against the AgentPlan contract")
    try:
        plan = await timed(
            "plan, warm", LocalAgentPlanner(settings=settings, model=chosen).plan(QUESTION)
        )
    except Exception as exc:  # report and continue: TESTS 3-5 build their own evidence
        check(False, f"plan validates against AgentPlan ({exc})")
    else:
        print(f"   planner chose: {[s.tool for s in plan.steps]}")
        check(isinstance(plan, AgentPlan), "plan validates against AgentPlan")
        completed = complete_plan(QUESTION, plan)
        print(f"   executed plan:  {[s.tool for s in completed.steps]}")
        check(
            any(s.tool in {"spectral_indices", "ndwi_statistics"} for s in completed.steps),
            "NDWI is computed",
        )

    print("\nRetrieving one real Sentinel-2 scene and its deterministic NDWI...")
    executed = await AgentExecutor(
        query_execution_service=QueryExecutionService(), analysis_service=AnalysisService()
    ).execute(AgentPlan.model_validate(FIXED_PLAN))
    evidence = executed.evidence
    imagery = evidence.execution.windows[0].imagery if evidence.execution else None
    check(imagery is not None, "real imagery retrieved")
    if imagery is None:
        return 1
    png = base64.b64decode(imagery.image_base64)
    print(f"   scene {imagery.scene_id}: {imagery.width}x{imagery.height} px, {len(png)} bytes")

    print("\nTEST 3 - the real image")
    seen = await timed(
        "image",
        LocalVisualAnalyst(settings=settings, model=chosen).observe(
            question=IMAGE_QUESTION, image=png, media_type=imagery.media_type
        ),
    )
    print(f"   model saw: {seen.answer}")
    check(bool(seen.answer.strip()), "the model described the image")
    print(f"   memory: {loaded_models(settings.local_ai_base_url)}")

    print("\nTEST 4 - synthesis from deterministic evidence")
    draft = await timed(
        "synthesis",
        LocalAnswerSynthesizer(settings=settings, model=chosen).synthesize(QUESTION, evidence),
    )
    validation = validate_answer(draft, evidence)
    print(f"   answer: {draft.summary!r}  refs={draft.evidence_refs}")
    print(f"   grounding: {validation.model_dump()}")
    check(validation.numeric_grounding == "pass", "every number in the answer is grounded")

    print("\nTEST 5 - an unsupported number cannot reach the user")
    tempted = await timed(
        "synthesis, tempting question",
        LocalAnswerSynthesizer(settings=settings, model=chosen).synthesize(TEMPTING, evidence),
    )
    verdict = validate_answer(tempted, evidence)
    shown = (
        verdict.numeric_grounding == "pass"
        and verdict.evidence_refs == "pass"
        and verdict.forbidden_terms == "pass"
    )
    print(f"   model wrote: {tempted.summary!r}")
    outcome = "SHOWN (every number traced to evidence)" if shown else "WITHHELD by grounding"
    print(f"   would be {outcome}")
    invented = DraftAnswer(
        summary="The mean NDWI was 0.91 index.", evidence_refs=["ndwi.ndwi_mean"]
    )
    check(
        validate_answer(invented, evidence).numeric_grounding == "fail",
        "an invented NDWI value is refused",
    )

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="an installed Ollama model; default LOCAL_AI_MODEL")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="   %(message)s")
    logging.getLogger("satquery.agent.local").setLevel(logging.INFO)
    try:
        sys.exit(asyncio.run(main(args.model)))
    except Exception as exc:  # report, never a bare traceback, in a smoke run
        print(f"\nSMOKE TEST ABORTED: {type(exc).__name__}: {exc}")
        sys.exit(1)
