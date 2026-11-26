"""The Render free-plan configuration fits in its 512 MB.

The figures are measured, not assumed: ``scripts/benchmark_memory.py`` runs
the heaviest path (NDVI + NDWI + NDBI with SCL masking, both dates held, as a
comparison does) and its docstring records the results. If a limit in
render.yaml is raised, this test says whether the instance can still hold it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PLAN_MB = 512
#: Leave room for allocator fragmentation, GDAL and the HTTP stack under load.
HEADROOM = 0.85
IDLE_MB = 104
#: Peak resident memory one workflow adds, by quantitative window ceiling.
PER_WORKFLOW_MB = {1024: 132, 2048: 454}


def _render_env() -> dict[str, str]:
    text = (ROOT / "render.yaml").read_text()
    return dict(re.findall(r"- key: (\w+)\s*\n\s*value: \"?([^\"\n]+)\"?", text))


def test_render_limits_fit_the_free_plan() -> None:
    env = _render_env()
    workflows = int(env["SATQUERY_MAX_CONCURRENT_WORKFLOWS"])
    window = int(env["SATQUERY_IMAGERY_HARD_MAX_DIMENSION"])

    assert window in PER_WORKFLOW_MB, "measure this window size before using it"
    worst_case = IDLE_MB + workflows * PER_WORKFLOW_MB[window]
    assert worst_case <= PLAN_MB * HEADROOM, (
        f"{workflows} workflows at {window} px need about {worst_case} MB "
        f"on a {PLAN_MB} MB instance"
    )


def test_the_defaults_would_not_fit_the_free_plan() -> None:
    # Why render.yaml overrides them: four full-size runs need ~1.9 GB.
    from app.core.config import Settings

    defaults = Settings(_env_file=None)  # type: ignore[call-arg]
    worst_case = IDLE_MB + defaults.max_concurrent_workflows * PER_WORKFLOW_MB[
        defaults.imagery_hard_max_dimension
    ]
    assert worst_case > PLAN_MB
