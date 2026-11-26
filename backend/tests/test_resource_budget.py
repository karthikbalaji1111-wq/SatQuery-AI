"""The workflow time budget and the raster-read gate are ENFORCED.

Both were configured, documented in DEPLOYMENT.md and described in
``core/limits.py`` as working, and nothing ever applied them: the budget was
never read and the asyncio ``RASTER_GATE`` was never entered. These tests pin
that each control now changes behaviour.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from affine import Affine
from app.api.routes.query import get_ai_service
from app.core import limits as limits_mod
from app.core.config import Settings
from app.core.errors import ServiceOverloadedError, WorkflowTimeoutError
from app.core.limits import raster_read_slot, within_budget
from app.main import create_app
from app.services.satellite.raster import BandWindow
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_raster_semaphores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets its own process-wide gate, so none can leak a held slot."""

    monkeypatch.setattr(limits_mod, "_RASTER_SEMAPHORES", {})


# --------------------------------------------------------------------------- #
# Raster-read gate
# --------------------------------------------------------------------------- #


def test_the_raster_gate_caps_simultaneous_reads() -> None:
    settings = Settings(max_concurrent_raster_reads=2, raster_admission_wait_seconds=5)
    active = 0
    peak = 0
    lock = threading.Lock()

    def read() -> None:
        nonlocal active, peak
        with raster_read_slot(settings):
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    threads = [threading.Thread(target=read) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 2


def test_a_read_that_cannot_get_a_slot_in_time_is_refused() -> None:
    settings = Settings(max_concurrent_raster_reads=1, raster_admission_wait_seconds=0.05)
    held = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with raster_read_slot(settings):
            held.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    held.wait(5)
    try:
        with pytest.raises(ServiceOverloadedError), raster_read_slot(settings):
            pytest.fail("a second read must not run while the only slot is held")
    finally:
        release.set()
        thread.join()

    # The slot is released afterwards: the next read proceeds.
    with raster_read_slot(settings):
        pass


def _band() -> BandWindow:
    values = np.ones((1, 1), dtype=np.float32)
    return BandWindow(
        values=values,
        valid=values > 0,
        width=1,
        height=1,
        crs="EPSG:32644",
        transform=Affine(10, 0, 0, 0, -10, 0),
        resolution=10.0,
        nodata=-32768.0,
        window={"col_off": 0, "row_off": 0, "width": 1, "height": 1},
        source_shape=[1, 1],
    )


def test_imagery_service_reads_bands_through_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.satellite.imagery as imagery_mod
    from app.services.satellite.imagery import ImageryService
    from app.services.satellite.rtc import RTC_COLLECTION

    monkeypatch.setattr(imagery_mod, "sign_rtc_asset", lambda href, **kw: href)
    settings = Settings(max_concurrent_raster_reads=1, raster_admission_wait_seconds=0.05)
    reads: list[str] = []

    def band_reader(href: str, bbox: Any, **kwargs: Any) -> BandWindow:
        reads.append(href)
        return _band()

    service = ImageryService(
        settings,
        stac_item_fetcher=lambda scene_id, collection: {
            "assets": {"vv": {
                "href": "https://example.invalid/vv.tiff",
                "type": "image/tiff; application=geotiff",
            }}
        },
        band_reader=band_reader,
    )
    from app.services.geospatial import BoundingBox

    bbox = BoundingBox(west=80.27, south=13.04, east=80.29, north=13.06)

    # With every slot taken, the read is refused before the reader runs.
    semaphore = limits_mod._raster_semaphore(1)
    assert semaphore.acquire(timeout=1)
    try:
        with pytest.raises(ServiceOverloadedError):
            service.read_band(scene_id="s1", bbox=bbox, asset="vv", collection=RTC_COLLECTION)
        assert reads == []
    finally:
        semaphore.release()

    service.read_band(scene_id="s1", bbox=bbox, asset="vv", collection=RTC_COLLECTION)
    assert reads == ["https://example.invalid/vv.tiff"]


def test_both_remote_raster_reads_sit_inside_the_gate() -> None:
    source = (BACKEND / "app/services/satellite/imagery.py").read_text()
    for call in ("_read_window", "_read_band"):
        assert source.count(f"self.{call}(") == 1, call
        guarded = rf"with raster_read_slot\(self\._settings\):\s*\n\s*\w+ = self\.{call}\("
        assert re.search(guarded, source), call


# --------------------------------------------------------------------------- #
# Workflow budget
# --------------------------------------------------------------------------- #


class _SlowParser:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    async def parse_intent(self, prompt: str) -> Any:
        await asyncio.sleep(self.seconds)
        raise AssertionError("the budget should have ended this run")


def _client(**overrides: Any) -> TestClient:
    settings = Settings(workflow_budget_seconds=0.1, max_concurrent_workflows=1, **overrides)
    app = create_app(settings)
    app.dependency_overrides[get_ai_service] = lambda: _SlowParser(5)
    return TestClient(app)


def test_a_run_past_its_budget_is_ended_with_504() -> None:
    client = _client()
    started = time.perf_counter()
    response = client.post("/api/v1/query/parse", json={"prompt": "NDVI at X"})
    elapsed = time.perf_counter() - started

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "workflow_timeout"
    assert elapsed < 2.0


def test_an_ended_run_releases_its_workflow_slot() -> None:
    # One slot in total. If the first run kept it, the second would wait for a
    # slot and be refused with 503 instead of reaching its own budget.
    client = _client()
    first = client.post("/api/v1/query/parse", json={"prompt": "NDVI at X"})
    second = client.post("/api/v1/query/parse", json={"prompt": "NDVI at X"})

    assert (first.status_code, second.status_code) == (504, 504)


def _fake_request(budget: float) -> Any:
    settings = Settings(workflow_budget_seconds=budget)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        limits=SimpleNamespace(settings=settings)
    )))


def test_a_timeout_raised_by_the_work_itself_is_not_reported_as_the_budget() -> None:
    async def upstream_timed_out() -> None:
        raise TimeoutError("upstream")

    with pytest.raises(TimeoutError) as caught:
        asyncio.run(within_budget(_fake_request(5), upstream_timed_out()))
    assert not isinstance(caught.value, WorkflowTimeoutError)


def test_work_inside_the_budget_returns_its_result() -> None:
    async def quick() -> str:
        return "done"

    assert asyncio.run(within_budget(_fake_request(5), quick())) == "done"


def test_every_expensive_route_awaits_through_the_budget() -> None:
    for path, calls in {
        "app/api/routes/query.py": 4,
        "app/api/routes/satellite.py": 1,
    }.items():
        source = (BACKEND / path).read_text()
        assert source.count("await within_budget(http_request, ") == calls, path
