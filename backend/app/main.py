"""SatQuery FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import build_api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.limits import install_request_limits
from app.core.logging import configure_logging, get_logger
from app.services.geospatial.nominatim import warm_geocoder_cache, warm_places


def _lifespan(
    settings: Settings,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Start the optional geocoder warm-up; stop it with the application.

    The warm-up runs in the BACKGROUND: startup never waits on a third-party
    service, and a warm-up that cannot reach the geocoder only leaves the cache
    as cold as it would have been anyway.
    """

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task[list[str]] | None = None
        if warm_places(settings):
            task = asyncio.create_task(warm_geocoder_cache(settings))
        try:
            yield
        finally:
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""

    settings = settings or get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("main")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        summary="Natural-language satellite query platform - foundation build.",
        lifespan=_lifespan(settings),
    )

    # Order matters, and it is the reverse of the reading order: Starlette runs
    # the most recently added middleware OUTERMOST. The size guard is installed
    # first so that CORS ends up outside it - otherwise a 413 would be returned
    # without CORS headers, and a browser would report it as a network failure
    # rather than as the refusal it is.
    install_request_limits(app, settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    # Built from THIS application's settings, not from an import-time singleton.
    app.include_router(build_api_router(settings))

    logger.info(
        "SatQuery API initialised (env=%s, version=%s)",
        settings.environment,
        __version__,
    )
    return app


app = create_app()
