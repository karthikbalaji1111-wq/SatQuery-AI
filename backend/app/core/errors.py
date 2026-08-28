"""Application error types and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger("errors")


class AppError(Exception):
    """Base class for expected, handled application errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "app_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class NotImplementedFeatureError(AppError):
    """Raised by service stubs that are not implemented yet."""

    status_code = status.HTTP_501_NOT_IMPLEMENTED
    code = "not_implemented"


class InvalidInputError(AppError):
    """Raised when a request is well-formed but semantically invalid."""

    status_code = 422  # Unprocessable Content
    code = "invalid_input"


class NotFoundError(AppError):
    """Raised when a requested resource cannot be found."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class UpstreamServiceError(AppError):
    """Raised when a third-party service (e.g. Nominatim) fails or times out."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "upstream_error"


class ImageryError(AppError):
    """Raised when bounded imagery cannot be read or converted to RGB."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "imagery_error"


def _error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON error handlers to the app."""

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("AppError [%s]: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("internal_error", "An unexpected error occurred."),
        )
