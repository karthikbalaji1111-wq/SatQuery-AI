"""Application error types and FastAPI exception handlers."""

from __future__ import annotations

from math import ceil

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
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
        #: Response headers this error carries, or ``None``. Exists for the one
        #: case where the status code alone is not actionable: a 429 without
        #: ``Retry-After`` tells a client to back off for an unknown time, so it
        #: retries immediately and the limit does its job twice.
        self.headers: dict[str, str] | None = None
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


class PointLocationError(InvalidInputError):
    """The place resolved to a single point, which has no area to measure.

    The geocoder found the place, so it is not "not found"; but what it has is
    a point - a stop, a shop, or a label placed on a large feature such as a
    desert - and the box around a point is a display box, not the feature's
    extent. An analysis area is never drawn around it here: that would invent
    the area being measured. The remedy is the user's - name an area.
    """

    code = "location_is_point"


class GeocodingUnavailableError(UpstreamServiceError):
    """The location service cannot be used right now.

    Raised when the geocoder refuses (HTTP 429, or this process's cooldown
    after one) or cannot answer (5xx, timeout, unreachable) after the bounded
    retries. 503 rather than the parent's 502: the request was fine and will
    very likely succeed later. When the wait is KNOWN - the upstream's
    Retry-After or this process's cooldown - it is carried, and sent as
    ``Retry-After``; when it is not, nothing is invented. A subclass, so every
    existing ``except UpstreamServiceError`` still handles it.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "geocoding_unavailable"

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        if retry_after_seconds is not None:
            self.headers = {"Retry-After": str(max(1, ceil(retry_after_seconds)))}


class ImageryError(AppError):
    """Raised when bounded imagery cannot be read or converted to RGB."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "imagery_error"


class RateLimitedError(AppError):
    """Raised when one client has exceeded its request allowance.

    Carries ``Retry-After`` so a well-behaved client knows how long to wait
    instead of guessing.
    """

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"

    def __init__(self, message: str, *, retry_after_seconds: float) -> None:
        super().__init__(message)
        # Whole seconds, rounded UP: rounding down would invite a retry that is
        # still inside the window.
        self.retry_after_seconds = retry_after_seconds
        self.headers = {"Retry-After": str(max(1, ceil(retry_after_seconds)))}


class ServiceOverloadedError(AppError):
    """Raised when every slot for an expensive operation is already in use.

    Deliberately 503 rather than 429: the client did nothing wrong and the same
    request may well succeed shortly. It is a statement about this process, not
    about this caller.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_overloaded"

    def __init__(self, message: str, *, retry_after_seconds: float = 5.0) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.headers = {"Retry-After": str(max(1, ceil(retry_after_seconds)))}


class PayloadTooLargeError(AppError):
    """Raised when a request body exceeds the configured ceiling."""

    # The 413 constant was renamed; the older alias is deprecated and warns.
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"


class WorkflowTimeoutError(AppError):
    """Raised when a workflow exhausts its total execution budget.

    504 rather than 500: the work was still running when the budget expired, so
    nothing is known to be broken - the request simply cost more than this
    deployment allows one request to cost.
    """

    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    code = "workflow_timeout"


class IntentParsingError(AppError):
    """Raised when a prompt cannot be turned into a valid ``SatQueryIntent``.

    Used when the language model produced an empty, malformed, or
    contract-violating result - the parser fails clearly rather than inventing.
    """

    status_code = 422  # Unprocessable Content
    code = "intent_parse_error"


def _error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def error_response(
    code: str,
    message: str,
    *,
    status_code: int,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """The one error envelope, as a response.

    Exists for code that cannot raise: an exception raised inside an outer HTTP
    middleware travels past the handlers registered below - they sit further in
    - and would surface as a bare 500 with a different shape. Such a middleware
    returns this instead, so one envelope still describes every failure.
    """

    return JSONResponse(
        status_code=status_code,
        content=_error_body(code, message),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON error handlers to the app."""

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("AppError [%s]: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Give a schema rejection the same envelope as every other error.

        FastAPI's default renders ``{"detail": [...]}``, so a 422 arrived in one
        of two shapes depending on whether Pydantic or an :class:`AppError`
        produced it. A client reading ``error.message`` got nothing from the
        first kind and fell back to a status-code-only string, discarding the
        one useful thing a validation failure carries: which field, and why.

        The code stays distinct from ``invalid_input``: a structurally
        malformed body and a well-formed but semantically invalid one are
        different failures, and only the envelope is being unified here.

        The offending VALUE is deliberately never echoed - only its location
        and the reason. Echoing input back is how a request's own contents
        return to it through an error, and a request body may carry a
        credential the sender should not be handed back.
        """

        details = []
        for error in exc.errors():
            location = ".".join(
                str(part) for part in error.get("loc", ()) if part != "body"
            )
            reason = str(error.get("msg", "is invalid"))
            details.append(f"{location or 'body'}: {reason}")
        message = "; ".join(details) or "The request body is invalid."
        logger.info("Request validation failed: %s", message)
        return JSONResponse(
            status_code=422,  # Unprocessable Content, as above
            content=_error_body("validation_error", message),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("internal_error", "An unexpected error occurred."),
        )
