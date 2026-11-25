from typing import Any
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


# Custom error class so every part of the app can raise a clean HTTP error with a code
class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def _request_id(request: Request) -> str:
    # Grab the request ID attached by our middleware, or fall back to a placeholder
    return getattr(request.state, "request_id", "req_unknown")


def register_exception_handlers(app: FastAPI) -> None:
    # Turn our own AppError exceptions into a consistent JSON error response
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": _request_id(request),
                }
            },
        )

    # Turn FastAPI/Pydantic bad-input errors into the same JSON error format
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "REQUEST_VALIDATION_ERROR",
                    "message": "Request payload failed validation",
                    "details": {"errors": exc.errors()},
                    "request_id": _request_id(request),
                }
            },
        )

    # Catch any unexpected crash and return a clean 500 error instead of a raw stack trace
    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal error occurred",
                    "details": {"type": type(exc).__name__},
                    "request_id": _request_id(request),
                }
            },
        )
