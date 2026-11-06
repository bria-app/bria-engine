from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
from app.presentation.router import router

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.app_debug,
        version="0.1.0"
    )

    # 422 without echoing the rejected input: a NaN/inf value cannot be serialised as
    # JSON (the default handler would turn it into a 500) and an oversized batch would
    # be sent back whole.
    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{k: v for k, v in err.items() if k != "input"} for err in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(errors)},
        )

    # Public liveness endpoint (used by platform health checks).
    @app.get("/health", tags=["System"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router, prefix=settings.api_v1_prefix)

    return app

app = create_app()
