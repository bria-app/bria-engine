"""HTTP routes for the FTGM engine."""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.application.forecast_service import ForecastService, engine_thresholds
from app.config import settings
from app.ftgm import FTGMError
from app.presentation.schemas import EngineConfig, ForecastRequest, ForecastResponse

router = APIRouter(tags=["FTGM Engine"])

# Stateless service — safe to share across requests.
_service = ForecastService()


def require_engine_key(x_engine_key: str | None = Header(default=None)) -> None:
    """Shared-secret check between the API and the engine (audit A-06).

    Active only when ``ENGINE_API_KEY`` is configured; the comparison is constant-time.
    """
    expected = settings.engine_api_key
    if not expected:
        return
    if x_engine_key is None or not hmac.compare_digest(
        x_engine_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clave del motor ausente o inválida (header X-Engine-Key).",
        )


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "bria-engine"}


@router.get("/config", response_model=EngineConfig)
def engine_config() -> EngineConfig:
    """Read-only thresholds the pipeline applies (frequency, intermittency, stock-outs)."""
    return engine_thresholds()


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    dependencies=[Depends(require_engine_key)],
)
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Run the FTGM forecast pipeline for a batch of product series."""
    if not request.series:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="series must contain at least one product",
        )
    try:
        return _service.forecast(request)
    except FTGMError as exc:  # defensive: per-product fallbacks should prevent this
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
