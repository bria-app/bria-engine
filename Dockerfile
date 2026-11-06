# FTGM analytical engine (FastAPI + numpy/scipy). No database.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Dependencies first, in their own cached layer (stub package), so code edits rebuild fast.
COPY pyproject.toml ./
# Runtime dependencies only: test tooling lives in the `dev` extra (pip install -e ".[dev]").
RUN mkdir -p app && touch app/__init__.py && pip install . && pip uninstall -y bria-engine

COPY . .
RUN pip install --no-deps .

EXPOSE 8010
# Bind to $PORT when the platform injects one (Render/Railway), else 8010 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8010}"]
