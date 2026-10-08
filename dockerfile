# ==========================================================================
# STAGE 1 : Build stage with uv
# ==========================================================================

FROM python:3.12-slim AS dependencies

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install deps using uv
RUN uv sync --frozen --no-dev

# ==========================================================================
# STAGE 2: Runtime
# ==========================================================================

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from build stage
COPY --from=dependencies /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# Copy app code — src, config and semantic layer only, data is bind-mounted
COPY src ./src
COPY config ./config
COPY semantic_layer ./semantic_layer
COPY app.py .

# Chainlit UI: config and translations, custom CSS / elements / icons, Readme page
COPY .chainlit ./.chainlit
COPY public ./public
COPY chainlit.md .

# Change ownership to app user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "-w"]
