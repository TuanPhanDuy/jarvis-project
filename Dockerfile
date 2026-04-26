FROM python:3.11-slim

# Install system deps needed by some packages (PyAudio, Playwright build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency manifests first for layer caching
COPY pyproject.toml ./
COPY src/ src/

# Install production dependencies (no voice extras in server image)
RUN uv pip install --system -e ".[fastapi]" 2>/dev/null || \
    uv pip install --system -e .

# Non-root user for security
RUN useradd -m -u 1000 jarvis
USER jarvis

EXPOSE 8000

# Default: run the API server
CMD ["python", "-m", "jarvis.api.server"]
