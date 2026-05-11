# Build context must be the pnyx/ parent directory (see build.sh) so that
# pnyx-privacy-filter/ is accessible to the COPY instructions.
FROM python:3.11-slim

# Install build tools and healthcheck requirements (curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libssl-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached independently from source code)
COPY pnyx-privacy-filter/private_chat_manager/pyproject.toml ./pyproject.toml
# Create a minimal package stub so pip can resolve the project's own dependencies
RUN mkdir -p app && touch app/__init__.py && \
    pip install --no-cache-dir ".[prod]"

# Copy application source
COPY pnyx-privacy-filter/private_chat_manager/app ./app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
