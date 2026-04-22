FROM python:3.12-slim

WORKDIR /app

# System dependencies. tini runs as PID 1 so SIGTERM actually reaches
# uvicorn and lifespan shutdown can run — without it `docker stop`
# kills the container after its 10s grace period without giving the
# FastAPI shutdown hook a chance to flush positions / logout brokers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    tini \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create required directories
RUN mkdir -p logs data reports .tmp

# Non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
