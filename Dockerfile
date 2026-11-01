# MedBot runtime image.
# Build:  docker build -t medbot .
# Run:    docker run --rm -p 8080:8080 --env-file .env medbot
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PORT=8080

# libgomp is required by torch; curl-free healthcheck keeps the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY requirements.txt requirements-base.txt requirements-ml.txt ./

# CPU-only torch: the default Linux wheel bundles ~2 GB of CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" \
    && pip install -r requirements.txt \
    && pip install "gunicorn==23.0.0"

COPY --chown=app:app . .

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"

# One worker per core is unnecessary here: the embedding model is loaded per
# worker, so keep the count low and let threads handle concurrent requests.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", \
     "--timeout", "120", "--access-logfile", "-", "app:app"]
