# MedBot runtime image.
# Build:  docker build -t medbot .
# Run:    docker run --rm -p 8080:8080 --env-file .env medbot
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY requirements.txt requirements-base.txt requirements-rag.txt ./
RUN pip install -r requirements.txt \
    && pip install "gunicorn==23.0.0"

COPY --chown=app:app . .

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", \
     "--timeout", "120", "--access-logfile", "-", "app:app"]
