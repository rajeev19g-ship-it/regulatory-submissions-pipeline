dockerfile# ── Regulatory Submissions Pipeline — Production Dockerfile ───────────────────
#
# Multi-stage build for the eCTD/IND/NDA regulatory automation platform.
# Includes LLM document generation, medical translation, and drug discovery.
#
# Author : Girish Rajeev
#          Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc g++ git curl libxml2-dev libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Production runtime ───────────────────────────────────────────────
FROM python:3.10-slim AS production

LABEL maintainer="Girish Rajeev"
LABEL description="Regulatory Submissions Pipeline — eCTD/IND/NDA/BLA Automation"
LABEL version="1.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY tests/ ./tests/

RUN mkdir -p /app/data /app/submissions /app/translations /app/logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SUBMISSIONS_DIR=/app/submissions
ENV TRANSLATIONS_DIR=/app/translations

RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from src.ectd.validator import eCTDValidator; print('healthy')" || exit 1

CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
