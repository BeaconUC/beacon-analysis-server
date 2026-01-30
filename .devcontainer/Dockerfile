FROM python:3.12-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    KAGGLE_CONFIG_DIR=/workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openssh-client \
    curl \
    git \
    tini \
    tree \
    vim \
    psmisc \
    procps \
    libnss3 \
    libnspr4 \
    libasound2 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

EXPOSE 3002

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3002"]