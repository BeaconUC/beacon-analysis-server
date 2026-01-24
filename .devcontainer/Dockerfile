FROM python:3.14-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    KAGGLE_CONFIG_DIR=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    tini \
    tree \
    vim \
    libnss3 \
    libnspr4 \
    libasound2 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

EXPOSE 3002

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3002"]