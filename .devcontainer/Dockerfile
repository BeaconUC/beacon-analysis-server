FROM python:3.12-slim-trixie

ARG USERNAME=vscode \
    USER_UID=1000 \
    USER_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    KAGGLE_CONFIG_DIR=/workspace \
    APP_HOME=/workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    openssh-client \
    sudo \
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
    && rm -rf /var/lib/apt/lists/* \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN mkdir -p $APP_HOME /home/$USERNAME/.vscode-server/extensions /home/$USERNAME/.vscode-server/data \
    && chown -R $USERNAME:$USERNAME $APP_HOME /home/$USERNAME

WORKDIR $APP_HOME

ENTRYPOINT ["/usr/bin/tini", "--"]

USER $USERNAME

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3002"]

EXPOSE 3002
