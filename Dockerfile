FROM python:3.11-slim

RUN apt-get update && apt-get install -y git gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip

WORKDIR /app
RUN git clone https://github.com/TIGER-AI-Lab/Mantis.git --single-branch /app/Mantis
RUN git clone https://github.com/togethercomputer/Dragonfly --single-branch /app/Dragonfly

COPY requirements.txt .
ARG VERSION=latest
RUN if [ "$VERSION" = "alt" ]; then echo "transformers==4.41.2" >> requirements.txt; else echo "transformers>=4.45.0" >> requirements.txt ; fi
RUN --mount=type=cache,target=/root/.cache/pip pip install -U -r requirements.txt

WORKDIR /app/Mantis
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -e .

WORKDIR /app/Dragonfly
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -e .

WORKDIR /app

COPY *.py .
COPY backend /app/backend
COPY model_conf_tests.json .

ENV CLI_COMMAND="python vision.py"
CMD $CLI_COMMAND
