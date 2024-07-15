FROM python:3.11-slim

RUN apt-get update && apt-get install -y git gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip

WORKDIR /app
RUN git clone https://github.com/01-ai/Yi --single-branch /app/Yi
RUN git clone https://github.com/dvlab-research/MGM.git --single-branch /app/MGM
RUN git clone https://github.com/TIGER-AI-Lab/Mantis.git --single-branch /app/Mantis
RUN git clone https://github.com/togethercomputer/Dragonfly --single-branch /app/Dragonfly

COPY requirements.txt .
ARG VERSION=latest
RUN if [ "$VERSION" = "alt" ]; then echo "transformers==4.36.2" >> requirements.txt; else echo "transformers==4.41.2\nautoawq>=0.2.5" >> requirements.txt ; fi
# TODO: nvidia apex wheel
RUN --mount=type=cache,target=/root/.cache/pip pip install -U -r requirements.txt

WORKDIR /app/MGM
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -e .

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
