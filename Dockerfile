FROM python:3.11-slim

RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir --upgrade pip

RUN mkdir -p /app
RUN git clone https://github.com/01-ai/Yi --single-branch /app/Yi
RUN git clone https://github.com/dvlab-research/MGM.git --single-branch /app/MGM

WORKDIR /app
COPY requirements.txt .
ARG VERSION=latest
RUN if [ "$VERSION" = "alt" ]; then echo "transformers==4.36.2" >> requirements.txt; else echo "transformers>=4.39.0\nautoawq" >> requirements.txt ; fi
# TODO: nvidia apex wheel
RUN --mount=type=cache,target=/root/.cache/pip pip install -U -r requirements.txt

WORKDIR /app/MGM
RUN pip install --no-cache-dir --no-deps -e .

WORKDIR /app

COPY *.py .
COPY backend /app/backend

COPY model_conf_tests.json /app/model_conf_tests.json

ENV CLI_COMMAND="python vision.py"
CMD $CLI_COMMAND
