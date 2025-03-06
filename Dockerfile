FROM python:3.11-slim

RUN apt-get update && apt-get install -y git gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip

WORKDIR /app
RUN git clone https://github.com/deepseek-ai/DeepSeek-VL2 --single-branch /app/DeepSeek-VL2 && \
    git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git --single-branch /app/LLaVA-NeXT

COPY requirements.txt .
ARG VERSION=latest
RUN if [ "$VERSION" = "alt" ]; then echo "transformers==4.41.2" >> requirements.txt; else echo "git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision" >> requirements.txt ; fi
RUN --mount=type=cache,target=/root/.cache/pip pip install -U -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps "git+https://github.com/casper-hansen/AutoAWQ.git"
RUN --mount=type=cache,target=/root/.cache/pip pip install gptqmodel --no-build-isolation

WORKDIR /app/DeepSeek-VL2
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -e .

WORKDIR /app/LLaVA-NeXT
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -e .

WORKDIR /app

COPY *.py model_conf_tests.json README.md LICENSE /app/
COPY backend /app/backend

ARG USER_ID=1000
ENV USER_ID=${USER_ID}
ARG GROUP_ID=1000
ENV GROUP_ID=${GROUP_ID}
RUN groupadd -g ${GROUP_ID} openedai && \
    useradd -r -u ${USER_ID} -g ${GROUP_ID} -M -d /app openedai
RUN chown openedai:openedai /app # for .triton, .config/matplotlib

USER openedai
ENV CLI_COMMAND="python vision.py"
CMD $CLI_COMMAND
