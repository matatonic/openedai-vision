FROM python:3.11-slim

RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir --upgrade pip

RUN mkdir -p /app
RUN git clone https://github.com/01-ai/Yi --single-branch /app/Yi
RUN git clone https://github.com/dvlab-research/MGM.git --single-branch /app/MGM

WORKDIR /app
COPY requirements.txt .
ARG VERSION=latest
# transformers==4.36.2 supports most models except MGM-2B, llava-1.6, nanollava
RUN if [ "$VERSION" = "alt" ]; then echo "transformers==4.36.2" >> requirements.txt; else echo "transformers>=4.39.0" >> requirements.txt ; fi
# TODO: nvidia apex wheel
RUN pip install --no-cache-dir -U -r requirements.txt \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

WORKDIR /app/MGM
RUN pip install --no-cache-dir --no-deps -e .

WORKDIR /app

COPY *.py .
COPY backend /app/backend

COPY model_conf_tests.json /app/model_conf_tests.json

ENV CLI_COMMAND="python vision.py"
CMD $CLI_COMMAND
