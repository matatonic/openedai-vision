FROM python:3.11-slim

RUN mkdir -p /app
WORKDIR /app
COPY requirements.txt .
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install -r requirements.txt

COPY *.py .
COPY backend /app/backend
CMD python vision.py
