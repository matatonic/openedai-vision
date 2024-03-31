FROM python:3-slim

RUN mkdir -p /app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py .
COPY backend /app/backend
CMD python vision.py
