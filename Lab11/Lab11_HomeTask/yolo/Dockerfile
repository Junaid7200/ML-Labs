FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    jinja2 \
    pillow \
    numpy \
    ultralytics

COPY . /app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
