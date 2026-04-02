FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY api ./api

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[api]" \
    && pip install --no-cache-dir -r requirements.txt

COPY model ./model
COPY dashboard ./dashboard
COPY run_services.py ./

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
