# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install build
COPY . .
RUN pip install -e ".[dev]"

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /app /app
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "polymarket_agents.domains.crypto.agent"]
