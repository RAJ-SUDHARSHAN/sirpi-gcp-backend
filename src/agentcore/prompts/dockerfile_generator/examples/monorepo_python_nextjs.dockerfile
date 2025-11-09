# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci && npm cache clean --force
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Backend Dependencies
FROM python:3.12-slim AS backend-builder
WORKDIR /app
RUN pip install uv
COPY backend/pyproject.toml backend/uv.lock* ./
RUN uv venv && uv sync --no-dev

# Stage 3: Runtime - Combine Both
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=backend-builder /app/.venv ./.venv
COPY backend/ .
# Copy frontend build output - {FRONTEND_BUILD_OUTPUT} will be replaced dynamically
COPY --from=frontend-builder /app/frontend/{FRONTEND_BUILD_OUTPUT} ./static
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app
ENV PATH="/app/.venv/bin:$PATH"
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0"]

