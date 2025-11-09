FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app
ENV PATH=/root/.local/bin:$PATH
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0"]

