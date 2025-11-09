# Sirpi Backend

AI-powered infrastructure automation for Google Cloud Platform. Automatically analyzes GitHub repositories, generates infrastructure code, and deploys to Google Cloud Run using Google ADK.

## Features

- Multi-agent AI system that analyzes codebases and generates optimal infrastructure
- Automated Docker builds and Cloud Run deployments
- AI assistant powered by Gemini for infrastructure management
- Real-time deployment logs via Server-Sent Events
- OAuth 2.0 integration with encrypted environment variables
- Infrastructure as code with Terraform

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Google Cloud project with Vertex AI enabled
- Supabase project
- E2B account
- Clerk account
- GitHub App credentials

### Installation

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your credentials (see .env.example for all required variables)

# Run development server
uv run uvicorn src.main:app --reload --port 8000
```

API documentation available at:
- http://localhost:8000/docs (Swagger)
- http://localhost:8000/redoc

### Environment Variables

See [.env.example](.env.example) for all required environment variables including:
- Database (Supabase)
- Authentication (Clerk)
- Google Cloud & Vertex AI
- GCP OAuth credentials
- E2B API key
- GitHub App credentials
- Encryption key

## How It Works

1. Import repository via GitHub App
2. AI agents analyze code to detect language, framework, and dependencies
3. Generate Dockerfile and Terraform configurations
4. Build Docker image in E2B sandbox and push to Artifact Registry
5. Deploy to Cloud Run using Terraform
6. Manage and scale via AI assistant

## AI Assistant

The Sirpi Assistant uses Gemini to help manage your infrastructure. It can get service details, update scaling configuration, estimate costs, analyze logs, and explain infrastructure decisions.

## Tech Stack

- FastAPI - Web framework
- Google ADK - AI agent orchestration
- Google Gemini - LLM via Vertex AI
- E2B - Sandboxed code execution
- Supabase - PostgreSQL database
- Clerk - Authentication
- Terraform - Infrastructure as code

## Development

```bash
# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Format
uv run ruff format src/

# Tests
uv run pytest
```

## Deployment

### Docker
```bash
docker build -t sirpi-backend .
docker run -p 8000:8000 --env-file .env sirpi-backend
```

### Cloud Run
```bash
gcloud run deploy sirpi-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## Database Setup

```bash
# Apply schema
psql $SUPABASE_URL < database/schema_complete.sql

# Apply migrations
psql $SUPABASE_URL < database/migrations/001_add_agent_logs.sql
```
