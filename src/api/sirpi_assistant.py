"""
Sirpi AI Assistant API - Powered by Google Gemini.
Chat interface for deployment assistance and GCP infrastructure management.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

from src.services.sirpi_assistant import get_sirpi_assistant
from src.utils.clerk_auth import get_current_user_id
from src.services.supabase import get_supabase_service

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    project_id: str
    question: str
    include_logs: bool = True


@router.post("/assistant/chat")
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user_id)):
    """Chat with Sirpi AI Assistant (powered by Google Gemini)."""
    try:
        supabase = get_supabase_service()

        # Verify ownership
        project = supabase.get_project_by_id(request.project_id)
        if not project or project["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Note: The assistant creates its own session. We don't reuse the orchestrator's
        # session because they use different session services (orchestrator uses DB,
        # assistant uses in-memory). The assistant can still access orchestrator context
        # via the get_adk_agent_context tool.

        # Enhance user message with project context (use name for better UX)
        project_name = project.get("name") or project.get("repository_name", "Unknown")
        cloud_provider = project.get("cloud_provider", "gcp")

        # Build context string with all relevant project info
        context_parts = [
            f"Project Name: {project_name}",
            f"Project ID: {request.project_id}",
            f"Cloud Provider: {cloud_provider.upper()}",
        ]

        # Add Cloud Run service name (derived from project name)
        if cloud_provider == "gcp":
            # Cloud Run service name is typically the project name (lowercase, no special chars)
            service_name = project_name.lower().replace(" ", "-").replace("_", "-")
            context_parts.append(f"Cloud Run Service: {service_name}")

        # Add application URL if available
        app_url = project.get("application_url")
        if app_url:
            context_parts.append(f"Application URL: {app_url}")

        context_string = " | ".join(context_parts)
        enhanced_message = f"[{context_string}]\n\nUser Question: {request.question}"

        # Call assistant with streaming
        assistant = get_sirpi_assistant()

        # Collect the full response
        full_response = ""

        async for event in assistant.chat(
            user_message=enhanced_message,
            user_id=user_id,
            session_id=None,  # Let assistant create its own session
        ):
            if event["type"] == "error":
                raise HTTPException(status_code=500, detail=event.get("error", "Unknown error"))

            if event["type"] == "final" and event.get("content"):
                full_response = event["content"]
                break
            elif event["type"] == "partial" and event.get("content"):
                # Accumulate partial responses
                full_response += event["content"]

        if not full_response:
            full_response = "I couldn't process that request. Please try again."

        return {
            "success": True,
            "data": {
                "answer": full_response,
                "model": "gemini-2.5-flash",
                "agentcore_memory_used": False,  # Assistant has its own session
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
