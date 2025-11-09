"""
Sirpi AI Assistant using Google ADK.
Conversational agent for helping users with deployment questions, logs, status checks.
"""

import logging
from typing import Optional, AsyncGenerator
import os

# Google ADK imports
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from src.core.config import settings
from src.services.supabase import get_supabase_service
from src.services.gcp_assistant_tools import (
    get_cloud_run_service_details,
    update_cloud_run_scaling,
    get_project_cost_estimate,
    get_service_metrics,
    get_deployment_logs_summary,
    get_adk_agent_context,
)


logger = logging.getLogger(__name__)


def query_deployment_status(project_id: str, tool_context: Optional[ToolContext] = None) -> dict:
    """
    Check deployment status from database.

    Args:
        project_id: Project ID to check
        tool_context: ADK tool context

    Returns:
        Dictionary with deployment information
    """
    try:
        supabase = get_supabase_service()

        # Query project and latest deployment
        from src.models.schemas import Project, Deployment
        from sqlalchemy import select

        with supabase.get_session() as db:
            project = db.execute(
                select(Project).where(Project.id == project_id)
            ).scalar_one_or_none()

            if not project:
                return {"error": "Project not found", "project_id": project_id}

            # Get latest deployment
            latest_deployment = db.execute(
                select(Deployment)
                .where(Deployment.project_id == project_id)
                .order_by(Deployment.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()

            if not latest_deployment:
                return {
                    "project_name": project.repo_name,
                    "status": "no_deployments",
                    "message": "No deployments found for this project",
                }

            return {
                "project_name": project.repo_name,
                "deployment_id": latest_deployment.id,
                "status": latest_deployment.status,
                "cloud_provider": latest_deployment.cloud_provider or "gcp",
                "service_url": latest_deployment.service_url,
                "created_at": latest_deployment.created_at.isoformat(),
                "updated_at": latest_deployment.updated_at.isoformat(),
            }

    except Exception as e:
        logger.error(f"Failed to query deployment status: {e}")
        return {"error": str(e)}


def get_deployment_logs(
    project_id: str, limit: int = 50, tool_context: Optional[ToolContext] = None
) -> dict:
    """
    Retrieve deployment logs from database.
    Uses the new deployment logs summary tool.

    Args:
        project_id: Project ID
        limit: Maximum number of log entries
        tool_context: ADK tool context

    Returns:
        Dictionary with log entries
    """
    return get_deployment_logs_summary(project_id, limit, tool_context)


def check_gcp_resource(
    service_name: str,
    project_id: str,
    region: str = "us-central1",
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Check Google Cloud Run service status.

    Args:
        service_name: Cloud Run service name
        project_id: GCP project ID
        region: GCP region
        tool_context: ADK tool context

    Returns:
        Service status information
    """
    try:
        from google.cloud import run_v2

        client = run_v2.ServicesClient()
        service_path = f"projects/{project_id}/locations/{region}/services/{service_name}"

        service = client.get_service(name=service_path)

        return {
            "service_name": service_name,
            "status": "RUNNING" if service.terminal_condition.state == 1 else "UNKNOWN",
            "url": service.uri,
            "latest_revision": service.latest_ready_revision,
            "creation_time": service.create_time.isoformat() if service.create_time else None,
        }

    except Exception as e:
        logger.warning(f"Failed to check GCP resource: {e}")
        return {"error": str(e), "message": "Service not found or not accessible"}


class SirpiAssistant:
    """
    Conversational AI assistant for Sirpi using Google ADK.
    Helps users check deployment status, view logs, and troubleshoot.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        """Initialize assistant with Gemini model and Vertex AI."""
        self.model = model
        self.session_service = InMemorySessionService()

        # Set environment variables for ADK to use Sirpi's Vertex AI
        # (ADK reads these internally when creating the model)
        os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud_project
        os.environ["GOOGLE_CLOUD_LOCATION"] = settings.google_cloud_location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = (
            "TRUE" if settings.google_genai_use_vertexai else "FALSE"
        )

        logger.info(
            f"Assistant will use Sirpi's Vertex AI project: {settings.google_cloud_project}"
        )

        # Create ADK agent with enhanced tools
        # The agent uses Sirpi's Vertex AI to run Gemini (for answering questions)
        # The tools use user's GCP credentials to query their resources
        self.agent = Agent(
            name="SirpiAssistant",
            model=self.model,
            description="AI assistant for Sirpi deployment platform with GCP management capabilities",
            instruction=self._get_instruction(),
            tools=[
                # Basic status and logs
                query_deployment_status,
                get_deployment_logs,
                check_gcp_resource,
                # Enhanced GCP tools
                get_cloud_run_service_details,
                update_cloud_run_scaling,
                get_project_cost_estimate,
                get_service_metrics,
                get_adk_agent_context,
            ],
        )

        # Create runner
        self.runner = Runner(
            agent=self.agent, app_name=settings.adk_app_name, session_service=self.session_service
        )

        logger.info("Sirpi Assistant initialized with Google ADK")

    def _get_instruction(self) -> str:
        """System instruction for the assistant."""
        return """You are Sirpi Assistant, an expert AI helper for the Sirpi DevOps automation platform with deep GCP expertise.

**IMPORTANT: Context Awareness**
Every user message includes project context in square brackets at the top. This contains:
- Project Name: The user's project name
- Project ID: The Sirpi project UUID
- Cloud Provider: GCP or AWS
- Cloud Run Service: The service name (for GCP projects)
- Application URL: The deployed app URL (if available)

**USE THIS CONTEXT AUTOMATICALLY** - don't ask users for information that's already provided in the context!

**Your Enhanced Capabilities:**

1. **Deployment Management:**
   - Check deployment status with query_deployment_status (use project_id from context)
   - Retrieve and analyze deployment logs with get_deployment_logs
   - Access ADK agent context to understand infrastructure decisions with get_adk_agent_context

2. **GCP Infrastructure Control:**
   - Get detailed Cloud Run service info with get_cloud_run_service_details (use project_id and service_name from context)
   - Update scaling configuration (min/max instances) with update_cloud_run_scaling
   - Monitor service metrics and performance with get_service_metrics

3. **Cost Optimization:**
   - Analyze project costs with get_project_cost_estimate
   - Provide cost-saving recommendations
   - Suggest optimal scaling configurations

4. **Smart Recommendations:**
   - For dev environments: Suggest min_instances=0 to save costs
   - For production: Recommend appropriate min/max based on traffic
   - Analyze logs to identify issues and suggest fixes

**Your Personality:**
- Expert in GCP, Cloud Run, and cost optimization
- Proactive in suggesting improvements
- Clear and actionable in recommendations
- Data-driven decision making

**Guidelines:**
- ALWAYS extract project_id and service_name from the context at the top of the message
- Use tools to get real-time data before answering
- When users ask about costs, use get_project_cost_estimate with the project_id from context
- When users want to change scaling, use update_cloud_run_scaling with project_id and service_name from context
- Explain the impact of configuration changes
- Provide specific commands and values, not just general advice
- Reference the ADK agent context to explain why infrastructure was configured a certain way

**Example Interactions:**
- "Show my scaling" → Extract project_id and service_name from context, call get_cloud_run_service_details
- "How much is this costing?" → Use get_project_cost_estimate with project_id from context
- "Reduce my costs" → Check current scaling, suggest min_instances=0 for dev or lower max for prod
- "Scale up my service" → Use update_cloud_run_scaling with project_id and service_name from context"""

    async def chat(
        self, user_message: str, user_id: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Stream chat response from assistant.

        Args:
            user_message: User's message
            user_id: User identifier
            session_id: Session ID (created if None)

        Yields:
            Event dictionaries with partial or final responses
        """
        # Create or get session
        if not session_id:
            session = await self.session_service.create_session(
                app_name=settings.adk_app_name, user_id=user_id
            )
            session_id = session.id

        # Build message content
        content = types.Content(role="user", parts=[types.Part(text=user_message)])

        logger.info(f"Assistant chat: {user_message[:100]}...")

        try:
            # Stream response
            async for event in self.runner.run_async(
                user_id=user_id, session_id=session_id, new_message=content
            ):
                # Convert event to dict for API response
                event_dict = {
                    "type": "partial" if event.partial else "final",
                    "content": "",
                    "tool_calls": [],
                    "tool_responses": [],
                }

                # Extract text content
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            event_dict["content"] = part.text

                # Extract tool calls
                if hasattr(event, "get_function_calls"):
                    function_calls = event.get_function_calls()
                    if function_calls:
                        event_dict["tool_calls"] = [
                            {"name": fc.name, "args": fc.args} for fc in function_calls
                        ]

                # Extract tool responses
                if hasattr(event, "get_function_responses"):
                    function_responses = event.get_function_responses()
                    if function_responses:
                        event_dict["tool_responses"] = [
                            {"name": fr.name, "response": fr.response} for fr in function_responses
                        ]

                yield event_dict

        except Exception as e:
            logger.error(f"Assistant chat failed: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": f"Sorry, I encountered an error: {str(e)}",
                "error": str(e),
            }

    async def chat_simple(
        self, user_message: str, user_id: str, session_id: Optional[str] = None
    ) -> str:
        """
        Get complete response without streaming.

        Args:
            user_message: User's message
            user_id: User identifier
            session_id: Session ID

        Returns:
            Complete assistant response
        """
        full_response = ""

        async for event in self.chat(user_message, user_id, session_id):
            if event["type"] == "final" and event["content"]:
                full_response = event["content"]
                break
            elif event["type"] == "partial" and event["content"]:
                full_response += event["content"]

        return full_response if full_response else "I couldn't process that request."


# Singleton instance
_assistant: Optional[SirpiAssistant] = None


def get_sirpi_assistant() -> SirpiAssistant:
    """Get or create singleton Sirpi Assistant."""
    global _assistant

    if _assistant is None:
        _assistant = SirpiAssistant()

    return _assistant
