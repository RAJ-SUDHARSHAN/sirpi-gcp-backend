"""
GCP Assistant Tools - Enhanced tools for infrastructure management and cost analysis.
Provides real-time GCP resource monitoring, cost analysis, and configuration management.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from google.cloud import run_v2
from google.adk.tools import ToolContext

# Optional imports for advanced features
try:
    from google.cloud import monitoring_v3

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

from src.services.supabase import get_supabase_service
from src.utils.gcp_credentials_validator import _get_gcp_credentials

logger = logging.getLogger(__name__)


def get_cloud_run_service_details(
    project_id: str,
    service_name: str,
    region: str = "us-central1",
    user_id: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Get detailed Cloud Run service information including scaling config.

    Args:
        project_id: Sirpi project ID (UUID)
        service_name: Cloud Run service name
        region: GCP region
        user_id: User ID for OAuth credentials
        tool_context: ADK tool context

    Returns:
        Service details including min/max instances, CPU, memory, etc.
    """
    try:
        supabase = get_supabase_service()

        # Get project to find GCP project ID
        project = supabase.get_project_by_id(project_id)
        if not project:
            return {"error": "Project not found"}

        # Get GCP credentials from database
        gcp_creds = supabase.get_gcp_credentials(user_id or project["user_id"])
        if not gcp_creds:
            return {"error": "GCP credentials not found. Please connect your GCP account."}

        gcp_project_id = gcp_creds.get("project_id")

        # Get OAuth credentials for API calls
        try:
            credentials = _get_gcp_credentials(user_id or project["user_id"], gcp_project_id)
        except Exception as e:
            return {"error": f"Failed to get credentials: {str(e)}"}

        # Get Cloud Run service
        client = run_v2.ServicesClient(credentials=credentials)
        service_path = f"projects/{gcp_project_id}/locations/{region}/services/{service_name}"

        service = client.get_service(name=service_path)

        # Extract scaling configuration
        template = service.template
        scaling = template.scaling if template else None

        return {
            "service_name": service_name,
            "status": "RUNNING" if service.terminal_condition.state == 1 else "DEPLOYING",
            "url": service.uri,
            "region": region,
            "scaling": {
                "min_instances": scaling.min_instance_count if scaling else 0,
                "max_instances": scaling.max_instance_count if scaling else 100,
            },
            "resources": {
                "cpu": template.containers[0].resources.limits.get("cpu", "1")
                if template and template.containers
                else "1",
                "memory": template.containers[0].resources.limits.get("memory", "512Mi")
                if template and template.containers
                else "512Mi",
            },
            "latest_revision": service.latest_ready_revision,
            "created_at": service.create_time.isoformat() if service.create_time else None,
            "updated_at": service.update_time.isoformat() if service.update_time else None,
        }

    except Exception as e:
        logger.error(f"Failed to get Cloud Run service: {e}")
        return {
            "error": str(e),
            "message": "Could not retrieve service details. Make sure the service exists and you have access.",
        }


def update_cloud_run_scaling(
    project_id: str,
    service_name: str,
    min_instances: Optional[int] = None,
    max_instances: Optional[int] = None,
    region: str = "us-central1",
    user_id: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Update Cloud Run service scaling configuration.

    Args:
        project_id: Sirpi project ID
        service_name: Cloud Run service name
        min_instances: Minimum number of instances (0-100)
        max_instances: Maximum number of instances (1-1000)
        region: GCP region
        user_id: User ID for OAuth credentials
        tool_context: ADK tool context

    Returns:
        Updated service configuration
    """
    try:
        supabase = get_supabase_service()

        # Get project
        project = supabase.get_project_by_id(project_id)
        if not project:
            return {"error": "Project not found"}

        # Get GCP credentials
        gcp_creds = supabase.get_gcp_credentials(user_id or project["user_id"])
        if not gcp_creds:
            return {"error": "GCP credentials not found"}

        gcp_project_id = gcp_creds.get("project_id")

        # Get OAuth credentials for API calls
        try:
            credentials = _get_gcp_credentials(user_id or project["user_id"], gcp_project_id)
        except Exception as e:
            return {"error": f"Failed to get credentials: {str(e)}"}

        # Get current service
        client = run_v2.ServicesClient(credentials=credentials)
        service_path = f"projects/{gcp_project_id}/locations/{region}/services/{service_name}"

        service = client.get_service(name=service_path)

        # Update scaling configuration
        if min_instances is not None:
            service.template.scaling.min_instance_count = min_instances
        if max_instances is not None:
            service.template.scaling.max_instance_count = max_instances

        # Update service (returns an Operation)
        operation = client.update_service(service=service)

        # Wait for the operation to complete
        operation.result()  # This blocks until the operation completes

        # Get the updated service
        updated_service = client.get_service(name=service_path)

        logger.info(
            f"Updated Cloud Run scaling: {service_name} (min={min_instances}, max={max_instances})"
        )

        return {
            "success": True,
            "service_name": service_name,
            "updated_scaling": {
                "min_instances": updated_service.template.scaling.min_instance_count,
                "max_instances": updated_service.template.scaling.max_instance_count,
            },
            "message": f"Successfully updated scaling configuration for {service_name}",
        }

    except Exception as e:
        logger.error(f"Failed to update Cloud Run scaling: {e}")
        return {"error": str(e), "message": "Could not update scaling configuration"}


def get_project_cost_estimate(
    project_id: str,
    days: int = 30,
    user_id: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Get cost estimate for GCP project resources.

    Args:
        project_id: Sirpi project ID
        days: Number of days to analyze (default 30)
        user_id: User ID for OAuth credentials
        tool_context: ADK tool context

    Returns:
        Cost breakdown and estimates
    """
    try:
        supabase = get_supabase_service()

        # Get project
        project = supabase.get_project_by_id(project_id)
        if not project:
            return {"error": "Project not found"}

        # Get GCP credentials
        gcp_creds = supabase.get_gcp_credentials(user_id or project["user_id"])
        if not gcp_creds:
            return {"error": "GCP credentials not found"}

        gcp_project_id = gcp_creds.get("project_id")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Note: Detailed billing data requires Cloud Billing API + BigQuery export
        # This provides estimated costs based on typical Cloud Run usage

        return {
            "project_name": project.get("name"),
            "period": f"Last {days} days",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "estimated_costs": {
                "cloud_run": {
                    "description": "Serverless container execution",
                    "estimated_monthly": "$5-20",
                    "factors": ["Request count", "CPU time", "Memory usage", "Instance hours"],
                },
                "artifact_registry": {
                    "description": "Container image storage",
                    "estimated_monthly": "$0.10-1",
                    "factors": ["Storage size", "Network egress"],
                },
                "cloud_build": {
                    "description": "CI/CD builds",
                    "estimated_monthly": "$0-5",
                    "factors": ["Build minutes", "Build frequency"],
                },
            },
            "optimization_tips": [
                "Set min_instances to 0 for dev environments to reduce idle costs",
                "Use Cloud Run's request-based pricing by keeping max_instances low",
                "Clean up old container images in Artifact Registry",
                "Monitor and set appropriate CPU/memory limits",
            ],
            "note": "For detailed billing data, enable Cloud Billing API and export to BigQuery",
        }

    except Exception as e:
        logger.error(f"Failed to get cost estimate: {e}")
        return {
            "error": str(e),
            "message": "Could not retrieve cost data. Make sure Cloud Billing API is enabled.",
        }


def get_service_metrics(
    project_id: str,
    service_name: str,
    hours: int = 24,
    region: str = "us-central1",
    user_id: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Get Cloud Run service metrics (requests, latency, errors).

    Args:
        project_id: Sirpi project ID
        service_name: Cloud Run service name
        hours: Hours of metrics to retrieve
        region: GCP region
        user_id: User ID for OAuth credentials
        tool_context: ADK tool context

    Returns:
        Service metrics and performance data
    """
    try:
        supabase = get_supabase_service()

        # Get project
        project = supabase.get_project_by_id(project_id)
        if not project:
            return {"error": "Project not found"}

        # Get GCP credentials
        gcp_creds = supabase.get_gcp_credentials(user_id or project["user_id"])
        if not gcp_creds:
            return {"error": "GCP credentials not found"}

        gcp_project_id = gcp_creds.get("project_id")

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        if not MONITORING_AVAILABLE:
            return {
                "service_name": service_name,
                "period": f"Last {hours} hours",
                "note": "Cloud Monitoring API not available. Install google-cloud-monitoring for detailed metrics.",
                "metrics": {"status": "Metrics require google-cloud-monitoring package"},
            }

        # Get OAuth credentials for API calls
        try:
            credentials = _get_gcp_credentials(user_id or project["user_id"], gcp_project_id)
        except Exception as e:
            return {"error": f"Failed to get credentials: {str(e)}"}

        # Get metrics using Cloud Monitoring API
        monitoring_client = monitoring_v3.MetricServiceClient(credentials=credentials)
        project_name = f"projects/{gcp_project_id}"

        # Query metrics
        interval = monitoring_v3.TimeInterval(
            {
                "end_time": {"seconds": int(end_time.timestamp())},
                "start_time": {"seconds": int(start_time.timestamp())},
            }
        )

        # This is a simplified version - actual implementation would query specific metrics
        return {
            "service_name": service_name,
            "period": f"Last {hours} hours",
            "metrics": {
                "request_count": "Available via Cloud Monitoring",
                "request_latency_ms": "Available via Cloud Monitoring",
                "error_rate": "Available via Cloud Monitoring",
                "instance_count": "Available via Cloud Monitoring",
                "cpu_utilization": "Available via Cloud Monitoring",
                "memory_utilization": "Available via Cloud Monitoring",
            },
            "note": "Detailed metrics require Cloud Monitoring API queries with specific metric types",
        }

    except Exception as e:
        logger.error(f"Failed to get service metrics: {e}")
        return {"error": str(e), "message": "Could not retrieve metrics"}


def get_deployment_logs_summary(
    project_id: str, limit: int = 50, tool_context: Optional[ToolContext] = None
) -> dict:
    """
    Get recent deployment logs from database.

    Args:
        project_id: Sirpi project ID
        limit: Maximum number of log entries
        tool_context: ADK tool context

    Returns:
        Recent deployment logs
    """
    try:
        supabase = get_supabase_service()

        # Get deployment logs from database
        logs = supabase.get_deployment_logs(project_id)

        if not logs:
            return {"project_id": project_id, "message": "No deployment logs found"}

        # Summarize logs by operation type
        summary = {}
        for log_record in logs[:limit]:
            op_type = log_record.get("operation_type", "unknown")
            if op_type not in summary:
                summary[op_type] = {"count": 0, "last_status": None, "last_timestamp": None}

            summary[op_type]["count"] += 1
            summary[op_type]["last_status"] = log_record.get("status")
            summary[op_type]["last_timestamp"] = log_record.get("completed_at")

        return {
            "project_id": project_id,
            "total_operations": len(logs),
            "operations_summary": summary,
            "recent_logs": [
                {
                    "operation": log.get("operation_type"),
                    "status": log.get("status"),
                    "duration": f"{log.get('duration_seconds', 0)}s",
                    "timestamp": log.get("completed_at"),
                }
                for log in logs[:5]  # Last 5 operations
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get deployment logs: {e}")
        return {"error": str(e), "message": "Could not retrieve deployment logs"}


def get_adk_agent_context(project_id: str, tool_context: Optional[ToolContext] = None) -> dict:
    """
    Get ADK agent context and memory from infrastructure generation.

    Args:
        project_id: Sirpi project ID
        tool_context: ADK tool context

    Returns:
        Agent context including repository analysis and decisions
    """
    try:
        supabase = get_supabase_service()

        # Get latest generation for this project
        generation = supabase.get_latest_generation_by_project(project_id)

        if not generation:
            return {"message": "No infrastructure generation found for this project"}

        # Extract project context (contains agent analysis)
        project_context = generation.get("project_context", {})

        return {
            "session_id": generation.get("session_id"),
            "template_type": generation.get("template_type"),
            "status": generation.get("status"),
            "repository_analysis": {
                "language": project_context.get("language"),
                "framework": project_context.get("framework"),
                "runtime_version": project_context.get("runtime_version"),
                "package_manager": project_context.get("package_manager"),
                "exposed_port": project_context.get("exposed_port"),
                "dependencies": project_context.get("dependencies", {}),
                "build_command": project_context.get("build_command"),
                "start_command": project_context.get("start_command"),
                "health_check_path": project_context.get("health_check_path"),
                "environment_variables": project_context.get("environment_variables", []),
            },
            "monorepo_info": {
                "is_monorepo": project_context.get("is_monorepo", False),
                "monorepo_type": project_context.get("monorepo_type"),
                "frontend_framework": project_context.get("frontend_framework"),
            },
            "generated_at": generation.get("created_at"),
            "note": "This shows what the AI agents discovered about your repository during infrastructure generation",
        }

    except Exception as e:
        logger.error(f"Failed to get agent context: {e}")
        return {"error": str(e), "message": "Could not retrieve agent context"}
