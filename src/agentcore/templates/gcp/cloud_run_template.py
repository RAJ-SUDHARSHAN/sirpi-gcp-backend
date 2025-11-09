"""
GCP Cloud Run template with dynamic state backend.
Backend is created separately based on bucket availability.
"""

from typing import Dict
from src.agentcore.templates.registry import TemplateGenerator, TemplateMetadata, DeploymentPlatform


class CloudRunTemplateGenerator:
    """Generate Terraform for GCP Cloud Run deployment."""
    
    def generate(
        self,
        analysis_result,  # AnalysisResult from code_analyzer_agent
        project_id: str,
        repo_full_name: str | None = None,
        gcp_project_id: str | None = None,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Generate Cloud Run Terraform files with fully dynamic configuration.
        All values are parameterized - no hardcoded names or ports.
        """
        # Get dynamic values from analysis
        port = analysis_result.exposed_port or 8080
        health_path = analysis_result.health_check_path or "/"

        # Use central registry from config (same as build step)
        from src.core.config import settings

        registry_repo = settings.gcp_artifact_registry_repository

        files = {}

        # main.tf WITHOUT backend (will be added dynamically in backend.tf)
        # Image name comes from var.app_name which will be set to project name at deploy time
        files["main.tf"] = f'''terraform {{
  required_version = ">= 1.9.0"

  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
  }}
}}

provider "google" {{
  project = var.project_id
  region  = var.region
}}

# Artifact Registry Repository - Reference existing shared registry
# Build step creates this repository if it doesn't exist
data "google_artifact_registry_repository" "main" {{
  location      = var.region
  repository_id = "{registry_repo}"
}}

# Cloud Run Service
resource "google_cloud_run_v2_service" "main" {{
  name     = var.app_name
  location = var.region

  template {{
    containers {{
      # Image name matches the project name set during build
      # Format: REGION-docker.pkg.dev/PROJECT_ID/REGISTRY/APP_NAME:TAG
      image = "${{var.image_uri}}"
      
      ports {{
        container_port = {port}
      }}
      
      # User-provided environment variables
      # Note: PORT is automatically set by Cloud Run to match container_port
      dynamic "env" {{
        for_each = var.app_env_vars != null ? var.app_env_vars : {{}}
        content {{
          name  = env.key
          value = env.value
        }}
      }}
      
      resources {{
        limits = {{
          cpu    = "1000m"
          memory = "512Mi"
        }}
      }}

      # Health check configuration
      startup_probe {{
        http_get {{
          path = "{health_path}"
          port = {port}
        }}
        initial_delay_seconds = 30
        period_seconds        = 10
        timeout_seconds       = 5
        failure_threshold     = 5
      }}
    }}
    
    scaling {{
      min_instance_count = 0
      max_instance_count = 10
    }}
  }}
  
  traffic {{
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }}
  
  lifecycle {{
    ignore_changes = [
      template[0].containers[0].image,  # Allow image updates
    ]
  }}
}}

# IAM policy for public access
resource "google_cloud_run_v2_service_iam_member" "public" {{
  location = google_cloud_run_v2_service.main.location
  name     = google_cloud_run_v2_service.main.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}}
'''
        
        # variables.tf - All values parameterized for flexibility
        files["variables.tf"] = '''variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "app_name" {
  description = "Application name (must match Docker image name)"
  type        = string
}

variable "image_uri" {
  description = "Full Docker image URI from Artifact Registry"
  type        = string
}

variable "app_env_vars" {
  description = "Application environment variables"
  type        = map(string)
  default     = null
}
'''
        
        # outputs.tf
        files["outputs.tf"] = """output "service_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.main.uri
}

output "artifact_registry" {
  description = "Artifact Registry repository"
  value       = data.google_artifact_registry_repository.main.name
}

output "service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_v2_service.main.name
}

output "location" {
  description = "Deployment location"
  value       = google_cloud_run_v2_service.main.location
}
"""
        
        return files
    
    def get_metadata(self) -> TemplateMetadata:
        """Template metadata."""
        return TemplateMetadata(
            name="GCP Cloud Run",
            platform=DeploymentPlatform.GCP_CLOUD_RUN,
            cloud_provider="gcp",
            description="Fully managed serverless container platform with auto-scaling to zero",
            requires_load_balancer=False,
            requires_container_registry=True,
            supports_autoscaling=True,
            min_cost_estimate_monthly=0.0,
            difficulty="beginner",
        )


# Singleton instance
cloud_run_template = CloudRunTemplateGenerator()
