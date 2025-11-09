"""
GCP Deployment Service - Cloud Run deployments via E2B sandboxes.
Handles Docker builds, Artifact Registry pushes, and Terraform deployments.
"""

import logging
import asyncio
import json
from typing import Optional, Callable, Dict, Any
from google.oauth2.credentials import Credentials

from src.services.deployment.sandbox_manager import SandboxManager
from src.api.gcp_auth import get_gcp_credentials
from src.core.config import settings

logger = logging.getLogger(__name__)


class GCPDeploymentService:
    """
    GCP deployment service using E2B sandboxes.
    Deploys applications to Google Cloud Run.
    """

    def __init__(self, sandbox_manager: SandboxManager):
        """
        Initialize GCP deployment service.

        Args:
            sandbox_manager: Unified E2B sandbox manager
        """
        self.sandbox = sandbox_manager

    async def build_and_push_image(
        self,
        user_id: str,
        gcp_project_id: str,
        repository_url: str,
        branch: str,
        image_name: str,
        dockerfile_content: str,
        log_callback: Optional[Callable[[str], None]] = None,
        build_args: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build Docker image and push to Artifact Registry.

        Args:
            user_id: User's Clerk ID
            gcp_project_id: GCP project ID
            repository_url: GitHub repository URL
            branch: Git branch
            image_name: Name for the Docker image
            dockerfile_content: Dockerfile content (from generated files)
            log_callback: Optional callback for streaming logs
            build_args: Optional build arguments for Docker build (e.g., env vars for Next.js)

        Returns:
            Full image URI (e.g., us-central1-docker.pkg.dev/PROJECT/REPO/IMAGE:TAG)
        """
        if log_callback:
            self.sandbox.set_log_callback(log_callback)

        try:
            # Clone repository
            repo_dir = await self.sandbox.clone_repository(repository_url, branch)

            # Write Dockerfile
            dockerfile_path = f"{repo_dir}/Dockerfile"
            await self.sandbox.write_file(dockerfile_path, dockerfile_content)

            # Build Docker image with build args if provided
            await self.sandbox.build_docker_image(
                dockerfile_path=dockerfile_path,
                image_name=image_name,
                context_dir=repo_dir,
                build_args=build_args,
            )

            # Configure gcloud AND Terraform credentials
            await self._configure_gcloud_and_terraform(user_id, gcp_project_id)

            # Get fresh credentials for Python SDK operations
            credentials = get_gcp_credentials(user_id, gcp_project_id)

            # Enable required GCP APIs (using Python SDK!)
            await self._ensure_required_apis(gcp_project_id, credentials)

            # Define registry details
            registry_location = settings.gcp_artifact_registry_location
            registry_name = settings.gcp_artifact_registry_repository

            # Create Artifact Registry repository using Python SDK
            from google.cloud import artifactregistry_v1

            # Use Python SDK to create registry
            registry_client = artifactregistry_v1.ArtifactRegistryClient(credentials=credentials)
            parent = f"projects/{gcp_project_id}/locations/{registry_location}"
            repository_path = f"{parent}/repositories/{registry_name}"

            try:
                # Check if exists
                registry_client.get_repository(name=repository_path)
                self.sandbox._log(f"✅ Artifact Registry already exists: {registry_name}")
            except Exception as e:
                if "NotFound" in str(e) or "404" in str(e):
                    # Create it
                    self.sandbox._log(f"Creating Artifact Registry: {registry_name}...")

                    repository = artifactregistry_v1.Repository()
                    repository.format_ = artifactregistry_v1.Repository.Format.DOCKER
                    repository.description = "Sirpi deployment repository"

                    operation = registry_client.create_repository(
                        parent=parent, repository_id=registry_name, repository=repository
                    )

                    import asyncio

                    await asyncio.to_thread(operation.result, timeout=60)

                    self.sandbox._log(f"✅ Created Artifact Registry: {registry_name}")
                else:
                    raise

            # Tag image for Artifact Registry
            full_image_uri = (
                f"{registry_location}-docker.pkg.dev/{gcp_project_id}/{registry_name}/{image_name}"
            )

            await self.sandbox.run_command(
                f"docker tag {image_name} {full_image_uri}", stream_output=True
            )

            # Configure Docker for Artifact Registry using OAuth access token
            # ALWAYS use OAuth token (no gcloud needed!)
            self.sandbox._log("Configuring Docker with OAuth token...")

            docker_login_cmd = f"echo '{self.gcloud_access_token}' | docker login -u oauth2accesstoken --password-stdin https://{registry_location}-docker.pkg.dev"
            login_result = await self.sandbox.run_command(docker_login_cmd, stream_output=False)

            if login_result["exit_code"] != 0:
                raise RuntimeError(f"Docker login failed: {login_result['stderr']}")

            self.sandbox._log(f"✅ Docker authenticated for Artifact Registry")

            # Push to Artifact Registry
            self.sandbox._log(f"Pushing image to Artifact Registry...")
            push_result = await self.sandbox.run_command(
                f"docker push {full_image_uri}", stream_output=True
            )

            if push_result["exit_code"] != 0:
                raise RuntimeError(f"Failed to push image: {push_result['stderr']}")

            self.sandbox._log(f"✅ Image pushed successfully: {full_image_uri}")

            return full_image_uri

        except Exception as e:
            logger.error(f"Build and push failed: {e}")
            raise

    async def _configure_gcloud_and_terraform(self, user_id: str, project_id: str):
        """
        Configure Terraform credentials using ADC (Application Default Credentials).
        SIMPLE: Just create ADC file - Terraform auto-discovers it!
        """
        try:
            self.sandbox._log("Configuring authentication with OAuth + ADC...")

            # Get OAuth credentials (auto-refreshed if expired)
            credentials = get_gcp_credentials(user_id, project_id)

            from google.auth.transport.requests import Request

            # Ensure we have a valid access token
            if not credentials.token or credentials.expired:
                credentials.refresh(Request())

            # Create ADC file for Terraform + Python SDKs
            adc_file = "/home/user/gcp-adc.json"
            adc_content = json.dumps(
                {
                    "type": "authorized_user",
                    "client_id": settings.gcp_oauth_client_id,
                    "client_secret": settings.gcp_oauth_client_secret,
                    "refresh_token": credentials.refresh_token,
                    "quota_project_id": project_id,
                }
            )

            await self.sandbox.write_file(adc_file, adc_content)

            # Store for Terraform env var
            self.terraform_creds_file = adc_file
            # Store access token for Docker login
            self.gcloud_access_token = credentials.token

            self.sandbox._log(f"✅ ADC configured for Terraform + Python SDKs")

        except Exception as e:
            logger.error(f"Failed to configure credentials: {e}")
            raise

    async def _ensure_required_apis(self, project_id: str, credentials):
        """Enable required GCP APIs using Python SDK (no gcloud needed!)."""
        required_apis = [
            "artifactregistry.googleapis.com",
            "run.googleapis.com",
            "compute.googleapis.com",
        ]

        self.sandbox._log("Checking required GCP APIs...")

        try:
            from google.cloud import service_usage_v1
            from google.api_core import exceptions

            client = service_usage_v1.ServiceUsageClient(credentials=credentials)

            for api in required_apis:
                try:
                    service_name = f"projects/{project_id}/services/{api}"

                    # Check if enabled
                    service = client.get_service(name=service_name)

                    if service.state == service_usage_v1.State.ENABLED:
                        self.sandbox._log(f"✅ {api} already enabled")
                    else:
                        # Enable the service
                        self.sandbox._log(f"Enabling {api}...")
                        operation = client.enable_service(name=service_name)

                        # Wait for operation (async)
                        import asyncio

                        await asyncio.to_thread(operation.result, timeout=60)

                        self.sandbox._log(f"✅ Enabled {api}")

                except exceptions.AlreadyExists:
                    self.sandbox._log(f"✅ {api} already enabled")

                except Exception as e:
                    # Don't fail deployment - just warn
                    self.sandbox._log(f"⚠️ Could not check {api}: {str(e)[:100]}")

        except Exception as e:
            # API checking failed - log but continue
            logger.warning(f"Could not check APIs: {e}")
            self.sandbox._log("⚠️ Could not verify APIs (non-fatal)")

    async def _ensure_artifact_registry(
        self, user_id: str, project_id: str, location: str, repository_name: str
    ):
        """Create Artifact Registry repository if it doesn't exist."""
        try:
            # Use Python SDK to create registry (no gcloud needed!)
            from google.cloud import artifactregistry_v1

            # Get credentials
            credentials = get_gcp_credentials(user_id, project_id)

            registry_client = artifactregistry_v1.ArtifactRegistryClient(credentials=credentials)

            parent = f"projects/{project_id}/locations/{location}"
            repository_path = f"{parent}/repositories/{repository_name}"

            try:
                # Try to get repository
                repository = registry_client.get_repository(name=repository_path)
                self.sandbox._log(f"✅ Artifact Registry already exists: {repository_name}")
                return

            except Exception as e:
                # Repository doesn't exist, create it
                if "NotFound" not in str(e) and "404" not in str(e):
                    raise

                self.sandbox._log(f"Creating Artifact Registry: {repository_name}...")

                repository = artifactregistry_v1.Repository()
                repository.format_ = artifactregistry_v1.Repository.Format.DOCKER
                repository.description = "Sirpi deployment repository"

                operation = registry_client.create_repository(
                    parent=parent, repository_id=repository_name, repository=repository
                )

                # Wait for operation (async)
                import asyncio

                await asyncio.to_thread(operation.result, timeout=60)

                self.sandbox._log(f"✅ Created Artifact Registry: {repository_name}")

        except Exception as e:
            logger.error(f"Failed to ensure Artifact Registry: {e}")
            raise


def get_gcp_deployment_service(sandbox_manager: SandboxManager) -> GCPDeploymentService:
    """Get GCP deployment service instance."""
    return GCPDeploymentService(sandbox_manager)
