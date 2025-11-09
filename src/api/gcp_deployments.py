"""
GCP Deployment API Endpoints.
Handles build, plan, apply, and destroy operations for Google Cloud Run.
"""

import logging
import json
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional

from src.services.deployment.sandbox_manager import SandboxManager
from src.services.deployment.gcp_deployment import GCPDeploymentService
from src.services.supabase import supabase
from src.services.gcs_storage import get_gcs_storage
from src.utils.clerk_auth import get_current_user_id
from src.core.config import settings
from src.api.deployment_logs import register_deployment, get_log_callback, send_log, send_completion
from src.api.env_vars import get_decrypted_env_vars
from src.utils.gcp_credentials_validator import check_gcp_credentials
import asyncio

router = APIRouter(prefix="/gcp/deployment", tags=["GCP Deployment"])
logger = logging.getLogger(__name__)


@router.post("/projects/{project_id}/build_image")
async def build_gcp_image(
    project_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    """
    Build Docker image and push to Google Artifact Registry.
    """
    import time

    start_time = time.time()

    try:
        # Validate GCP credentials first
        cred_status = check_gcp_credentials(user_id)

        if cred_status["needs_reconnect"]:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "gcp_credentials_expired",
                    "message": cred_status["message"],
                    "action_required": "reconnect_gcp",
                },
            )

        # Get project
        project = supabase.get_project_by_id(project_id)
        if not project or project["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get latest generation
        generation = supabase.get_latest_generation_by_project(project_id)
        if not generation:
            raise HTTPException(status_code=404, detail="No generation found for project")

        # Get GCP credentials
        gcp_creds = supabase.get_gcp_credentials(user_id)
        if not gcp_creds:
            raise HTTPException(
                status_code=400,
                detail="GCP credentials not found. Please connect your GCP account.",
            )

        gcp_project_id = gcp_creds["project_id"]

        # Get generated Dockerfile from GCS
        gcs = get_gcs_storage()
        owner, repo = project["repository_name"].split("/")
        files = await gcs.get_repository_files(owner, repo)

        dockerfile = next((f for f in files if f.get("path") == "Dockerfile"), None)
        if not dockerfile:
            raise HTTPException(status_code=404, detail="Dockerfile not found in generated files")

        # Register deployment for SSE streaming
        register_deployment(project_id)
        await send_log(project_id, "Starting Docker image build...")

        # Create sandbox and deployment service
        sandbox = SandboxManager(template_id=settings.e2b_template_id)
        gcp_service = GCPDeploymentService(sandbox)

        # Build and push image
        image_name = f"{project['name'].lower()}:latest"

        # Collect logs for database storage
        collected_logs = []

        def collecting_callback(message: str):
            """Callback that both sends to SSE and collects for database."""
            collected_logs.append(message)
            # Also send to SSE
            asyncio.create_task(send_log(project_id, message))

        async with sandbox:
            # Set log callback FIRST before any operations
            sandbox.set_log_callback(collecting_callback)

            # Get default branch from project (fallback to 'main' if not set)
            branch = project.get("default_branch", "main")
            
            # Check if Dockerfile has ARG declarations and prepare build args
            build_args = None
            dockerfile_content = dockerfile["content"]
            
            # Detect ARG declarations in Dockerfile (case-insensitive)
            import re
            arg_pattern = re.compile(r'^ARG\s+([A-Z_][A-Z0-9_]*)', re.MULTILINE | re.IGNORECASE)
            declared_args = arg_pattern.findall(dockerfile_content)
            
            if declared_args:
                await send_log(project_id, f"📋 Detected {len(declared_args)} ARG declarations in Dockerfile")
                
                # Get environment variables from database
                env_vars = await get_decrypted_env_vars(project_id)
                
                # Filter env vars to only those declared as ARGs in Dockerfile
                build_args = {key: value for key, value in env_vars.items() if key in declared_args}
                
                if build_args:
                    await send_log(project_id, f"🔧 Passing {len(build_args)} build arguments to Docker")
                else:
                    await send_log(project_id, "⚠️  No matching environment variables found for declared ARGs")
            
            image_uri = await gcp_service.build_and_push_image(
                user_id=user_id,
                gcp_project_id=gcp_project_id,
                repository_url=project["repository_url"],
                branch=branch,
                image_name=image_name,
                dockerfile_content=dockerfile_content,
                log_callback=collecting_callback,
                build_args=build_args,
            )

        # Send completion signal to close the stream
        await send_completion(project_id)

        # Save build logs IMMEDIATELY (before sleep to avoid race condition)
        build_duration = int(time.time() - start_time)

        supabase.save_deployment_logs(
            project_id=project_id,
            operation_type="build_image",
            logs=collected_logs,
            status="success",
            duration_seconds=build_duration,
            metadata={"image_uri": image_uri},  # Store image URI directly
        )

        # IMPORTANT: Give time for all SSE messages to be sent before returning
        logger.info("Waiting for log queue to flush...")
        await asyncio.sleep(5)

        return {
            "success": True,
            "data": {
                "operation_id": f"build_{project_id}",
                "image_uri": image_uri,
                "message": "Image built and pushed to Artifact Registry",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Build image failed: {e}", exc_info=True)

        # Save error logs
        supabase.save_deployment_logs(
            project_id=project_id,
            operation_type="build_image",
            logs=[f"Build failed: {str(e)}"],
            status="error",
            error_message=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Build failed: {str(e)}")


@router.post("/projects/{project_id}/plan")
async def plan_gcp_deployment(
    project_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    """
    Generate Terraform plan for Cloud Run deployment.
    """
    try:
        # Validate GCP credentials first
        cred_status = check_gcp_credentials(user_id)

        if cred_status["needs_reconnect"]:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "gcp_credentials_expired",
                    "message": cred_status["message"],
                    "action_required": "reconnect_gcp",
                },
            )

        # Get project and generation
        project = supabase.get_project_by_id(project_id)
        if not project or project["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Project not found")

        generation = supabase.get_latest_generation_by_project(project_id)
        if not generation:
            raise HTTPException(status_code=404, detail="No generation found")

        # Get Terraform files from GCS
        gcs = get_gcs_storage()
        owner, repo = project["repository_name"].split("/")
        files = await gcs.get_repository_files(owner, repo)

        tf_files = {f["path"]: f["content"] for f in files if f["path"].endswith(".tf")}

        if not tf_files:
            raise HTTPException(status_code=404, detail="Terraform files not found")

        # Get GCP credentials
        gcp_creds = supabase.get_gcp_credentials(user_id)
        gcp_project_id = gcp_creds["project_id"]

        # Get image URI from build logs (check metadata first for faster access)
        build_logs = supabase.get_deployment_logs(project_id, "build_image")
        if not build_logs:
            raise HTTPException(
                status_code=400, detail="Image not built yet. Please build the image first."
            )

        # Try to get image URI from metadata first (faster and more reliable)
        image_uri = None
        if build_logs.get("metadata") and build_logs["metadata"].get("image_uri"):
            image_uri = build_logs["metadata"]["image_uri"]
        else:
            # Fallback: Extract from logs if metadata not available
            if build_logs.get("logs"):
                for log in build_logs["logs"]:
                    if "docker.pkg.dev" in log:
                        import re

                        match = re.search(r"([a-z0-9-]+\-docker\.pkg\.dev/[^\s]+)", log)
                        if match:
                            image_uri = match.group(1)
                            break

        if not image_uri:
            raise HTTPException(
                status_code=400, detail="Could not find image URI. Please rebuild the image."
            )

        # Get environment variables from database (decrypted)
        env_vars = await get_decrypted_env_vars(project_id)
        logger.info(f"Retrieved {len(env_vars)} environment variables for deployment")

        # Register for SSE streaming
        register_deployment(project_id)
        await send_log(project_id, "Starting Terraform planning...")

        # Create sandbox
        sandbox = SandboxManager(template_id=settings.e2b_template_id)
        gcp_service = GCPDeploymentService(sandbox)

        # Collect logs for database storage
        collected_logs = []

        def collecting_callback(message: str):
            """Callback that both sends to SSE and collects for database."""
            collected_logs.append(message)
            asyncio.create_task(send_log(project_id, message))

        async with sandbox:
            # Set log callback FIRST before any operations
            sandbox.set_log_callback(collecting_callback)

            # Configure gcloud AND Terraform credentials
            await gcp_service._configure_gcloud_and_terraform(user_id, gcp_project_id)

            # Write Terraform files
            tf_dir = "/home/user/terraform"
            await sandbox.run_command(f"mkdir -p {tf_dir}", stream_output=False)

            sandbox._log("Writing Terraform configuration...")

            # Check if any TF file already has a backend block
            has_backend_in_files = False
            for filename, content in tf_files.items():
                if 'backend "gcs"' in content or 'backend "s3"' in content:
                    has_backend_in_files = True
                    sandbox._log(f"⚠️ Found backend configuration in {filename}, removing it...")
                    # Remove backend block from content
                    import re

                    content = re.sub(r'backend\s+"[^"]+"\s*\{[^}]*\}', "", content, flags=re.DOTALL)

                await sandbox.write_file(f"{tf_dir}/{filename}", content)
                sandbox._log(f"Wrote {filename}")

            # Ensure GCS state bucket exists FIRST (using Python SDK - no gcloud!)
            from src.services.deployment.gcs_state_manager import ensure_gcs_state_bucket
            from src.api.gcp_auth import get_gcp_credentials

            # Get fresh credentials (auto-refreshed)
            credentials = get_gcp_credentials(user_id, gcp_project_id)
            service_name = project["name"].lower().replace("_", "-")
            bucket_name = await ensure_gcs_state_bucket(
                credentials, gcp_project_id, service_name, sandbox
            )

            # Create backend.tf with the bucket (now safe - no duplicates)
            backend_content = f'''terraform {{
  backend "gcs" {{
    bucket = "{bucket_name}"
    prefix = "projects/{service_name}"
  }}
}}
'''
            await sandbox.write_file(f"{tf_dir}/backend.tf", backend_content)
            sandbox._log(f"✅ Configured GCS backend: gs://{bucket_name}/projects/{service_name}")

            # Create tfvars file with CORRECT variable names matching the template
            service_name = project["name"].lower().replace("_", "-")

            # Format env vars as Terraform map
            env_vars_tf = ""
            if env_vars:
                env_items = [f'  {k} = "{v}"' for k, v in env_vars.items()]
                env_vars_tf = f"""app_env_vars = {{
{chr(10).join(env_items)}
}}"""
            else:
                env_vars_tf = "app_env_vars = null"

            tfvars_content = f"""# Generated by Sirpi
project_id = "{gcp_project_id}"
region     = "{settings.gcp_cloud_run_region}"
app_name   = "{service_name}"
image_uri  = "{image_uri}"
{env_vars_tf}
"""

            await sandbox.write_file(f"{tf_dir}/terraform.tfvars", tfvars_content)
            sandbox._log(f"Created terraform.tfvars with {len(env_vars)} environment variables")

            # Get Terraform credentials env var
            terraform_env = {}
            if hasattr(gcp_service, "terraform_creds_file"):
                terraform_env["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_service.terraform_creds_file

            # Initialize Terraform
            sandbox._log("Initializing Terraform...")
            await sandbox.run_terraform("init", tf_dir, env_vars=terraform_env)

            # Generate plan
            sandbox._log("Generating deployment plan...")
            plan_result = await sandbox.run_terraform(
                "plan", tf_dir, var_file="terraform.tfvars", env_vars=terraform_env
            )

            # Add the terraform plan output to collected logs
            if plan_result["stdout"]:
                for line in plan_result["stdout"].split("\n"):
                    if line.strip():
                        collected_logs.append(line)

            sandbox._log("✅ Terraform plan generated successfully!")

        # Send completion signal to close the stream
        await send_completion(project_id)

        # Give time for logs to flush
        await asyncio.sleep(3)

        # Save plan logs
        supabase.save_deployment_logs(
            project_id=project_id, operation_type="plan", logs=collected_logs, status="success"
        )

        return {
            "success": True,
            "data": {
                "operation_id": f"plan_{project_id}",
                "plan_output": collected_logs[-50:],  # Last 50 lines for response
                "env_var_count": len(env_vars),
                "message": "Terraform plan generated successfully",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Plan failed: {str(e)}")


@router.post("/projects/{project_id}/apply")
async def apply_gcp_deployment(
    project_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    """
    Deploy infrastructure to Google Cloud Run.
    """
    import time

    start_time = time.time()

    try:
        # Validate GCP credentials first
        cred_status = check_gcp_credentials(user_id)

        if cred_status["needs_reconnect"]:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "gcp_credentials_expired",
                    "message": cred_status["message"],
                    "action_required": "reconnect_gcp",
                },
            )

        # Get project
        project = supabase.get_project_by_id(project_id)
        if not project or project["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get Terraform files from GCS
        gcs = get_gcs_storage()
        owner, repo = project["repository_name"].split("/")
        files = await gcs.get_repository_files(owner, repo)

        tf_files = {f["path"]: f["content"] for f in files if f["path"].endswith(".tf")}

        # Get GCP credentials
        gcp_creds = supabase.get_gcp_credentials(user_id)
        gcp_project_id = gcp_creds["project_id"]

        # Get image URI from build logs (check metadata first for faster access)
        build_logs = supabase.get_deployment_logs(project_id, "build_image")
        if not build_logs:
            raise HTTPException(
                status_code=400, detail="Image not built yet. Please build the image first."
            )

        # Try to get image URI from metadata first (faster and more reliable)
        image_uri = None
        if build_logs.get("metadata") and build_logs["metadata"].get("image_uri"):
            image_uri = build_logs["metadata"]["image_uri"]
        else:
            # Fallback: Extract from logs if metadata not available
            for log in build_logs.get("logs", []):
                if "docker.pkg.dev" in log:
                    import re

                    match = re.search(r"([a-z0-9-]+\-docker\.pkg\.dev/[^\s]+)", log)
                    if match:
                        image_uri = match.group(1)
                        break

        if not image_uri:
            raise HTTPException(
                status_code=400, detail="Could not find image URI. Please rebuild the image."
            )

        # Get environment variables
        env_vars = await get_decrypted_env_vars(project_id)
        logger.info(f"Retrieved {len(env_vars)} environment variables for deployment")

        # Register for SSE streaming
        register_deployment(project_id)
        await send_log(project_id, "Starting deployment...")

        # Create sandbox
        sandbox = SandboxManager(template_id=settings.e2b_template_id)
        gcp_service = GCPDeploymentService(sandbox)
        log_callback = get_log_callback(project_id)

        # Collect logs
        collected_logs = []

        def collecting_callback(message: str):
            """Callback that both sends to SSE and collects for database."""
            collected_logs.append(message)
            asyncio.create_task(send_log(project_id, message))

        async with sandbox:
            # Set log callback FIRST before any operations
            sandbox.set_log_callback(collecting_callback)

            # Configure gcloud AND Terraform credentials
            await gcp_service._configure_gcloud_and_terraform(user_id, gcp_project_id)

            # Write Terraform files
            tf_dir = "/home/user/terraform"
            await sandbox.run_command(f"mkdir -p {tf_dir}", stream_output=False)

            # Check if any TF file already has a backend block and remove it
            for filename, content in tf_files.items():
                if 'backend "gcs"' in content or 'backend "s3"' in content:
                    sandbox._log(f"⚠️ Found backend configuration in {filename}, removing it...")
                    import re

                    content = re.sub(r'backend\s+"[^"]+"\s*\{[^}]*\}', "", content, flags=re.DOTALL)

                await sandbox.write_file(f"{tf_dir}/{filename}", content)

            # Ensure GCS state bucket exists FIRST (using Python SDK!)
            from src.services.deployment.gcs_state_manager import ensure_gcs_state_bucket
            from src.api.gcp_auth import get_gcp_credentials

            credentials = get_gcp_credentials(user_id, gcp_project_id)
            service_name = project["name"].lower().replace("_", "-")
            bucket_name = await ensure_gcs_state_bucket(
                credentials, gcp_project_id, service_name, sandbox
            )

            # Create backend.tf (now safe - no duplicates)
            backend_content = f'''terraform {{
  backend "gcs" {{
    bucket = "{bucket_name}"
    prefix = "projects/{service_name}"
  }}
}}
'''
            await sandbox.write_file(f"{tf_dir}/backend.tf", backend_content)
            sandbox._log(f"✅ Configured GCS backend: gs://{bucket_name}/projects/{service_name}")

            # Create tfvars
            service_name = project["name"].lower().replace("_", "-")
            env_vars_tf = ""
            if env_vars:
                env_items = [f'  {k} = "{v}"' for k, v in env_vars.items()]
                env_vars_tf = f"""app_env_vars = {{
{chr(10).join(env_items)}
}}"""
            else:
                env_vars_tf = "app_env_vars = null"

            tfvars_content = f"""project_id = "{gcp_project_id}"
region     = "{settings.gcp_cloud_run_region}"
app_name   = "{service_name}"
image_uri  = "{image_uri}"
{env_vars_tf}
"""
            await sandbox.write_file(f"{tf_dir}/terraform.tfvars", tfvars_content)

            # Get Terraform credentials env var
            terraform_env = {}
            if hasattr(gcp_service, "terraform_creds_file"):
                terraform_env["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_service.terraform_creds_file

            # Init and apply
            sandbox._log("Initializing Terraform...")
            await sandbox.run_terraform("init", tf_dir, env_vars=terraform_env)

            sandbox._log("Applying infrastructure...")
            apply_result = await sandbox.run_terraform(
                "apply",
                tf_dir,
                var_file="terraform.tfvars",
                auto_approve=True,
                env_vars=terraform_env,
            )

            # Get outputs (need ADC credentials for this too!)
            sandbox._log("Retrieving deployment outputs...")
            output_cmd = f"GOOGLE_APPLICATION_CREDENTIALS={terraform_env['GOOGLE_APPLICATION_CREDENTIALS']} terraform output -json"
            output_result = await sandbox.run_command(
                output_cmd, working_dir=tf_dir, stream_output=False
            )

            outputs = {}
            if output_result["stdout"]:
                outputs = json.loads(output_result["stdout"])

            sandbox._log("✅ Deployment completed successfully!")

        # Send completion signal to close the stream
        await send_completion(project_id)

        # Save apply logs and outputs
        await asyncio.sleep(3)

        apply_duration = int(time.time() - start_time)

        supabase.save_deployment_logs(
            project_id=project_id,
            operation_type="apply",
            logs=collected_logs,
            status="success",
            duration_seconds=apply_duration,
        )

        # Extract service URL from outputs
        service_url = None
        if "service_url" in outputs:
            service_url = outputs["service_url"].get("value")

        # Save outputs and URL to project
        if outputs:
            supabase.save_terraform_outputs(project_id, outputs)

        if service_url:
            supabase.update_application_url(project_id, service_url)

        # Update project deployment status
        supabase.update_project_deployment_status(project_id, "deployed")

        return {
            "success": True,
            "data": {
                "operation_id": f"apply_{project_id}",
                "service_url": service_url,
                "outputs": outputs,
                "message": "Deployed to Cloud Run successfully",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deploy failed: {e}", exc_info=True)

        # Save error logs
        supabase.save_deployment_logs(
            project_id=project_id,
            operation_type="apply",
            logs=collected_logs if "collected_logs" in locals() else [f"Deploy failed: {str(e)}"],
            status="error",
            error_message=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Deploy failed: {str(e)}")


@router.post("/projects/{project_id}/destroy")
async def destroy_gcp_deployment(
    project_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    """
    Destroy Cloud Run infrastructure using Terraform.
    Properly uses OAuth credentials from database (has full permissions!).
    """
    import time

    start_time = time.time()

    try:
        # Validate GCP credentials first
        cred_status = check_gcp_credentials(user_id)

        if cred_status["needs_reconnect"]:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "gcp_credentials_expired",
                    "message": cred_status["message"],
                    "action_required": "reconnect_gcp",
                },
            )

        # Get project
        project = supabase.get_project_by_id(project_id)
        if not project or project["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get Terraform files from GCS
        gcs = get_gcs_storage()
        owner, repo = project["repository_name"].split("/")
        files = await gcs.get_repository_files(owner, repo)

        tf_files = {f["path"]: f["content"] for f in files if f["path"].endswith(".tf")}

        if not tf_files:
            raise HTTPException(status_code=404, detail="Terraform files not found")

        # Get GCP credentials
        gcp_creds = supabase.get_gcp_credentials(user_id)
        gcp_project_id = gcp_creds["project_id"]

        # Get env vars (needed for tfvars to match plan/apply)
        env_vars = await get_decrypted_env_vars(project_id)

        # Get image URI from build logs (optional for destroy, but needed for tfvars consistency)
        build_logs = supabase.get_deployment_logs(project_id, "build_image")
        image_uri = "placeholder:latest"  # Default fallback for destroy
        if build_logs:
            if build_logs.get("metadata") and build_logs["metadata"].get("image_uri"):
                image_uri = build_logs["metadata"]["image_uri"]
            else:
                # Fallback: Extract from logs
                for log in build_logs.get("logs", []):
                    if "docker.pkg.dev" in log:
                        import re

                        match = re.search(r"([a-z0-9-]+\-docker\.pkg\.dev/[^\s]+)", log)
                        if match:
                            image_uri = match.group(1)
                            break

        # Register for SSE streaming
        register_deployment(project_id)
        await send_log(project_id, "Starting infrastructure destruction...")

        # Create sandbox
        sandbox = SandboxManager(template_id=settings.e2b_template_id)
        gcp_service = GCPDeploymentService(sandbox)
        log_callback = get_log_callback(project_id)

        # Collect logs
        collected_logs = []

        def collecting_callback(message: str):
            """Callback for SSE and database."""
            collected_logs.append(message)
            asyncio.create_task(send_log(project_id, message))

        async with sandbox:
            # Set log callback FIRST before any operations
            sandbox.set_log_callback(collecting_callback)

            # Configure credentials
            await gcp_service._configure_gcloud_and_terraform(user_id, gcp_project_id)

            # Write Terraform files
            tf_dir = "/home/user/terraform"
            await sandbox.run_command(f"mkdir -p {tf_dir}", stream_output=False)

            sandbox._log("Writing Terraform configuration...")

            # Check if any TF file already has a backend block and remove it
            for filename, content in tf_files.items():
                if 'backend "gcs"' in content or 'backend "s3"' in content:
                    sandbox._log(f"⚠️ Found backend configuration in {filename}, removing it...")
                    import re

                    content = re.sub(r'backend\s+"[^"]+"\s*\{[^}]*\}', "", content, flags=re.DOTALL)

                await sandbox.write_file(f"{tf_dir}/{filename}", content)

            # Ensure GCS state bucket exists (using Python SDK!)
            from src.services.deployment.gcs_state_manager import (
                ensure_gcs_state_bucket,
                cleanup_terraform_state,
            )
            from src.api.gcp_auth import get_gcp_credentials

            credentials = get_gcp_credentials(user_id, gcp_project_id)
            service_name = project["name"].lower().replace("_", "-")
            bucket_name = await ensure_gcs_state_bucket(
                credentials, gcp_project_id, service_name, sandbox
            )

            # Create backend.tf (now safe - no duplicates)
            backend_content = f'''terraform {{
  backend "gcs" {{
    bucket = "{bucket_name}"
    prefix = "projects/{service_name}"
  }}
}}
'''
            await sandbox.write_file(f"{tf_dir}/backend.tf", backend_content)
            sandbox._log(f"✅ Configured GCS backend: gs://{bucket_name}/projects/{service_name}")

            # Create tfvars (same as plan/apply for consistency)
            service_name = project["name"].lower().replace("_", "-")
            env_vars_tf = ""
            if env_vars:
                env_items = [f'  {k} = "{v}"' for k, v in env_vars.items()]
                env_vars_tf = f"""app_env_vars = {{
{chr(10).join(env_items)}
}}"""
            else:
                env_vars_tf = "app_env_vars = null"

            tfvars_content = f"""project_id = "{gcp_project_id}"
region     = "{settings.gcp_cloud_run_region}"
app_name   = "{service_name}"
image_uri  = "{image_uri}"
{env_vars_tf}
"""
            await sandbox.write_file(f"{tf_dir}/terraform.tfvars", tfvars_content)

            # Get Terraform credentials
            terraform_env = {}
            if hasattr(gcp_service, "terraform_creds_file"):
                terraform_env["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_service.terraform_creds_file

            # Init and destroy
            sandbox._log("Initializing Terraform...")
            await sandbox.run_terraform("init", tf_dir, env_vars=terraform_env)

            sandbox._log("Destroying infrastructure...")
            destroy_result = await sandbox.run_terraform(
                "destroy",
                tf_dir,
                var_file="terraform.tfvars",
                auto_approve=True,
                env_vars=terraform_env,
            )

            # Clean up Terraform state from GCS (using Python SDK!)
            await cleanup_terraform_state(credentials, gcp_project_id, service_name, sandbox)

            sandbox._log("✅ Infrastructure destroyed successfully!")

        # Send completion signal to close the stream
        await send_completion(project_id)

        # Save destroy logs
        await asyncio.sleep(3)

        destroy_duration = int(time.time() - start_time)

        supabase.save_deployment_logs(
            project_id=project_id,
            operation_type="destroy",
            logs=collected_logs,
            status="success",
            duration_seconds=destroy_duration,
        )

        # Update project status
        supabase.update_project_deployment_status(project_id, "destroyed")

        # Clear application URL and outputs
        supabase.update_application_url(project_id, None)
        supabase.save_terraform_outputs(project_id, {})

        return {
            "success": True,
            "data": {
                "operation_id": f"destroy_{project_id}",
                "message": "Infrastructure destroyed successfully",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Destroy failed: {e}", exc_info=True)

        # Save error logs
        supabase.save_deployment_logs(
            project_id=project_id,
            operation_type="destroy",
            logs=collected_logs if "collected_logs" in locals() else [f"Destroy failed: {str(e)}"],
            status="error",
            error_message=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Destroy failed: {str(e)}")


@router.get("/projects/{project_id}/logs")
async def get_deployment_logs_endpoint(
    project_id: str, user_id: str = Depends(get_current_user_id)
):
    """
    Get all deployment logs for a project.
    Used to restore state on page refresh.
    """
    try:
        # Verify project ownership
        project = supabase.get_project_by_id(project_id)
        if not project or project["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get all deployment logs
        logs = supabase.get_deployment_logs(project_id)

        if not logs:
            return {"success": True, "data": {"logs": []}}

        return {"success": True, "data": {"logs": logs}}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deployment logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/service-info")
async def get_cloud_run_service_info(project_id: str, user_id: str = Depends(get_current_user_id)):
    """Get Cloud Run service information including scaling configuration."""
    try:
        # Verify project ownership
        project = supabase.get_project_by_id(project_id)
        if not project or project["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if deployed
        if project.get("deployment_status") != "deployed":
            return {
                "success": True,
                "data": {"deployed": False, "message": "Service not deployed yet"},
            }

        # Get service name from project
        service_name = project.get("name", "").lower().replace(" ", "-").replace("_", "-")

        # Get GCP credentials
        gcp_creds = supabase.get_gcp_credentials(user_id)
        if not gcp_creds:
            raise HTTPException(status_code=400, detail="GCP credentials not found")

        gcp_project_id = gcp_creds.get("project_id")

        # Get OAuth credentials
        from src.utils.gcp_credentials_validator import _get_gcp_credentials

        credentials = _get_gcp_credentials(user_id, gcp_project_id)

        # Get Cloud Run service
        from google.cloud import run_v2

        client = run_v2.ServicesClient(credentials=credentials)
        service_path = f"projects/{gcp_project_id}/locations/{settings.gcp_cloud_run_region}/services/{service_name}"

        try:
            service = client.get_service(name=service_path)

            # Extract scaling info
            template = service.template
            scaling = template.scaling if template else None

            return {
                "success": True,
                "data": {
                    "deployed": True,
                    "service_name": service_name,
                    "scaling": {
                        "min_instances": scaling.min_instance_count if scaling else 0,
                        "max_instances": scaling.max_instance_count if scaling else 10,
                    },
                    "url": service.uri,
                    "status": "RUNNING" if service.terminal_condition.state == 1 else "UNKNOWN",
                },
            }
        except Exception as e:
            logger.warning(f"Failed to get Cloud Run service: {e}")
            return {
                "success": True,
                "data": {
                    "deployed": False,
                    "error": str(e),
                    "message": "Could not fetch service details",
                },
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
