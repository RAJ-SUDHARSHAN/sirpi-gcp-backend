from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    Request,
)
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel
import logging
import uuid
import asyncio
import json

from src.services.supabase import supabase, DatabaseError
from src.services.github_app import get_github_app, GitHubAppError
from src.utils.clerk_auth import get_current_user_id

router = APIRouter()
logger = logging.getLogger(__name__)


class ImportRepositoryRequest(BaseModel):
    full_name: str
    installation_id: int


class UpdateProjectRequest(BaseModel):
    deployment_status: str | None = None
    aws_role_arn: str | None = None


@router.post("/projects/import")
async def import_repository(
    request: ImportRepositoryRequest, user_id: str = Depends(get_current_user_id)
):
    try:
        parts = request.full_name.split("/")
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid repository name format")

        owner, repo_name = parts

        github = get_github_app()

        try:
            repos = await github.get_installation_repositories(request.installation_id)
        except GitHubAppError:
            raise HTTPException(status_code=502, detail="GitHub API error")

        repo_data = next((r for r in repos if r["full_name"] == request.full_name), None)

        if not repo_data:
            raise HTTPException(status_code=404, detail="Repository not found in installation")

        project_id = str(uuid.uuid4())
        project_slug = repo_name.lower().replace("_", "-").replace(" ", "-")

        try:
            with supabase.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO projects
                        (id, user_id, name, slug, repository_url, repository_name,
                         github_repo_id, installation_id, language, description, status, default_branch)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, slug) DO UPDATE SET
                            github_repo_id = EXCLUDED.github_repo_id,
                            installation_id = EXCLUDED.installation_id,
                            language = EXCLUDED.language,
                            description = EXCLUDED.description,
                            status = EXCLUDED.status,
                            default_branch = EXCLUDED.default_branch,
                            updated_at = NOW()
                        RETURNING id, name, slug, status, created_at
                    """,
                        (
                            project_id,
                            user_id,
                            repo_name,
                            project_slug,
                            repo_data["html_url"],
                            request.full_name,
                            repo_data["id"],
                            request.installation_id,
                            repo_data.get("language"),
                            repo_data.get("description"),
                            "pending",
                            repo_data.get("default_branch", "main"),
                        ),
                    )

                    result = cur.fetchone()

        except DatabaseError:
            raise HTTPException(status_code=500, detail="Failed to save project")

        return {
            "success": True,
            "project": {
                "id": result["id"],
                "name": result["name"],
                "slug": result["slug"],
                "status": result["status"],
                "created_at": result["created_at"].isoformat(),
                "repository_name": request.full_name,
                "language": repo_data.get("language"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Import error: {type(e).__name__}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to import repository")


@router.get("/projects")
async def get_user_projects(user_id: str = Depends(get_current_user_id)):
    try:
        with supabase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, slug, repository_url, repository_name,
                           language, description, status, created_at, updated_at,
                           deployment_status, deployment_error, deployment_started_at,
                           deployment_completed_at, aws_connection_id, application_url,
                           terraform_outputs, deployment_summary, cloud_provider
                    FROM projects
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                """,
                    (user_id,),
                )

                projects = cur.fetchall()

    except DatabaseError:
        raise HTTPException(status_code=500, detail="Failed to retrieve projects")

    return {
        "success": True,
        "count": len(projects),
        "projects": [
            {
                "id": p["id"],
                "name": p["name"],
                "slug": p["slug"],
                "repository_url": p["repository_url"],
                "repository_name": p["repository_name"],
                "installation_id": p.get("installation_id"),
                "language": p["language"],
                "description": p["description"],
                "status": p["status"],
                "created_at": p["created_at"].isoformat(),
                "deployment_status": p.get("deployment_status", "not_deployed"),
                "deployment_error": p.get("deployment_error"),
                "deployment_started_at": p.get("deployment_started_at"),
                "deployment_completed_at": p.get("deployment_completed_at"),
                "aws_connection_id": p.get("aws_connection_id"),
                "application_url": p.get("application_url"),
                "terraform_outputs": p.get("terraform_outputs"),
                "deployment_summary": p.get("deployment_summary"),
                "cloud_provider": p.get("cloud_provider", "gcp"),
                "framework_info": {
                    "framework": p["language"] or "other",
                    "display_name": p["language"] or "Other",
                },
                "deployment_info": {"url": None, "ip": None, "status": p["status"]},
            }
            for p in projects
        ],
    }


@router.get("/projects/repositories")
async def get_imported_repositories(user_id: str = Depends(get_current_user_id)):
    try:
        with supabase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, github_repo_id, repository_name,
                           language, created_at, installation_id
                    FROM projects
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                """,
                    (user_id,),
                )

                projects = cur.fetchall()

    except DatabaseError:
        raise HTTPException(status_code=500, detail="Failed to retrieve repositories")

    return {
        "success": True,
        "repositories": [
            {
                "id": p["id"],
                "github_id": str(p["github_repo_id"]),
                "name": p["name"],
                "full_name": p["repository_name"],
                "language": p["language"],
                "user_id": user_id,
                "created_at": p["created_at"].isoformat(),
            }
            for p in projects
        ],
    }


@router.get("/projects/{project_id}")
async def get_project_by_id(project_id: str, user_id: str = Depends(get_current_user_id)):
    """Get project by ID."""
    try:
        with supabase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, slug, repository_url, repository_name,
                           installation_id, language, description, status, created_at, updated_at,
                           deployment_status, deployment_error, deployment_started_at,
                           deployment_completed_at, aws_connection_id, aws_role_arn,
                           terraform_outputs, deployment_summary, application_url, cloud_provider
                    FROM projects
                    WHERE id = %s AND user_id = %s
                """,
                    (project_id, user_id),
                )

                project = cur.fetchone()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "success": True,
            "project": {
                "id": project["id"],
                "name": project["name"],
                "slug": project["slug"],
                "repository_url": project["repository_url"],
                "repository_name": project["repository_name"],
                "installation_id": project.get("installation_id"),
                "language": project["language"],
                "description": project["description"],
                "status": project["status"],
                "created_at": project["created_at"].isoformat(),
                "deployment_status": project.get("deployment_status", "not_deployed"),
                "deployment_error": project.get("deployment_error"),
                "deployment_started_at": project.get("deployment_started_at"),
                "deployment_completed_at": project.get("deployment_completed_at"),
                "aws_connection_id": project.get("aws_connection_id"),
                "aws_role_arn": project.get("aws_role_arn"),
                "terraform_outputs": project.get("terraform_outputs"),
                "deployment_summary": project.get("deployment_summary"),
                "application_url": project.get("application_url"),
                "cloud_provider": project.get("cloud_provider", "gcp"),
                "framework_info": {
                    "framework": project["language"] or "other",
                    "display_name": project["language"] or "Other",
                },
                "deployment_info": {"url": None, "ip": None, "status": project["status"]},
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project by ID: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve project")


@router.post("/projects/{project_id}/deploy")
async def deploy_project(
    project_id: str, request: Request, user_id: str = Depends(get_current_user_id)
):
    """Trigger manual CloudFormation deployment for a project."""
    try:
        logger.info(f"Deploying project {project_id} for user {user_id}")
        # Parse request body
        body = await request.json()
        generation_id = body.get("generation_id")

        if not generation_id:
            raise HTTPException(status_code=400, detail="Missing generation_id")

        # Get project details
        with supabase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.id, p.repository_name, p.installation_id,
                           g.id as gen_id, g.session_id
                    FROM projects p
                    JOIN generations g ON p.id = g.project_id
                    WHERE p.id = %s AND p.user_id = %s AND g.id = %s
                """,
                    (project_id, user_id, generation_id),
                )

                result = cur.fetchone()
        logger.info(f"Result: {result}")
        if not result:
            raise HTTPException(status_code=404, detail="Project or generation not found")

        # Update project status to deploying
        with supabase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE projects
                    SET deployment_status = 'deploying',
                        deployment_started_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                """,
                    (project_id,),
                )

        # Execute Terraform deployment synchronously for demo with real-time logs
        logger.info(f"Starting synchronous Terraform deployment for project {project_id}")

        from src.services.cloudformation_deployment import get_terraform_service

        tf_service = get_terraform_service()
        deployment_result = await tf_service.deploy_terraform(
            project_id=project_id,
            generation_id=generation_id,
            owner=result["repository_name"].split("/")[0],
            repo=result["repository_name"].split("/")[1],
            installation_id=result["installation_id"],
            session_id=result["session_id"],
            user_id=user_id,
        )

        # Update final status based on deployment result
        final_status = "deployed" if deployment_result.success else "deployment_failed"

        with supabase.get_connection() as conn:
            with conn.cursor() as cur:
                if deployment_result.success:
                    cur.execute(
                        """
                        UPDATE projects
                        SET deployment_status = %s,
                            deployment_completed_at = NOW(),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (final_status, project_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE projects
                        SET deployment_status = %s,
                            deployment_error = %s,
                            deployment_completed_at = NOW(),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (
                            final_status,
                            deployment_result.error[:500] if deployment_result.error else None,
                            project_id,
                        ),
                    )

        if deployment_result.success:
            logger.info(f"Terraform deployment completed successfully for project {project_id}")
            return {
                "success": True,
                "message": "Terraform deployment completed",
                "logs": deployment_result.logs,
                "outputs": deployment_result.outputs,
            }
        else:
            logger.error(
                f"Terraform deployment failed for project {project_id}: {deployment_result.error}"
            )
            return {
                "success": False,
                "message": "Terraform deployment failed",
                "error": deployment_result.error,
                "logs": deployment_result.logs,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start deployment")


@router.get("/projects/{project_id}/generations/{generation_id}/deploy/stream")
async def deploy_stream_sse(
    project_id: str,
    generation_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Server-Sent Events endpoint for streaming deployment logs in real-time."""

    async def event_generator():
        try:
            # Verify project ownership
            project = supabase.get_project_by_id(project_id)
            if not project or project["user_id"] != user_id:
                yield f"data: {json.dumps({'error': 'Project not found or access denied'})}\n\n"
                return

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting deployment...', 'step': 'initializing'})}\n\n"

            # Get deployment details
            with supabase.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT p.repository_name, p.installation_id, g.session_id
                        FROM projects p
                        JOIN generations g ON p.id = g.project_id
                        WHERE p.id = %s AND g.id = %s
                        """,
                        (project_id, generation_id),
                    )
                    result = cur.fetchone()

            if not result:
                yield f"data: {json.dumps({'error': 'Project or generation not found'})}\n\n"
                return

            # Update project status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Updating project status...', 'step': 'updating'})}\n\n"

            with supabase.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE projects
                        SET deployment_status = 'deploying',
                            deployment_started_at = NOW(),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (project_id,),
                    )

            # Run deployment with streaming
            from src.services.cloudformation_deployment import get_terraform_service

            tf_service = get_terraform_service()

            # Send progress updates
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting Terraform deployment...', 'step': 'terraform'})}\n\n"

            try:
                deployment_result = await tf_service.deploy_terraform(
                    project_id=project_id,
                    generation_id=generation_id,
                    owner=result["repository_name"].split("/")[0],
                    repo=result["repository_name"].split("/")[1],
                    installation_id=result["installation_id"],
                    session_id=result["session_id"],
                    user_id=user_id,
                    use_streaming=True,
                )

                # Stream logs as they come
                for log in deployment_result.logs:
                    yield f"data: {json.dumps({'type': 'log', 'message': log})}\n\n"
                    await asyncio.sleep(0.1)  # Small delay to ensure proper streaming

                # Send final status
                if deployment_result.success:
                    yield f"data: {json.dumps({'type': 'complete', 'success': True, 'message': 'Deployment completed successfully', 'outputs': deployment_result.outputs})}\n\n"

                    # Update database
                    with supabase.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                UPDATE projects
                                SET deployment_status = 'deployed',
                                    deployment_completed_at = NOW(),
                                    updated_at = NOW()
                                WHERE id = %s
                                """,
                                (project_id,),
                            )
                else:
                    yield f"data: {json.dumps({'type': 'complete', 'success': False, 'message': 'Deployment failed', 'error': deployment_result.error})}\n\n"

                    # Update database with error
                    with supabase.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                UPDATE projects
                                SET deployment_status = 'deployment_failed',
                                    deployment_error = %s,
                                    deployment_completed_at = NOW(),
                                    updated_at = NOW()
                                WHERE id = %s
                                """,
                                (
                                    deployment_result.error[:500]
                                    if deployment_result.error
                                    else None,
                                    project_id,
                                ),
                            )

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Deployment error: {str(e)}'})}\n\n"

        except Exception as e:
            logger.error(f"SSE deployment error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'Deployment error: {str(e)}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/projects/{project_slug}")
async def get_project_detail(project_slug: str, user_id: str = Depends(get_current_user_id)):
    try:
        with supabase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, slug, repository_url, repository_name,
                           installation_id, language, description, status, created_at, updated_at,
                           deployment_status, deployment_error, deployment_started_at,
                           deployment_completed_at, aws_connection_id, aws_role_arn
                    FROM projects
                    WHERE slug = %s AND user_id = %s
                """,
                    (project_slug, user_id),
                )

                project = cur.fetchone()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "success": True,
            "project": {
                "id": project["id"],
                "name": project["name"],
                "slug": project["slug"],
                "repository_name": project["repository_name"],
                "language": project["language"],
                "description": project["description"],
                "status": project["status"],
                "created_at": project["created_at"].isoformat(),
                "deployment_status": project.get("deployment_status", "not_deployed"),
                "deployment_error": project.get("deployment_error"),
                "deployment_started_at": project.get("deployment_started_at"),
                "deployment_completed_at": project.get("deployment_completed_at"),
                "aws_connection_id": project.get("aws_connection_id"),
                "aws_role_arn": project.get("aws_role_arn"),
            },
        }

    except HTTPException:
        raise
    except DatabaseError:
        raise HTTPException(status_code=500, detail="Failed to retrieve project")
    except Exception as e:
        logger.error(f"Project detail error: {type(e).__name__}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: str, request: UpdateProjectRequest, user_id: str = Depends(get_current_user_id)
):
    """Update project deployment status and AWS connection info."""
    try:
        # Verify project ownership
        project = supabase.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if project["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Update project
        update_data = {}
        if request.deployment_status is not None:
            update_data["deployment_status"] = request.deployment_status

        # If AWS role ARN is provided, find the corresponding AWS connection and link it
        if request.aws_role_arn is not None:
            # Find the AWS connection for this user with the matching role ARN
            aws_connection = supabase.get_aws_connection(user_id)
            if aws_connection and aws_connection.get("role_arn") == request.aws_role_arn:
                update_data["aws_connection_id"] = aws_connection["id"]
                update_data["aws_role_arn"] = request.aws_role_arn  # Also store for reference

        if update_data:
            with supabase.get_connection() as conn:
                with conn.cursor() as cur:
                    set_clauses = []
                    params = []

                    for key, value in update_data.items():
                        set_clauses.append(f"{key} = %s")
                        params.append(value)

                    params.append(project_id)

                    query = f"""
                        UPDATE projects
                        SET {", ".join(set_clauses)}, updated_at = NOW()
                        WHERE id = %s
                    """

                    cur.execute(query, params)

                    if cur.rowcount == 0:
                        raise HTTPException(status_code=404, detail="Project not found")

        # Return updated project
        updated_project = supabase.get_project_by_id(project_id)
        if not updated_project:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "success": True,
            "project": {
                "id": updated_project["id"],
                "name": updated_project["name"],
                "slug": updated_project["slug"],
                "repository_name": updated_project["repository_name"],
                "repository_url": updated_project.get("repository_url"),
                "language": updated_project["language"],
                "description": updated_project["description"],
                "status": updated_project["status"],
                "created_at": updated_project["created_at"].isoformat(),
                "deployment_status": updated_project.get("deployment_status", "not_deployed"),
                "deployment_error": updated_project.get("deployment_error"),
                "deployment_started_at": updated_project.get("deployment_started_at"),
                "deployment_completed_at": updated_project.get("deployment_completed_at"),
                "aws_connection_id": updated_project.get("aws_connection_id"),
                "aws_role_arn": updated_project.get("aws_role_arn"),
                "cloud_provider": updated_project.get("cloud_provider", "gcp"),
                "application_url": updated_project.get("application_url"),
                "terraform_outputs": updated_project.get("terraform_outputs"),
            },
        }

    except HTTPException:
        raise
    except DatabaseError:
        raise HTTPException(status_code=500, detail="Failed to update project")
    except Exception as e:
        logger.error(f"Project update error: {type(e).__name__}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")


@router.get("/projects/{project_id}/aws-status")
async def get_project_aws_status(project_id: str, user_id: str = Depends(get_current_user_id)):
    """Get AWS connection status for a specific project."""
    try:
        # Verify project ownership
        with supabase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT aws_connection_id FROM projects
                    WHERE id = %s AND user_id = %s
                """,
                    (project_id, user_id),
                )
                project = cur.fetchone()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get AWS connection details if exists
        aws_connection = None
        if project["aws_connection_id"]:
            aws_connection = supabase.get_aws_connection_by_id(project["aws_connection_id"])

        return {
            "success": True,
            "aws_connected": aws_connection is not None
            and aws_connection.get("status") == "verified",
            "aws_connection_id": project["aws_connection_id"],
            "aws_role_arn": aws_connection.get("role_arn") if aws_connection else None,
            "aws_status": aws_connection.get("status") if aws_connection else "not_connected",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get AWS status for project: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve AWS status")


@router.get("/projects/repositories")
async def get_imported_repositories(user_id: str = Depends(get_current_user_id)):
    try:
        with supabase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, github_repo_id, repository_name,
                           language, created_at, installation_id
                    FROM projects
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                """,
                    (user_id,),
                )

                projects = cur.fetchall()

    except DatabaseError:
        raise HTTPException(status_code=500, detail="Failed to retrieve repositories")

    return {
        "success": True,
        "repositories": [
            {
                "id": p["id"],
                "github_id": str(p["github_repo_id"]),
                "name": p["name"],
                "full_name": p["repository_name"],
                "language": p["language"],
                "user_id": user_id,
                "created_at": p["created_at"].isoformat(),
            }
            for p in projects
        ],
    }
