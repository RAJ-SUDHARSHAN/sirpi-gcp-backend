from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
import logging

from src.services.github_app import get_github_app, GitHubAppError
from src.services.supabase import supabase, DatabaseError
from src.core.config import settings
from src.utils.clerk_auth import get_current_user_id

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/github/callback")
async def github_app_callback(
    request: Request,
    installation_id: int,
    setup_action: str,
    state: str | None = None,
):
    try:
        if not state:
            raise HTTPException(status_code=400, detail="Missing state parameter")

        github = get_github_app()

        try:
            repos = await github.get_installation_repositories(installation_id)
        except GitHubAppError:
            raise HTTPException(status_code=502, detail="GitHub API error")

        if not repos:
            raise HTTPException(status_code=400, detail="No repositories found")

        installation_data = repos[0]["owner"]

        if not installation_data.get("login"):
            raise HTTPException(status_code=400, detail="Invalid repository owner data")

        try:
            supabase.save_github_installation(
                user_id=state,
                installation_id=installation_id,
                account_login=installation_data["login"],
                account_type=installation_data.get("type", "User"),
                account_avatar_url=installation_data.get("avatar_url"),
                repositories=[
                    {
                        "id": repo["id"],
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "private": repo["private"],
                    }
                    for repo in repos
                ],
            )
        except DatabaseError:
            raise HTTPException(status_code=500, detail="Failed to save installation")

        frontend_url = settings.cors_origins_list[0]

        try:
            clerk_user = supabase.get_user_by_clerk_id(state)
            if clerk_user and clerk_user.get("name"):
                username_slug = clerk_user["name"].split()[0].lower()
            else:
                username_slug = "projects"
        except DatabaseError:
            username_slug = "projects"

        redirect_url = f"{frontend_url}/{username_slug}/import?github_connected=true"
        return RedirectResponse(url=redirect_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GitHub callback error: {type(e).__name__}", exc_info=True)
        raise HTTPException(status_code=500, detail="Installation failed")


@router.get("/github/installation")
async def get_user_installation(user_id: str = Depends(get_current_user_id)):
    try:
        installation = supabase.get_user_installation(user_id)
    except DatabaseError:
        raise HTTPException(status_code=500, detail="Database error")

    if not installation:
        return {"connected": False, "installation_id": None}

    return {
        "connected": True,
        "installation_id": installation["installation_id"],
        "account_login": installation["account_login"],
        "repositories_count": len(installation.get("repositories", [])),
    }


@router.get("/github/repos/{installation_id}")
async def get_installation_repos(installation_id: int):
    github = get_github_app()

    try:
        repos = await github.get_installation_repositories(installation_id)
    except GitHubAppError:
        raise HTTPException(status_code=502, detail="GitHub API error")

    return {
        "installation_id": installation_id,
        "repositories": [
            {
                "id": repo["id"],
                "name": repo["name"],
                "full_name": repo["full_name"],
                "private": repo["private"],
                "html_url": repo["html_url"],
                "description": repo["description"],
                "language": repo["language"],
                "default_branch": repo["default_branch"],
                "updated_at": repo.get("updated_at"),
            }
            for repo in repos
        ],
    }
