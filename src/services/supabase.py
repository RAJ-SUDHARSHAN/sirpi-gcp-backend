"""
Supabase database service - Production ready.
Uses Transaction Pooler (Port 6543) optimized for AWS Lambda.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from src.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


class SupabaseService:
    """
    Supabase database service using Transaction Pooler.

    Production-ready configuration for AWS Lambda:
    - Uses Transaction Pooler (Port 6543)
    - Optimized connection settings for serverless
    - Automatic connection cleanup
    - Health check support
    """

    def __init__(self):
        """Initialize Supabase service with Transaction Pooler."""
        self._engine = None
        self._session_factory = None

    @property
    def engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            # Lambda-optimized configuration
            self._engine = create_engine(
                settings.database_url,
                poolclass=NullPool,
                pool_pre_ping=True,
                echo=False,  # Never echo in production
                connect_args={"connect_timeout": 10, "options": "-c statement_timeout=30000"},
            )

        return self._engine

    @property
    def session_factory(self):
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session with automatic commit/rollback.

        Usage:
            with supabase.get_session() as session:
                result = session.execute(text("SELECT * FROM users"))
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {type(e).__name__}", exc_info=True)
            raise DatabaseError("Database operation failed")
        finally:
            session.close()

    @contextmanager
    def get_connection(self):
        """
        Get a raw psycopg2 connection with automatic cleanup.

        Usage:
            with supabase.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM users")
                    results = cur.fetchall()
        """
        try:
            conn = psycopg2.connect(
                user=settings.supabase_user,
                password=settings.supabase_password,
                host=settings.supabase_host,
                port=settings.supabase_port,
                dbname=settings.supabase_dbname,
                cursor_factory=RealDictCursor,
                connect_timeout=10,
            )
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {type(e).__name__}")
            raise DatabaseError("Unable to connect to database")

        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database connection error: {type(e).__name__}", exc_info=True)
            raise DatabaseError("Database operation failed")
        finally:
            conn.close()

    def get_raw_connection(self):
        """
        Get a raw psycopg2 connection for LISTEN/NOTIFY (no context manager).
        Caller is responsible for closing the connection.

        Usage:
            conn = supabase.get_raw_connection()
            try:
                cur = conn.cursor()
                cur.execute("LISTEN channel_name;")
                # ... handle notifications
            finally:
                conn.close()
        """
        try:
            conn = psycopg2.connect(
                user=settings.supabase_user,
                password=settings.supabase_password,
                host=settings.supabase_host,
                port=settings.supabase_port,
                dbname=settings.supabase_dbname,
                connect_timeout=10,
            )
            return conn
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {type(e).__name__}")
            raise DatabaseError("Unable to connect to database")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check database connectivity and return status.

        Returns:
            Dict with status and latency
        """
        import time

        start = time.time()

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()

                    latency_ms = (time.time() - start) * 1000

                    return {"status": "healthy", "latency_ms": round(latency_ms, 2)}
        except Exception as e:
            logger.error(f"Database health check failed: {type(e).__name__}")
            return {"status": "unhealthy", "error": "Connection failed"}

    def save_generation(
        self,
        user_id: str,
        session_id: str,
        repository_url: str,
        template_type: str,
        status: str,
        project_id: Optional[str] = None,
        files: Optional[List[Dict[str, Any]]] = None,
        s3_keys: Optional[List[str]] = None,
        project_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save a new generation record to database."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO generations 
                        (user_id, session_id, repository_url, template_type, status, 
                         project_id, s3_keys, project_context, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id, created_at
                    """,
                        (
                            user_id,
                            session_id,
                            repository_url,
                            template_type,
                            status,
                            project_id,
                            Json(s3_keys or []),
                            Json(project_context or {}),
                        ),
                    )

                    result = cur.fetchone()
                    return result
        except Exception as e:
            logger.error(f"Failed to save generation: {type(e).__name__}")
            raise DatabaseError("Failed to save generation")

    def update_generation_status(
        self,
        session_id: str,
        status: str,
        files: Optional[List[Dict[str, Any]]] = None,
        s3_keys: Optional[List[str]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update generation status, s3_keys, and optionally add files or context."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    update_fields = ["status = %s", "updated_at = NOW()"]
                    params = [status]

                    if s3_keys is not None:
                        update_fields.append("s3_keys = %s")
                        params.append(Json(s3_keys))

                    if files is not None:
                        update_fields.append("files = %s")
                        params.append(Json(files))

                    if project_context is not None:
                        update_fields.append("project_context = %s")
                        params.append(Json(project_context))

                    if error is not None:
                        update_fields.append("error = %s")
                        params.append(error)

                    params.append(session_id)

                    query = f"""
                        UPDATE generations
                        SET {", ".join(update_fields)}
                        WHERE session_id = %s
                        RETURNING id
                    """

                    cur.execute(query, params)

                    result = cur.fetchone()
                    if result:
                        return True
                    return False
        except Exception as e:
            logger.error(f"Failed to update generation: {type(e).__name__}")
            raise DatabaseError("Failed to update generation")

    def get_generation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a generation by session_id."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, user_id, session_id, repository_url, template_type,
                               status, s3_keys, project_context, error,
                               pr_number, pr_url, pr_branch, pr_merged, pr_merged_at,
                               created_at, updated_at
                        FROM generations
                        WHERE session_id = %s
                    """,
                        (session_id,),
                    )

                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get generation: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve generation")

    def get_generation_by_repository(
        self, user_id: str, repository_url: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest generation for a repository."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, user_id, session_id, repository_url, template_type,
                               status, s3_keys, project_context, error,
                               pr_number, pr_url, pr_branch, pr_merged, pr_merged_at,
                               created_at, updated_at
                        FROM generations
                        WHERE user_id = %s AND repository_url = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """,
                        (user_id, repository_url),
                    )

                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get repository generation: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve repository generation")

    def get_user_generations(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all generations for a user (paginated)."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, session_id, repository_url, template_type,
                               status, pr_number, pr_url, pr_branch, pr_merged, pr_merged_at,
                               created_at, updated_at
                        FROM generations
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """,
                        (user_id, limit, offset),
                    )

                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get user generations: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve generations")

    def save_user_profile(
        self,
        clerk_user_id: str,
        email: str,
        name: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save or update user profile."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO users (clerk_user_id, email, name, avatar_url, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (clerk_user_id) 
                        DO UPDATE SET 
                            email = EXCLUDED.email,
                            name = EXCLUDED.name,
                            avatar_url = EXCLUDED.avatar_url,
                            updated_at = NOW()
                        RETURNING id, created_at
                    """,
                        (clerk_user_id, email, name, avatar_url),
                    )

                    result = cur.fetchone()
                    return result
        except Exception as e:
            logger.error(f"Failed to save user profile: {type(e).__name__}")
            raise DatabaseError("Failed to save user profile")

    def get_user_by_clerk_id(self, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by Clerk user ID."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM users
                        WHERE clerk_user_id = %s
                    """,
                        (clerk_user_id,),
                    )

                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get user: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve user")

    def save_github_installation(
        self,
        user_id: str,
        installation_id: int,
        account_login: str,
        account_type: str,
        account_avatar_url: Optional[str] = None,
        repositories: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Save or update GitHub App installation."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO github_installations 
                        (user_id, installation_id, account_login, account_type, account_avatar_url, repositories)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (installation_id)
                        DO UPDATE SET
                            user_id = EXCLUDED.user_id,
                            account_login = EXCLUDED.account_login,
                            account_type = EXCLUDED.account_type,
                            account_avatar_url = EXCLUDED.account_avatar_url,
                            repositories = EXCLUDED.repositories,
                            is_active = true,
                            updated_at = NOW()
                        RETURNING id, created_at
                    """,
                        (
                            user_id,
                            installation_id,
                            account_login,
                            account_type,
                            account_avatar_url,
                            Json(repositories or []),
                        ),
                    )

                    result = cur.fetchone()
                    return result
        except Exception as e:
            logger.error(f"Failed to save installation: {type(e).__name__}")
            raise DatabaseError("Failed to save GitHub installation")

    def get_user_installation(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's GitHub App installation."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM github_installations
                        WHERE user_id = %s AND is_active = true
                        ORDER BY created_at DESC
                        LIMIT 1
                    """,
                        (user_id,),
                    )

                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get installation: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve installation")

    def deactivate_installation(self, installation_id: int) -> bool:
        """Mark installation as inactive."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE github_installations
                        SET is_active = false, updated_at = NOW()
                        WHERE installation_id = %s
                        RETURNING id
                    """,
                        (installation_id,),
                    )

                    result = cur.fetchone()
                    if result:
                        return True
                    return False
        except Exception as e:
            logger.error(f"Failed to deactivate installation: {type(e).__name__}")
            raise DatabaseError("Failed to deactivate installation")

    def update_project_generation_status(
        self, project_id: str, status: str, increment_count: bool = True, cloud_provider: str = None
    ) -> bool:
        """
        Update project's generation status and count.
        When generation completes, also resets deployment_status to 'not_deployed' if it was previously destroyed.

        Args:
            project_id: Project UUID
            status: New status (pending/generating/completed/failed/pr_created)
            increment_count: Whether to increment generation_count
            cloud_provider: Optional cloud provider (gcp/aws) to update
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # If generation completed, reset deployment_status if it was destroyed
                    if status == "completed":
                        if increment_count:
                            if cloud_provider:
                                cur.execute(
                                    """
                                    UPDATE projects
                                    SET status = %s,
                                        cloud_provider = %s,
                                        generation_count = generation_count + 1,
                                        last_generated_at = NOW(),
                                        deployment_status = CASE 
                                            WHEN deployment_status = 'destroyed' THEN 'not_deployed'
                                            ELSE deployment_status
                                        END,
                                        updated_at = NOW()
                                    WHERE id = %s
                                    RETURNING id
                                """,
                                    (status, cloud_provider, project_id),
                                )
                            else:
                                cur.execute(
                                    """
                                    UPDATE projects
                                    SET status = %s,
                                        generation_count = generation_count + 1,
                                        last_generated_at = NOW(),
                                        deployment_status = CASE 
                                            WHEN deployment_status = 'destroyed' THEN 'not_deployed'
                                            ELSE deployment_status
                                        END,
                                        updated_at = NOW()
                                    WHERE id = %s
                                    RETURNING id
                                """,
                                    (status, project_id),
                                )
                        else:
                            cur.execute(
                                """
                                UPDATE projects
                                SET status = %s,
                                    deployment_status = CASE 
                                        WHEN deployment_status = 'destroyed' THEN 'not_deployed'
                                        ELSE deployment_status
                                    END,
                                    updated_at = NOW()
                                WHERE id = %s
                                RETURNING id
                            """,
                                (status, project_id),
                            )
                    else:
                        # For non-completed statuses, don't touch deployment_status
                        if increment_count:
                            if cloud_provider:
                                cur.execute(
                                    """
                                    UPDATE projects
                                    SET status = %s,
                                        cloud_provider = %s,
                                        generation_count = generation_count + 1,
                                        last_generated_at = NOW(),
                                        updated_at = NOW()
                                    WHERE id = %s
                                    RETURNING id
                                """,
                                    (status, cloud_provider, project_id),
                                )
                            else:
                                cur.execute(
                                    """
                                    UPDATE projects
                                    SET status = %s,
                                        generation_count = generation_count + 1,
                                        last_generated_at = NOW(),
                                        updated_at = NOW()
                                    WHERE id = %s
                                    RETURNING id
                                """,
                                    (status, project_id),
                                )
                        else:
                            cur.execute(
                                """
                                UPDATE projects
                                SET status = %s,
                                    updated_at = NOW()
                                WHERE id = %s
                                RETURNING id
                            """,
                                (status, project_id),
                            )

                    result = cur.fetchone()
                    return bool(result)
        except Exception as e:
            logger.error(f"Failed to update project status: {type(e).__name__}")
            raise DatabaseError("Failed to update project status")

    def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project by ID."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM projects
                        WHERE id = %s
                    """,
                        (project_id,),
                    )
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get project: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve project")

    def get_generation_by_id(self, generation_id: str) -> Optional[Dict[str, Any]]:
        """Get generation by ID."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM generations
                        WHERE id = %s
                    """,
                        (generation_id,),
                    )
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get generation: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve generation")

    def get_latest_generation_by_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get latest generation for a project."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM generations
                        WHERE project_id = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """,
                        (project_id,),
                    )
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get latest generation: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve latest generation")

    def update_generation_pr_info(
        self, generation_id: str, pr_number: int, pr_url: str, pr_branch: str
    ) -> bool:
        """Update generation with PR information."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE generations
                        SET pr_number = %s,
                            pr_url = %s,
                            pr_branch = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING id
                    """,
                        (pr_number, pr_url, pr_branch, generation_id),
                    )
                    result = cur.fetchone()
                    return bool(result)
        except Exception as e:
            logger.error(f"Failed to update generation PR info: {type(e).__name__}")
            raise DatabaseError("Failed to update generation PR info")

    def save_aws_connection(
        self, user_id: str, external_id: str, status: str = "pending"
    ) -> Dict[str, Any]:
        """
        Save AWS connection setup for user.

        Args:
            user_id: User ID
            external_id: Generated external ID for security
            status: Connection status (pending, verified, failed)

        Returns:
            AWS connection record
        """
        try:
            with self.get_session() as session:
                # Check if connection already exists
                existing = session.execute(
                    text("SELECT * FROM aws_connections WHERE user_id = :user_id"),
                    {"user_id": user_id},
                ).fetchone()

                if existing:
                    # Update existing connection
                    session.execute(
                        text("""
                        UPDATE aws_connections
                        SET external_id = :external_id, status = :status, updated_at = NOW()
                        WHERE user_id = :user_id
                        """),
                        {"user_id": user_id, "external_id": external_id, "status": status},
                    )
                else:
                    # Create new connection
                    session.execute(
                        text("""
                        INSERT INTO aws_connections (user_id, external_id, status)
                        VALUES (:user_id, :external_id, :status)
                        """),
                        {"user_id": user_id, "external_id": external_id, "status": status},
                    )

                session.commit()

                # Return the connection
                result = session.execute(
                    text("SELECT * FROM aws_connections WHERE user_id = :user_id"),
                    {"user_id": user_id},
                ).fetchone()

                return dict(result._mapping) if result else None

        except Exception as e:
            logger.error(f"Failed to save AWS connection: {e}")
            raise DatabaseError(f"Failed to save AWS connection: {str(e)}")

    def get_aws_connection(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get AWS connection for user.

        Args:
            user_id: User ID

        Returns:
            AWS connection record or None
        """
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("SELECT * FROM aws_connections WHERE user_id = :user_id"),
                    {"user_id": user_id},
                ).fetchone()

                return dict(result._mapping) if result else None

        except Exception as e:
            logger.error(f"Failed to get AWS connection: {e}")
            raise DatabaseError(f"Failed to get AWS connection: {str(e)}")

    def update_aws_connection(
        self, user_id: str, role_arn: str, status: str = "verified"
    ) -> Dict[str, Any]:
        """
        Update AWS connection with role ARN and status.
        Automatically extracts account ID from role ARN.

        Args:
            user_id: User ID
            role_arn: AWS IAM role ARN (format: arn:aws:iam::ACCOUNT_ID:role/RoleName)
            status: Connection status

        Returns:
            Updated AWS connection record
        """
        try:
            # Extract account ID from role ARN
            # Format: arn:aws:iam::353114555842:role/SirpiInfrastructureAutomationRole
            account_id = None
            if role_arn:
                parts = role_arn.split(":")
                if len(parts) >= 5:
                    account_id = parts[4]  # The account ID is the 5th part
                    logger.info(f"Extracted account ID: {account_id} from role ARN")

            with self.get_session() as session:
                # Update connection with role_arn, account_id, and status
                session.execute(
                    text("""
                    UPDATE aws_connections
                    SET role_arn = :role_arn, 
                        account_id = :account_id,
                        status = :status, 
                        verified_at = CASE WHEN :status = 'verified' THEN NOW() ELSE verified_at END,
                        updated_at = NOW()
                    WHERE user_id = :user_id
                    """),
                    {
                        "user_id": user_id,
                        "role_arn": role_arn,
                        "account_id": account_id,
                        "status": status,
                    },
                )

                session.commit()

                # Return updated connection
                result = session.execute(
                    text("SELECT * FROM aws_connections WHERE user_id = :user_id"),
                    {"user_id": user_id},
                ).fetchone()

                return dict(result._mapping) if result else None

        except Exception as e:
            logger.error(f"Failed to update AWS connection: {e}")
            raise DatabaseError(f"Failed to update AWS connection: {str(e)}")

    def get_aws_connection_by_id(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get AWS connection by ID."""
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("SELECT * FROM aws_connections WHERE id = :connection_id"),
                    {"connection_id": connection_id},
                )
                connection = result.fetchone()
                return dict(connection._mapping) if connection else None

        except Exception as e:
            logger.error(f"Failed to get AWS connection by ID: {e}")
            raise DatabaseError(f"Failed to get AWS connection: {str(e)}")

    def update_project_deployment_status(
        self, project_id: str, status: str, error: Optional[str] = None
    ) -> bool:
        """Update project deployment status."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if error:
                        cur.execute(
                            """
                            UPDATE projects
                            SET deployment_status = %s,
                                deployment_error = %s,
                                updated_at = NOW()
                            WHERE id = %s
                            RETURNING id
                            """,
                            (status, error, project_id),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE projects
                            SET deployment_status = %s,
                                deployment_completed_at = CASE WHEN %s = 'deployed' THEN NOW() ELSE deployment_completed_at END,
                                updated_at = NOW()
                            WHERE id = %s
                            RETURNING id
                            """,
                            (status, status, project_id),
                        )
                    result = cur.fetchone()
                    return bool(result)
        except Exception as e:
            logger.error(f"Failed to update project deployment status: {type(e).__name__}")
            raise DatabaseError("Failed to update deployment status")

    def save_deployment_logs(
        self,
        project_id: str,
        operation_type: str,
        logs: List[str],
        status: str,
        duration_seconds: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save deployment operation logs."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO deployment_logs 
                        (project_id, operation_type, logs, status, duration_seconds, error_message, completed_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        RETURNING id, created_at
                        """,
                        (
                            project_id,
                            operation_type,
                            Json(logs),
                            status,
                            duration_seconds,
                            error_message,
                        ),
                    )
                    result = cur.fetchone()
                    return result
        except Exception as e:
            logger.error(f"Failed to save deployment logs: {type(e).__name__}")
            raise DatabaseError("Failed to save deployment logs")

    def get_deployment_logs(
        self, project_id: str, operation_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get deployment logs for a project. Returns only the LATEST log per operation type."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if operation_type:
                        # Get latest log for specific operation type
                        cur.execute(
                            """
                            SELECT id, operation_type, logs, status, duration_seconds, 
                                   error_message, created_at, completed_at
                            FROM deployment_logs
                            WHERE project_id = %s AND operation_type = %s
                            ORDER BY created_at DESC
                            LIMIT 1
                            """,
                            (project_id, operation_type),
                        )
                    else:
                        # Get latest log for EACH operation type using DISTINCT ON
                        cur.execute(
                            """
                            SELECT DISTINCT ON (operation_type) 
                                   id, operation_type, logs, status, duration_seconds,
                                   error_message, created_at, completed_at
                            FROM deployment_logs
                            WHERE project_id = %s
                            ORDER BY operation_type, created_at DESC
                            """,
                            (project_id,),
                        )
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get deployment logs: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve deployment logs")

    def save_terraform_outputs(self, project_id: str, outputs: Dict[str, Any]) -> bool:
        """Save terraform outputs for a project."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE projects
                        SET terraform_outputs = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING id
                        """,
                        (Json(outputs), project_id),
                    )
                    result = cur.fetchone()
                    return bool(result)
        except Exception as e:
            logger.error(f"Failed to save terraform outputs: {type(e).__name__}")
            raise DatabaseError("Failed to save terraform outputs")

    def get_terraform_outputs(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get terraform outputs for a project."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT terraform_outputs
                        FROM projects
                        WHERE id = %s
                        """,
                        (project_id,),
                    )
                    result = cur.fetchone()
                    return result["terraform_outputs"] if result else None
        except Exception as e:
            logger.error(f"Failed to get terraform outputs: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve terraform outputs")

    def update_application_url(self, project_id: str, url: str) -> bool:
        """Update application URL for easy access."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE projects
                        SET application_url = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING id
                        """,
                        (url, project_id),
                    )
                    result = cur.fetchone()
                    if result:
                        logger.info(f"Updated application_url for project {project_id}: {url}")
                        return True
                    return False
        except Exception as e:
            logger.error(f"Failed to update application URL: {type(e).__name__}")
            raise DatabaseError("Failed to update application URL")

    # ============================================================================
    # ENVIRONMENT VARIABLES MANAGEMENT
    # ============================================================================

    def save_project_env_var(
        self,
        project_id: str,
        key: str,
        value_encrypted: str,
        is_secret: bool = True,
        description: str = None,
    ) -> Dict[str, Any]:
        """Save or update a project environment variable."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO project_env_vars (project_id, key, value_encrypted, is_secret, description)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (project_id, key)
                        DO UPDATE SET
                            value_encrypted = EXCLUDED.value_encrypted,
                            is_secret = EXCLUDED.is_secret,
                            description = EXCLUDED.description,
                            updated_at = NOW()
                        RETURNING id, created_at
                        """,
                        (project_id, key, value_encrypted, is_secret, description),
                    )
                    result = cur.fetchone()
                    return result
        except Exception as e:
            logger.error(f"Failed to save env var: {type(e).__name__}")
            raise DatabaseError("Failed to save environment variable")

    def get_project_env_vars(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all environment variables for a project."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, key, value_encrypted, is_secret, description, created_at, updated_at
                        FROM project_env_vars
                        WHERE project_id = %s
                        ORDER BY key
                        """,
                        (project_id,),
                    )
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get env vars: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve environment variables")

    def delete_project_env_var(self, project_id: str, key: str) -> bool:
        """Delete a project environment variable."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM project_env_vars
                        WHERE project_id = %s AND key = %s
                        RETURNING id
                        """,
                        (project_id, key),
                    )
                    result = cur.fetchone()
                    return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete env var: {type(e).__name__}")
            raise DatabaseError("Failed to delete environment variable")

    # ============================================================================
    # GCP CREDENTIALS MANAGEMENT
    # ============================================================================

    def save_gcp_credentials(
        self, user_id: str, gcp_project_id: str, service_account_json_encrypted: str
    ) -> Dict[str, Any]:
        """Save or update GCP credentials for a user."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO gcp_credentials (user_id, project_id, service_account_json_encrypted, status)
                        VALUES (%s, %s, %s, 'active')
                        ON CONFLICT (user_id)
                        DO UPDATE SET
                            project_id = EXCLUDED.project_id,
                            service_account_json_encrypted = EXCLUDED.service_account_json_encrypted,
                            status = 'active',
                            verified_at = NOW(),
                            updated_at = NOW()
                        RETURNING id, created_at
                        """,
                        (user_id, gcp_project_id, service_account_json_encrypted),
                    )
                    result = cur.fetchone()
                    return result
        except Exception as e:
            logger.error(f"Failed to save GCP credentials: {type(e).__name__}")
            raise DatabaseError("Failed to save GCP credentials")

    def get_gcp_credentials(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get GCP OAuth credentials for a user (access_token, refresh_token, token_expiry)."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, user_id, project_id, access_token, refresh_token, 
                               token_expiry, status, verified_at, updated_at
                        FROM gcp_credentials
                        WHERE user_id = %s AND status = 'active'
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (user_id,),
                    )
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get GCP credentials: {type(e).__name__}")
            raise DatabaseError("Failed to retrieve GCP credentials")

    def save_workflow_stage_logs(
        self,
        session_id: str,
        stage: str,
        logs: List[str],
        status: str,
        duration_seconds: Optional[int] = None,
    ) -> None:
        """Save workflow stage logs to database."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO workflow_logs (session_id, stage, logs, status, duration_seconds, completed_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (session_id, stage)
                        DO UPDATE SET
                            logs = EXCLUDED.logs,
                            status = EXCLUDED.status,
                            duration_seconds = EXCLUDED.duration_seconds,
                            completed_at = NOW()
                        """,
                        (session_id, stage, Json(logs), status, duration_seconds),
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save workflow logs: {e}")

    def get_workflow_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve workflow logs for a session."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT stage, logs, status, duration_seconds, created_at, completed_at
                        FROM workflow_logs
                        WHERE session_id = %s
                        ORDER BY created_at ASC
                        """,
                        (session_id,),
                    )
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get workflow logs: {e}")
            return []

    def save_deployment_logs(
        self,
        project_id: str,
        operation_type: str,
        logs: List[str],
        status: str,
        error_message: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save deployment logs (build, plan, apply, destroy) to database."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO deployment_logs 
                        (project_id, operation_type, logs, status, error_message, duration_seconds, metadata, completed_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (project_id, operation_type)
                        DO UPDATE SET
                            logs = EXCLUDED.logs,
                            status = EXCLUDED.status,
                            error_message = EXCLUDED.error_message,
                            duration_seconds = EXCLUDED.duration_seconds,
                            metadata = EXCLUDED.metadata,
                            completed_at = NOW(),
                            updated_at = NOW()
                        """,
                        (
                            project_id,
                            operation_type,
                            Json(logs),
                            status,
                            error_message,
                            duration_seconds,
                            Json(metadata or {}),
                        ),
                    )
                conn.commit()
                logger.info(f"Saved {operation_type} logs for project {project_id}")
        except Exception as e:
            logger.error(f"Failed to save deployment logs: {e}")

    def get_deployment_logs(
        self, project_id: str, operation_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve deployment logs for a project."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if operation_type:
                        # Get specific operation logs
                        cur.execute(
                            """
                            SELECT operation_type, logs, status, error_message, duration_seconds, metadata, completed_at
                            FROM deployment_logs
                            WHERE project_id = %s AND operation_type = %s
                            ORDER BY completed_at DESC
                            LIMIT 1
                            """,
                            (project_id, operation_type),
                        )
                        return cur.fetchone()
                    else:
                        # Get all operations
                        cur.execute(
                            """
                            SELECT operation_type, logs, status, error_message, duration_seconds, metadata, completed_at
                            FROM deployment_logs
                            WHERE project_id = %s
                            ORDER BY completed_at DESC
                            """,
                            (project_id,),
                        )
                        return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get deployment logs: {e}")
            return None


# Global singleton instance
supabase = SupabaseService()


def get_supabase_service() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase
