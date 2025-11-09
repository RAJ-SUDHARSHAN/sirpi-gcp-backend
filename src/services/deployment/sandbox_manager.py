"""
Unified E2B Sandbox Manager for Cloud-Agnostic Deployments.
Handles Docker builds, Terraform operations, and repository cloning.
"""

import logging
import asyncio
from typing import Optional, List, Callable, Dict, Any
from e2b_code_interpreter import Sandbox
from src.core.config import settings

logger = logging.getLogger(__name__)


class SandboxManager:
    """
    Unified E2B sandbox manager for both AWS and GCP deployments.
    Uses custom template with pre-installed tools (Docker, Terraform, gcloud, AWS CLI).
    """

    def __init__(self, template_id: Optional[str] = None):
        """
        Initialize sandbox manager.

        Args:
            template_id: Optional custom E2B template ID.
                        If None, uses default template (slower, installs packages at runtime)
        """
        self.template_id = template_id or getattr(settings, "e2b_template_id", None)
        self.sandbox: Optional[Sandbox] = None
        self._log_callback: Optional[Callable[[str], None]] = None

    async def create_sandbox(self) -> Sandbox:
        """Create and return E2B sandbox instance."""
        try:
            if self.template_id:
                self._log(f"Creating E2B sandbox with custom template: {self.template_id}")
                self.sandbox = await asyncio.to_thread(
                    Sandbox.create, template=self.template_id, api_key=settings.e2b_api_key
                )
            else:
                self._log("Creating E2B sandbox with default template")
                self.sandbox = await asyncio.to_thread(Sandbox.create, api_key=settings.e2b_api_key)

            self._log(f"Sandbox created: {self.sandbox.sandbox_id}")
            return self.sandbox

        except Exception as e:
            logger.error(f"Failed to create E2B sandbox: {e}")
            raise RuntimeError(f"Sandbox creation failed: {str(e)}")

    def set_log_callback(self, callback: Callable[[str], None]):
        """Set callback function for streaming logs."""
        self._log_callback = callback

    def _log(self, message: str):
        """Internal logging that also calls callback if set."""
        from datetime import datetime

        logger.info(message)
        if self._log_callback:
            # Add timestamp to deployment logs for consistency with workflow logs
            timestamp_str = datetime.now().strftime("%I:%M:%S %p")
            formatted_message = f"{timestamp_str}  {message}"
            self._log_callback(formatted_message)

    async def _ensure_docker_daemon(self):
        """Ensure Docker daemon is running and accessible."""
        try:
            # Check if Docker is already accessible
            result = await self.run_command(
                "docker info",
                stream_output=False,
                timeout=10,  # Quick check
            )

            if result["exit_code"] == 0:
                self._log("✅ Docker daemon already running and accessible")
                return
        except Exception:
            pass

        # Docker socket exists but permission denied - fix permissions
        self._log("Fixing Docker socket permissions...")

        try:
            # Check if dockerd is already running
            ps_result = await self.run_command(
                "ps aux | grep dockerd | grep -v grep", stream_output=False, timeout=10
            )

            if ps_result["exit_code"] == 0 and ps_result["stdout"]:
                self._log("Docker daemon already running, fixing socket permissions")

                # Fix socket permissions
                await asyncio.to_thread(
                    self.sandbox.commands.run, "sudo chmod 666 /var/run/docker.sock"
                )

                # Verify it works now
                await asyncio.sleep(1)
                result = await self.run_command("docker info", stream_output=False)
                if result["exit_code"] == 0:
                    self._log("✅ Docker socket permissions fixed")
                    return

            # If we get here, need to start dockerd
            self._log("Starting Docker daemon...")

            # Clean up stale PID file if exists
            await asyncio.to_thread(self.sandbox.commands.run, "sudo rm -f /var/run/docker.pid")

            # Start dockerd in background
            start_cmd = (
                "sudo nohup dockerd --host=unix:///var/run/docker.sock > /tmp/dockerd.log 2>&1 &"
            )
            await asyncio.to_thread(self.sandbox.commands.run, start_cmd)

            # Wait for Docker to be ready
            self._log("Waiting for Docker daemon...")
            for i in range(20):
                await asyncio.sleep(0.5)

                # Fix permissions on each attempt
                try:
                    await asyncio.to_thread(
                        self.sandbox.commands.run, "sudo chmod 666 /var/run/docker.sock"
                    )
                except Exception:
                    pass

                try:
                    result = await self.run_command("docker info", stream_output=False, timeout=5)
                    if result["exit_code"] == 0:
                        self._log("✅ Docker daemon started successfully")
                        return
                except Exception:
                    continue

            raise RuntimeError("Docker daemon failed to become ready")

        except Exception as e:
            # Log error details
            try:
                log_result = await self.run_command(
                    "cat /tmp/dockerd.log 2>/dev/null || echo 'No logs'",
                    stream_output=False,
                    timeout=5,
                )
                self._log(f"Docker logs: {log_result['stdout'][:500]}")
            except Exception:
                pass

            logger.error(f"Failed to setup Docker: {e}")
            raise

    async def clone_repository(
        self, repo_url: str, branch: str = "main", target_dir: str = "/home/user/repo"
    ) -> str:
        """
        Clone GitHub repository into sandbox.

        Args:
            repo_url: GitHub repository URL
            branch: Branch to clone
            target_dir: Where to clone the repo

        Returns:
            Path to cloned repository
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")

        self._log(f"Cloning {repo_url} (branch: {branch})...")

        try:
            # Clone repository (can take time for large repos)
            result = await asyncio.to_thread(
                self.sandbox.commands.run,
                f"git clone --branch {branch} --depth 1 {repo_url} {target_dir}",
                timeout=300,  # 5 minutes for large repos
            )

            if result.exit_code != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")

            self._log(f"Repository cloned to {target_dir}")
            return target_dir

        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise

    async def build_docker_image(
        self,
        dockerfile_path: str,
        image_name: str,
        context_dir: str = ".",
        build_args: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build Docker image in sandbox.

        Args:
            dockerfile_path: Path to Dockerfile
            image_name: Name and tag for the image (e.g., "myapp:latest")
            context_dir: Build context directory
            build_args: Optional build arguments to pass to docker build (e.g., {"API_KEY": "value"})

        Returns:
            Image name with tag
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not created")

        # Start Docker daemon if not already running
        await self._ensure_docker_daemon()

        self._log(f"Building Docker image: {image_name}...")

        try:
            # Build Docker command with optional build args
            build_cmd = f"cd {context_dir} && docker build -f {dockerfile_path}"

            # Add build args if provided
            if build_args:
                self._log(f"Using {len(build_args)} build arguments")
                for key, value in build_args.items():
                    # Escape quotes in values
                    escaped_value = value.replace('"', '\\"')
                    build_cmd += f' --build-arg {key}="{escaped_value}"'

            build_cmd += f" -t {image_name} ."

            result = await self.run_command(
                build_cmd,
                stream_output=True,
                timeout=600,  # 10 minutes for Docker builds
            )

            if result["exit_code"] != 0:
                self._log(f"Docker build failed: {result['stderr']}")
                raise RuntimeError(f"Docker build failed: {result['stderr']}")

            self._log(f"Docker image built successfully: {image_name}")
            return image_name

        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            raise

    async def run_command(
        self,
        command: str,
        working_dir: Optional[str] = None,
        stream_output: bool = True,
        timeout: int = 60,  # Default 60s, can be increased for long operations
    ) -> Dict[str, Any]:
        """
        Run arbitrary command in sandbox.

        Args:
            command: Command to execute
            working_dir: Optional working directory
            stream_output: Whether to stream output via callback
            timeout: Command timeout in seconds (0 = no timeout)

        Returns:
            Dict with exit_code, stdout, stderr
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not created")

        full_command = f"cd {working_dir} && {command}" if working_dir else command

        # Log the actual command being executed (for transparency)
        if stream_output and self._log_callback:
            # Show command in a distinct format
            self._log(f"$ {command}")

        try:
            # Pass timeout to E2B (0 = no timeout)
            result = await asyncio.to_thread(
                self.sandbox.commands.run, full_command, timeout=timeout
            )

            if stream_output:
                if result.stdout:
                    for line in result.stdout.split("\n"):
                        if line.strip():
                            self._log(line)
                            # Small delay to ensure async log callbacks complete
                            await asyncio.sleep(0.01)

                if result.stderr and result.exit_code != 0:
                    for line in result.stderr.split("\n"):
                        if line.strip():
                            self._log(f"ERROR: {line}")

            return {"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr}

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

    async def write_file(self, path: str, content: str):
        """Write content to a file in the sandbox."""
        if not self.sandbox:
            raise RuntimeError("Sandbox not created")

        try:
            # Use python to write file (more reliable than echo for multiline)
            await asyncio.to_thread(self.sandbox.files.write, path, content)
            self._log(f"Wrote file: {path}")

        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            raise

    async def read_file(self, path: str) -> str:
        """Read file content from sandbox."""
        if not self.sandbox:
            raise RuntimeError("Sandbox not created")

        try:
            content = await asyncio.to_thread(self.sandbox.files.read, path)
            return content

        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise

    async def run_terraform(
        self,
        operation: str,  # "init", "plan", "apply", "destroy"
        working_dir: str,
        var_file: Optional[str] = None,
        auto_approve: bool = False,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run Terraform operations with proper GCP authentication and timeouts.

        Args:
            operation: Terraform command (init/plan/apply/destroy)
            working_dir: Directory containing Terraform files
            var_file: Optional path to tfvars file
            auto_approve: Whether to auto-approve (for apply/destroy)
            env_vars: Optional environment variables (e.g., GOOGLE_APPLICATION_CREDENTIALS)

        Returns:
            Dict with exit_code, stdout, stderr
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not created")

        self._log(f"Running terraform {operation}...")

        # Set appropriate timeout for each operation
        timeouts = {
            "init": 180,  # 3 minutes (downloads providers)
            "plan": 180,  # 3 minutes (queries cloud state)
            "apply": 600,  # 10 minutes (creates resources, waits for health checks)
            "destroy": 600,  # 10 minutes (deletes resources)
        }

        timeout = timeouts.get(operation, 300)  # Default 5 mins

        try:
            # Build terraform command
            if operation == "init":
                cmd = "terraform init"
            elif operation == "plan":
                cmd = "terraform plan"
                if var_file:
                    cmd += f" -var-file={var_file}"
            elif operation in ["apply", "destroy"]:
                cmd = f"terraform {operation}"
                if var_file:
                    cmd += f" -var-file={var_file}"
                if auto_approve:
                    cmd += " -auto-approve"
            else:
                raise ValueError(f"Invalid terraform operation: {operation}")

            # Prepend environment variables if provided
            if env_vars:
                env_prefix = " ".join([f"{k}={v}" for k, v in env_vars.items()])
                cmd = f"{env_prefix} {cmd}"

            # Run terraform with appropriate timeout
            result = await self.run_command(
                cmd, working_dir=working_dir, stream_output=True, timeout=timeout
            )

            if result["exit_code"] != 0:
                raise RuntimeError(f"Terraform {operation} failed: {result['stderr']}")

            self._log(f"Terraform {operation} completed successfully")
            return result

        except Exception as e:
            error_msg = str(e)

            # If apply/destroy failed due to state lock, try to force-unlock
            if operation in ["apply", "destroy"] and "state lock" in error_msg.lower():
                try:
                    self._log("⚠️ Detected stale state lock, attempting to release...")

                    # Extract lock ID from error message if possible
                    import re

                    lock_id_match = re.search(r"ID:\s+(\d+)", error_msg)

                    if lock_id_match:
                        lock_id = lock_id_match.group(1)
                        self._log(f"Found lock ID: {lock_id}")

                        # Build unlock command with env vars
                        env_prefix = " ".join([f"{k}={v}" for k, v in (env_vars or {}).items()])
                        unlock_cmd = (
                            f"{env_prefix} terraform force-unlock -force {lock_id}"
                            if env_prefix
                            else f"terraform force-unlock -force {lock_id}"
                        )

                        await self.run_command(
                            unlock_cmd, working_dir=working_dir, stream_output=False, timeout=30
                        )
                        self._log("✅ State lock released successfully")
                        self._log("🔄 Please retry the operation")
                    else:
                        self._log("⚠️ Could not extract lock ID from error")

                except Exception as unlock_error:
                    # Lock cleanup failed, provide manual instruction
                    self._log(f"⚠️ Auto-unlock failed: {str(unlock_error)[:100]}")
                    if lock_id_match:
                        self._log(
                            f"Manual unlock: terraform force-unlock -force {lock_id_match.group(1)}"
                        )

            logger.error(f"Terraform {operation} failed: {e}")
            raise

    async def cleanup(self):
        """Clean up and close the sandbox."""
        if self.sandbox:
            try:
                await asyncio.to_thread(self.sandbox.kill)
                logger.info(f"Sandbox {self.sandbox.sandbox_id} terminated")
            except Exception as e:
                logger.error(f"Failed to kill sandbox: {e}")

    async def __aenter__(self):
        """Context manager entry."""
        await self.create_sandbox()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()


# Singleton instance getter
_sandbox_manager: Optional[SandboxManager] = None


def get_sandbox_manager(template_id: Optional[str] = None) -> SandboxManager:
    """Get or create sandbox manager instance."""
    global _sandbox_manager

    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager(template_id)

    return _sandbox_manager
