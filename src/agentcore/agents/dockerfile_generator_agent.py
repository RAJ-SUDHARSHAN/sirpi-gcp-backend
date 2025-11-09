"""
Dockerfile Generator Agent.
Generates production-hardened, multi-stage Dockerfiles.
"""

import logging
import re
import json
from pydantic import BaseModel, Field

from src.agentcore.agents.base_agent import BaseAgent
from src.agentcore.prompts import load_prompt_file, load_all_examples, format_prompt
from src.agentcore.config.framework_metadata import (
    get_build_output_path,
    detect_frontend_framework_from_dependencies,
)
from .code_analyzer_agent import AnalysisResult


class DockerfileOutput(BaseModel):
    """Structured Dockerfile output."""

    content: str = Field(..., description="Raw Dockerfile content")


class DockerfileGeneratorAgent(BaseAgent):
    """Generates production-grade, security-hardened Dockerfiles."""

    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.2)

    def get_system_instruction(self) -> str:
        """Load system instruction from prompts directory."""
        try:
            system_instruction = load_prompt_file("dockerfile_generator", "system_instruction.txt")

            # Load and append examples
            examples = load_all_examples("dockerfile_generator")
            if examples:
                examples_text = "\n\n**EXAMPLES:**\n\n"
                for name, content in sorted(examples.items()):
                    example_title = name.replace("_", " ").replace(".dockerfile", "").title()
                    examples_text += f"**{example_title}:**\n{content}\n\n"
                system_instruction += examples_text

            return system_instruction
        except Exception as e:
            self.logger.error(f"Failed to load system instruction: {e}")
            raise

    async def generate(self, analysis: AnalysisResult) -> str:
        """Generate production-hardened Dockerfile."""
        self._log_execution(
            "START", f"Generating Dockerfile for {analysis.framework or analysis.language}"
        )

        # Build prompt variables
        deps_str = "\n".join(f"- {k}: {v}" for k, v in list(analysis.dependencies.items())[:10])
        env_vars = (
            ", ".join(analysis.environment_variables) if analysis.environment_variables else "None"
        )

        # Determine if this is a monorepo deployment
        is_monorepo = analysis.is_monorepo
        monorepo_note = ""
        deployment_type = "SINGLE SERVICE: Generate 2-stage Dockerfile (build + runtime)"
        monorepo_instruction = "Standard single-service deployment"

        if is_monorepo:
            monorepo_note = f"""
**MONOREPO DEPLOYMENT:**
This is a monorepo with backend and frontend.
- You MUST generate a multi-stage Dockerfile that builds BOTH services
- Stage 1: Build frontend (assumes frontend/ directory with package.json)
- Stage 2: Build backend dependencies (from {analysis.monorepo_type or "backend"} directory)
- Stage 3: Combine both - backend serves frontend as static files
- Copy frontend build output to ./static or ./public in final stage
- Backend will serve frontend via static file middleware
"""
            deployment_type = (
                "MONOREPO: Generate 3-stage Dockerfile (frontend build + backend build + runtime)"
            )
            monorepo_instruction = "For monorepo: Copy frontend build to ./static in final stage"

        # Load and format prompt template
        try:
            prompt_template = load_prompt_file("dockerfile_generator", "prompt_template.txt")
            prompt = format_prompt(
                prompt_template,
                language=analysis.language,
                framework=analysis.framework or f"Generic {analysis.language}",
                runtime_version=analysis.runtime_version or "latest stable",
                package_manager=analysis.package_manager,
                exposed_port=analysis.exposed_port or 8080,
                start_command=analysis.start_command,
                build_command=analysis.build_command or "None required",
                health_check_path=analysis.health_check_path,
                is_monorepo=is_monorepo,
                monorepo_note=monorepo_note,
                dependencies=deps_str,
                environment_variables=env_vars,
                deployment_type=deployment_type,
                monorepo_instruction=monorepo_instruction,
            )
        except Exception as e:
            self.logger.error(f"Failed to load prompt template: {e}")
            raise

        try:
            # Generate raw text (not structured output)
            dockerfile = await self._generate_text(prompt=prompt)

            # Aggressive markdown cleanup
            dockerfile = self._cleanup_markdown(dockerfile)

            # Strip instructional comments and setup guidance
            dockerfile = self._strip_instructional_comments(dockerfile)

            # Fix common AI mistakes
            dockerfile = self._fix_common_cmd_mistakes(dockerfile, analysis)

            # Post-process: Replace dynamic placeholders
            if is_monorepo:
                # Detect frontend framework from dependencies or use provided
                frontend_fw = analysis.frontend_framework
                if not frontend_fw and analysis.dependencies:
                    frontend_fw = detect_frontend_framework_from_dependencies(analysis.dependencies)

                # Get build output path from framework metadata
                build_output = get_build_output_path(
                    framework=analysis.framework or "react", frontend_framework=frontend_fw
                )

                # Replace placeholder with actual path
                dockerfile = dockerfile.replace("{FRONTEND_BUILD_OUTPUT}", build_output)
                self.logger.info(f"Replaced frontend build output with: {build_output}")

            # Validate critical directives
            if len(dockerfile) < 300:
                raise ValueError(f"Dockerfile too short ({len(dockerfile)} chars)")

            dockerfile_upper = dockerfile.upper()
            if "FROM" not in dockerfile_upper:
                raise ValueError("Dockerfile missing FROM instruction")

            # CRITICAL: Validate Next.js standalone mode has .next/static copy
            if analysis.framework and "next" in analysis.framework.lower():
                if "STANDALONE" in dockerfile_upper or "OUTPUT:" in dockerfile_upper:
                    if ".NEXT/STATIC" not in dockerfile_upper:
                        self.logger.error(
                            "CRITICAL: Next.js standalone Dockerfile missing .next/static copy!"
                        )
                        self.logger.error(
                            "This will cause CSS/JS to not load (white page with unstyled text)"
                        )
                        # Add the missing copy after standalone
                        lines = dockerfile.split("\n")
                        fixed_lines = []
                        for i, line in enumerate(lines):
                            fixed_lines.append(line)
                            # If this line copies standalone, add static copy after it
                            if "COPY" in line.upper() and "STANDALONE" in line.upper():
                                # Check if next line already has static
                                if (
                                    i + 1 < len(lines)
                                    and ".next/static" not in lines[i + 1].lower()
                                ):
                                    fixed_lines.append("")
                                    fixed_lines.append(
                                        "# CRITICAL: Copy static assets for CSS/JS to work"
                                    )
                                    fixed_lines.append(
                                        "COPY --from=builder /app/.next/static ./.next/static"
                                    )
                                    fixed_lines.append("COPY --from=builder /app/public ./public")
                                    self.logger.warning(
                                        "Auto-fixed: Added missing .next/static copy"
                                    )
                        dockerfile = "\n".join(fixed_lines)
                        dockerfile_upper = dockerfile.upper()

            if "CMD" not in dockerfile_upper and "ENTRYPOINT" not in dockerfile_upper:
                self.logger.warning("Dockerfile missing CMD/ENTRYPOINT, adding default CMD")
                # Add a default CMD based on language
                if analysis.language.lower() == "python":
                    # For Python: Use python -m to run the main module
                    # The app should read PORT from environment (Cloud Run sets this)
                    dockerfile += '\nCMD ["python", "-m", "src.main"]'
                elif analysis.language.lower() in ["javascript", "typescript"]:
                    cmd = analysis.start_command or "node server.js"
                    dockerfile += f"\nCMD {json.dumps(cmd.split())}"
                else:
                    dockerfile += f'\nCMD ["/bin/sh", "-c", "{analysis.start_command or "echo No start command defined"}"]'

            # Log success
            has_user = "USER" in dockerfile_upper
            has_healthcheck = "HEALTHCHECK" in dockerfile_upper
            has_cmd = "CMD" in dockerfile_upper or "ENTRYPOINT" in dockerfile_upper
            self._log_execution(
                "COMPLETE",
                f"Generated {len(dockerfile)} chars (user: {has_user}, healthcheck: {has_healthcheck}, cmd: {has_cmd})",
            )

            return dockerfile

        except Exception as e:
            self.logger.error(f"Dockerfile generation failed: {e}", exc_info=True)
            raise ValueError(f"Failed to generate Dockerfile: {str(e)}")

    def _fix_common_cmd_mistakes(self, dockerfile: str, analysis: AnalysisResult) -> str:
        """
        Fix common AI mistakes in CMD directives.

        Common issues:
        1. Using "uv run" when venv is already activated via PATH
        2. Using "poetry run" unnecessarily
        3. Hardcoding port in uvicorn CMD (should read from PORT env var)
        """
        lines = dockerfile.split("\n")
        fixed_lines = []

        for line in lines:
            # Check if this is a CMD line
            if line.strip().startswith("CMD"):
                original_line = line

                # Fix: Remove "uv run" when using uv with activated venv
                if analysis.package_manager == "uv" and '"uv", "run"' in line:
                    # Remove the "uv", "run" part from the CMD
                    line = line.replace('"uv", "run", ', "")
                    self.logger.warning(f"Fixed CMD: Removed 'uv run' (venv already activated)")
                    self.logger.warning(f"  Before: {original_line.strip()}")
                    self.logger.warning(f"  After:  {line.strip()}")

                # Fix: Remove "poetry run" when using poetry with activated venv
                if analysis.package_manager == "poetry" and '"poetry", "run"' in line:
                    line = line.replace('"poetry", "run", ', "")
                    self.logger.warning(f"Fixed CMD: Removed 'poetry run' (venv already activated)")
                    self.logger.warning(f"  Before: {original_line.strip()}")
                    self.logger.warning(f"  After:  {line.strip()}")

                # Fix: Remove hardcoded --port from uvicorn (Cloud Run sets PORT env var)
                if "uvicorn" in line and ("--port" in line or "--workers" in line):
                    import re

                    # Remove --port and its value (e.g., "--port", "8000" or --port 8000)
                    line = re.sub(r',?\s*"--port",\s*"\d+"', "", line)  # JSON array format
                    line = re.sub(r"\s+--port\s+\d+", "", line)  # Shell format
                    # Remove --workers and its value (Cloud Run scales horizontally)
                    line = re.sub(r',?\s*"--workers",\s*"\d+"', "", line)  # JSON array format
                    line = re.sub(r"\s+--workers\s+\d+", "", line)  # Shell format
                    self.logger.warning(
                        f"Fixed CMD: Removed hardcoded --port/--workers (cloud-native)"
                    )
                    self.logger.warning(f"  Before: {original_line.strip()}")
                    self.logger.warning(f"  After:  {line.strip()}")

            # Fix: Update HEALTHCHECK path to match detected health endpoint
            if line.strip().startswith("HEALTHCHECK"):
                original_line = line
                # If analysis has health_check_path and it's not in the HEALTHCHECK, fix it
                if analysis.health_check_path and analysis.health_check_path not in line:
                    import re

                    # Replace /health with the correct path
                    line = re.sub(r'/health(?=["\'\s])', analysis.health_check_path, line)
                    if line != original_line:
                        self.logger.warning(f"Fixed HEALTHCHECK: Updated path to match analysis")
                        self.logger.warning(f"  Before: {original_line.strip()}")
                        self.logger.warning(f"  After:  {line.strip()}")

            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def _cleanup_markdown(self, dockerfile: str) -> str:
        """Remove markdown formatting from Dockerfile."""
        # Remove code blocks
        if "```" in dockerfile:
            match = re.search(r"```(?:dockerfile)?\s*\n(.+?)```", dockerfile, re.DOTALL)
            if match:
                dockerfile = match.group(1).strip()
            else:
                dockerfile = dockerfile.replace("```dockerfile", "").replace("```", "").strip()

        # Find and extract from FROM instruction
        lines = dockerfile.split("\n")
        from_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("FROM") or line.strip().startswith("ARG"):
                from_index = i
                break

        if from_index is not None and from_index > 0:
            dockerfile = "\n".join(lines[from_index:])

        return dockerfile.strip()

    def _strip_instructional_comments(self, dockerfile: str) -> str:
        """
        Aggressively remove instructional comments and setup guidance.
        These are not appropriate for production Dockerfiles in PRs.
        """
        lines = dockerfile.split("\n")
        cleaned_lines = []

        # Patterns that indicate instructional/setup comments (not production-ready)
        instructional_patterns = [
            r"#.*\bAdd to your\b",  # "Add to your main.py"
            r"#.*\bYou need to\b",  # "You need to configure"
            r"#.*\bMake sure to\b",  # "Make sure to update"
            r"#.*\bPlease\b",  # "Please configure"
            r"#.*\bTODO\b",  # "TODO: Configure"
            r"#.*\bFIXME\b",  # "FIXME:"
            r"#.*\bNOTE:.*configure\b",  # "NOTE: Configure your app"
            r"#.*\bINSTRUCTIONS?\b",  # "INSTRUCTIONS" or "INSTRUCTION"
            r"#.*\bDEPLOYMENT INSTRUCTIONS\b",  # "DEPLOYMENT INSTRUCTIONS"
            r"#.*\bSETUP\b.*\bREQUIRED\b",  # "SETUP REQUIRED"
            r"#.*\bExample:\s*\n#\s*```",  # Code examples in comments
        ]

        # Section headers that are instructional (not informative)
        skip_section_headers = [
            "# MONOREPO DEPLOYMENT INSTRUCTIONS",
            "# DEPLOYMENT INSTRUCTIONS",
            "# SETUP GUIDE",
            "# HOW TO USE",
            "# CONFIGURATION REQUIRED",
        ]

        skip_until_blank = False
        for line in lines:
            stripped = line.strip()

            # Skip section headers
            if stripped in skip_section_headers:
                skip_until_blank = True
                continue

            # If we're skipping a section, skip until blank line
            if skip_until_blank:
                if stripped == "" or stripped == "#":
                    skip_until_blank = False
                continue

            # Skip lines matching instructional patterns
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in instructional_patterns):
                continue

            # Keep all non-instructional lines
            cleaned_lines.append(line)

        # Remove excessive blank comment lines (more than 2 consecutive)
        final_lines = []
        consecutive_blank_comments = 0
        for line in cleaned_lines:
            if line.strip() == "#":
                consecutive_blank_comments += 1
                if consecutive_blank_comments <= 2:
                    final_lines.append(line)
            else:
                consecutive_blank_comments = 0
                final_lines.append(line)

        # Remove leading/trailing blank comment lines
        while final_lines and final_lines[0].strip() in ["", "#"]:
            final_lines.pop(0)
        while final_lines and final_lines[-1].strip() in ["", "#"]:
            final_lines.pop()

        return "\n".join(final_lines)
