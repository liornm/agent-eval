"""Rollout executor for running agents in Daytona sandboxes."""

import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaConfig,
    Image,
)
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    computed_field,
    model_serializer,
)

from acp_client import AcpClient, AgentInfo, ConversationItem

logger = logging.getLogger(__name__)


class RolloutConfig(BaseModel):
    """Configuration for executing a rollout."""

    agent: str = Field(default="codex", description="Agent executor to use")
    model: str | None = Field(default=None, description="Model to use (agent-specific)")
    repo: str | None = Field(default=None, description="Git repository URL to clone (optional)")
    commit: str | None = Field(default=None, description="Git commit to checkout")
    docker_image: str | None = Field(
        default=None, description="Docker image to use for sandbox (e.g., from SWE-bench Live)"
    )
    max_turns: int = Field(default=50, description="Maximum turns before stopping")
    auto_delete_interval: int = Field(
        default=120, description="Sandbox auto-delete interval (seconds)"
    )
    files_to_upload: dict[Path, str | bytes] = Field(
        default_factory=dict,
        description="Files to upload to sandbox: {sandbox_path: content}",
    )


class RolloutResult(BaseModel):
    """Result of a rollout execution (order matters)."""

    rollout_id: str = Field(
        default_factory=lambda: f"rollout_{uuid.uuid4().hex[:8]}", description="Unique identifier"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Timestamp"
    )
    agent_info: AgentInfo | None = Field(
        default=None, description="Agent implementation info from ACP initialization"
    )
    model: str | None = Field(default=None, description="Model used for the rollout")
    turn_count: int = Field(description="Number of turns completed")
    stopped_early: bool = Field(default=False, description="Whether stopped before completion")
    git_diff: str = Field(default="", description="Git diff of changes made during rollout")
    conversation: list[ConversationItem] = Field(description="Full conversation history")

    @computed_field
    @property
    def final_message(self) -> str | None:
        """The final message from the agent, or None if no agent messages exist."""
        for item in reversed(self.conversation):
            if item.type == "agent":
                return str(item.content) if item.content else None
        return None

    @computed_field
    @property
    def tool_histogram(self) -> defaultdict[str, int]:
        """The final message from the agent, or None if no agent messages exist."""
        hist = defaultdict(int)
        for item in self.conversation:
            if item.type == "tool":
                hist[item.tool_kind] += 1
        return hist

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict:
        """Ensure field order for visualization."""
        data = handler(self)
        return {
            "rollout_id": data["rollout_id"],
            "timestamp": data["timestamp"],
            "agent_info": data["agent_info"],
            "model": data["model"],
            "turn_count": data["turn_count"],
            "stopped_early": data["stopped_early"],
            "git_diff": data["git_diff"],
            "final_message": data["final_message"],
            "tool_histogram": data["tool_histogram"],
            "conversation": data["conversation"],
        }


class DaytonaSettings(BaseModel):
    """Daytona API configuration."""

    api_key: str | None = Field(default=None, description="API key (defaults to DAYTONA_API_KEY)")
    api_url: str = Field(default="https://app.daytona.io/api", description="Daytona API URL")
    target: str = Field(default="us", description="Daytona target region")


# Type alias for hooks
OnMessageHook = Callable[[ConversationItem], None]
OnResponseHook = Callable[[RolloutResult], str | None]
OnStatusHook = Callable[[str], None]


class SandboxedRolloutExecutor(ABC):
    """Abstract rollout executor that manages its own Daytona sandbox lifecycle."""

    def __init__(
        self,
        config: RolloutConfig,
        on_message: OnMessageHook | None = None,
        on_response: OnResponseHook | None = None,
        on_status: OnStatusHook | None = None,
        daytona: DaytonaSettings | None = None,
    ) -> None:
        self.config = config
        self._on_message = on_message
        self._on_response = on_response
        self._on_status = on_status
        self._daytona = daytona or DaytonaSettings()

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def command(self) -> str: ...

    @property
    def auth_method(self) -> str | None:
        """Auth method ID. Return None to skip authentication (default)."""
        return None

    def _status(self, status: str) -> None:
        """Emit a status update."""
        if self._on_status:
            self._on_status(status)

    @asynccontextmanager
    async def _sandbox(self) -> AsyncIterator[AsyncSandbox]:
        """Create and manage a Daytona sandbox."""
        logger.debug("Initializing Daytona client (target=%s)", self._daytona.target)
        api_key = self._daytona.api_key or os.environ["DAYTONA_API_KEY"]
        daytona = AsyncDaytona(
            DaytonaConfig(
                api_key=api_key, api_url=self._daytona.api_url, target=self._daytona.target
            )
        )
        env_vars = {k: v for k, v in dotenv_values().items() if v}
        logger.debug("Loaded %d environment variables from .env", len(env_vars))

        if self.config.docker_image:
            logger.debug("Creating sandbox from Docker image: %s", self.config.docker_image)
            self._status("Building image...")

            def _log_build(chunk: str) -> None:
                if logger.isEnabledFor(logging.DEBUG):
                    print(chunk, end="", flush=True)  # noqa: T201

            sandbox = await daytona.create(
                CreateSandboxFromImageParams(
                    image=Image.base(self.config.docker_image),
                    env_vars=env_vars,
                    auto_delete_interval=self.config.auto_delete_interval,
                ),
                on_snapshot_create_logs=_log_build,
            )
        else:
            logger.debug("Creating sandbox from default snapshot")
            self._status("Creating sandbox...")
            sandbox = await daytona.create(
                CreateSandboxFromSnapshotParams(
                    env_vars=env_vars,
                    auto_delete_interval=self.config.auto_delete_interval,
                )
            )
        logger.debug("Sandbox created: id=%s", sandbox.id)
        self._status("Sandbox ready")
        try:
            await self._upload_files(sandbox)
            yield sandbox
        finally:
            logger.debug("Deleting sandbox: id=%s", sandbox.id)
            await daytona.delete(sandbox)

    @property
    def executor_files(self) -> dict[Path, str | bytes]:
        """Files the executor needs to upload. Override in subclasses."""
        return {}

    async def _upload_files(self, sandbox: AsyncSandbox) -> None:
        """Upload configured files to the sandbox."""
        created_dirs: set[Path] = set()
        home = await sandbox.get_user_home_dir()
        all_files = {**self.executor_files, **self.config.files_to_upload}
        if all_files:
            logger.debug("Uploading %d files to sandbox", len(all_files))
        for raw_path, content in all_files.items():
            # Expand ~ to home directory
            dest_path = (
                Path(home) / raw_path.relative_to("~") if raw_path.parts[0] == "~" else raw_path
            )
            # Ensure parent directory exists
            parent_dir = dest_path.parent
            if parent_dir and parent_dir not in created_dirs:
                await sandbox.process.exec(f"mkdir -p {parent_dir}")
                created_dirs.add(parent_dir)
            # Convert string content to bytes (SDK interprets str as file path)
            content_bytes = content.encode("utf-8") if isinstance(content, str) else content
            # Upload file content
            logger.debug("Uploading file: %s", dest_path)
            await sandbox.fs.upload_file(content_bytes, str(dest_path))

    async def _setup_repo(self, sandbox: AsyncSandbox) -> tuple[str, str | None]:
        """Clone the repository and checkout the specified commit."""
        home = await sandbox.get_user_home_dir()
        if self.config.repo:
            clone_path = f"{home}/{uuid.uuid4().hex}"
            logger.debug("Cloning repository %s to %s", self.config.repo, clone_path)
            await sandbox.git.clone(url=self.config.repo, path=clone_path)

            if self.config.commit:
                logger.debug("Fetching and checking out commit %s", self.config.commit)
                await sandbox.process.exec(f"git fetch origin {self.config.commit}", cwd=clone_path)
                await sandbox.process.exec(f"git checkout {self.config.commit}", cwd=clone_path)

            # Capture base commit for diff calculation
            resp = await sandbox.process.exec("git rev-parse HEAD", cwd=clone_path)
            base_commit = (
                resp.artifacts.stdout.strip() if resp.artifacts and resp.artifacts.stdout else None
            )
            logger.debug("Repository ready at %s (base_commit=%s)", clone_path, base_commit)
            return clone_path, base_commit

        # Use home directory, or current working directory for Docker images
        if self.config.docker_image:
            resp = await sandbox.process.exec("pwd")
            cwd = (
                resp.artifacts.stdout.strip() if resp.artifacts and resp.artifacts.stdout else home
            )
            logger.debug("Using Docker image working directory: %s", cwd)
            return cwd, None

        return home, None

    async def _collect_git_diff(
        self, sandbox: AsyncSandbox, cwd: str, base_commit: str | None
    ) -> str:
        """Collect git diff from the sandbox, including untracked files."""
        if not base_commit:
            return ""
        # Stage all files (including untracked) to include them in the diff
        await sandbox.process.exec("git add -A", cwd=cwd)
        resp = await sandbox.process.exec(f"git diff --cached {base_commit}", cwd=cwd)
        return resp.artifacts.stdout if resp.artifacts and resp.artifacts.stdout else ""

    async def _ensure_node(self, sandbox: AsyncSandbox) -> None:
        """Ensure Node.js is available, installing via prebuilt binary if needed."""
        resp = await sandbox.process.exec("which npx")
        if resp.exit_code == 0:
            return
        logger.debug("Node.js not found, installing prebuilt binary...")
        node_version = "v22.12.0"
        install_cmd = (
            f"curl -fsSL https://nodejs.org/dist/{node_version}/node-{node_version}-linux-x64.tar.xz"
            " | tar -xJ -C /usr/local --strip-components=1"
        )
        resp = await sandbox.process.exec(install_cmd, timeout=300)
        if resp.exit_code != 0:
            raise RuntimeError(f"Failed to install Node.js: {resp.result}")
        # Verify npx is now available
        resp = await sandbox.process.exec("which npx")
        if resp.exit_code != 0:
            raise RuntimeError("Node.js installed but npx not found in PATH")
        logger.debug("Node.js installed successfully")

    async def execute(self, prompt: str) -> RolloutResult:
        """Execute a rollout with full sandbox lifecycle management."""
        async with self._sandbox() as sandbox:
            await self._ensure_node(sandbox)
            cwd, base_commit = await self._setup_repo(sandbox)

            async with AcpClient(
                sandbox,
                acp_command=self.command,
                cwd=cwd,
                auth_method=self.auth_method,
                model=self.config.model,
            ) as client:
                result = await self._run_conversation(client, prompt)
                agent_info = client.agent_info
                model = client.current_model_id

            git_diff = await self._collect_git_diff(sandbox, cwd, base_commit)
            return RolloutResult(
                agent_info=agent_info,
                model=model,
                conversation=result.conversation,
                turn_count=result.turn_count,
                stopped_early=result.stopped_early,
                git_diff=git_diff,
            )

    async def _run_conversation(self, client: AcpClient, prompt: str) -> RolloutResult:
        """Run a multi-turn conversation until completion or max turns."""
        turn_count = 0

        while True:
            async for item in client.run(prompt):
                if self._on_message:
                    self._on_message(item)
                turn_count += 1
                if turn_count >= self.config.max_turns:
                    break

            result = RolloutResult(
                agent_info=client.agent_info,
                model=client.current_model_id,
                conversation=client.conversation,
                turn_count=turn_count,
                stopped_early=turn_count >= self.config.max_turns,
            )
            if not self._on_response or (new_prompt := self._on_response(result)) is None:
                return result
            prompt = new_prompt


class CodexRolloutExecutor(SandboxedRolloutExecutor):
    @property
    def name(self) -> str:
        return "codex"

    @property
    def command(self) -> str:
        return (
            "npx -y @zed-industries/codex-acp"
            " -c approval_policy=never -c sandbox_mode=danger-full-access"
        )

    @property
    def auth_method(self) -> str:
        return "codex-api-key" if os.environ.get("CODEX_API_KEY") else "openai-api-key"


class ClaudeCodeRolloutExecutor(SandboxedRolloutExecutor):
    @property
    def name(self) -> str:
        return "claude-code"

    @property
    def command(self) -> str:
        return "npx -y @zed-industries/claude-code-acp"


class GeminiCliRolloutExecutor(SandboxedRolloutExecutor):
    @property
    def name(self) -> str:
        return "gemini"

    @property
    def command(self) -> str:
        return "npx -y @google/gemini-cli --experimental-acp"

    @property
    def auth_method(self) -> str | None:
        return "gemini-api-key"


class OpenCodeRolloutExecutor(SandboxedRolloutExecutor):
    @property
    def name(self) -> str:
        return "opencode"

    @property
    def command(self) -> str:
        return "npx -y opencode-ai@latest acp"


class AugmentCodeRolloutExecutor(SandboxedRolloutExecutor):
    @property
    def name(self) -> str:
        return "auggie"

    @property
    def command(self) -> str:
        return "npx -y @augmentcode/auggie --acp"


EXECUTORS: dict[str, type[SandboxedRolloutExecutor]] = {
    "codex": CodexRolloutExecutor,
    "claude-code": ClaudeCodeRolloutExecutor,
    "gemini": GeminiCliRolloutExecutor,
    "opencode": OpenCodeRolloutExecutor,
    "auggie": AugmentCodeRolloutExecutor,
}


async def run_rollout(
    prompt: str,
    config: RolloutConfig,
    on_message: OnMessageHook | None = None,
    on_response: OnResponseHook | None = None,
    on_status: OnStatusHook | None = None,
) -> RolloutResult:
    executor_cls = EXECUTORS.get(config.agent)
    if executor_cls is None:
        raise ValueError(f"Unknown agent: {config.agent}. Available: {list(EXECUTORS.keys())}")
    executor = executor_cls(
        config, on_message=on_message, on_response=on_response, on_status=on_status
    )
    return await executor.execute(prompt)
