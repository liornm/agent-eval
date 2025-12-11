"""ACP client backed by Daytona sandbox for file/terminal operations."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Self

from acp import Agent, Client, RequestError, spawn_agent_process
from acp.client.connection import ClientSideConnection
from acp.connection import StreamEvent
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CreateTerminalResponse,
    CurrentModeUpdate,
    EnvVariable,
    FileSystemCapability,
    Implementation,
    KillTerminalCommandResponse,
    ModelInfo,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalExitStatus,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from daytona import AsyncSandbox
from daytona.handle.async_pty_handle import AsyncPtyHandle
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentInfo(BaseModel):
    """Agent implementation info from ACP initialization."""

    name: str = Field(description="Programmatic name of the agent")
    title: str | None = Field(default=None, description="Human-readable display name")
    version: str = Field(description="Version of the agent implementation")

    @classmethod
    def from_implementation(cls, impl: Implementation | None) -> AgentInfo | None:
        """Create AgentInfo from ACP Implementation object."""
        if impl is None:
            return None
        return cls(name=impl.name, title=impl.title, version=impl.version)


class ConversationItem(BaseModel):
    """A single item in the ordered conversation."""

    type: Literal["user", "agent", "tool", "thought"] = Field(description="Item type")
    content: str | dict[str, Any] = Field(description="The text content of the message")
    tool_id: str | None = Field(default=None, description="Unique identifier for tool calls")
    tool_kind: str | None = Field(default=None, description="Type of tool call")
    tool_input: dict[str, Any] | None = Field(
        default=None, description="Input parameters for tool calls"
    )
    tool_output: str | dict[str, Any] | None = Field(
        default=None, description="Output from tool execution"
    )


# Map chunk types to conversation item types
CHUNK_TYPE_MAP: dict[type, str] = {
    UserMessageChunk: "user",
    AgentMessageChunk: "agent",
    AgentThoughtChunk: "thought",
}


# Sentinel to signal end of stream
_END_SENTINEL = object()


@dataclass
class _Terminal:
    """Tracks a PTY terminal session."""

    handle: AsyncPtyHandle
    output: str = ""
    output_limit: int | None = None


class AcpClient(Client):
    """ACP client backed by Daytona sandbox for file/terminal operations."""

    def __init__(
        self,
        sandbox: AsyncSandbox,
        acp_command: str,
        cwd: str,
        auth_method: str | None = None,
        model: str | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._acp_command = acp_command
        self._cwd = cwd
        self._auth_method = auth_method
        self._model = model
        self._terminals: dict[str, _Terminal] = {}
        # Track tool call states for updates
        self._tool_items: dict[str, ConversationItem] = {}
        # Track which tool calls have been emitted (to avoid duplicates)
        self._emitted_tools: set[str] = set()
        # Full conversation built in realtime
        self.conversation: list[ConversationItem] = []
        # Pending chunk for consolidation (agent/thought/user messages)
        self._pending_chunk: ConversationItem | None = None
        # Async queue for streaming items
        self._queue: asyncio.Queue[ConversationItem | object] = asyncio.Queue()
        # Connection state for multi-turn
        self._conn: ClientSideConnection | None = None
        self._session_id: str | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None
        # Agent info from initialization
        self.agent_info: AgentInfo | None = None
        # Available models from session creation
        self.available_models: list[ModelInfo] = []
        # Current model ID from session creation
        self.current_model_id: str | None = None

    def _emit_item(self, item: ConversationItem) -> None:
        """Emit an item to the queue."""
        self._queue.put_nowait(item)

    def _emit_pending(self) -> None:
        """Emit any pending consolidated chunk."""
        if self._pending_chunk is not None:
            self.conversation.append(self._pending_chunk)
            self._emit_item(self._pending_chunk)
            self._pending_chunk = None

    def _handle_chunk(
        self, item_type: Literal["user", "agent", "tool", "thought"], text: str
    ) -> None:
        """Handle a text chunk, consolidating consecutive chunks of the same type."""
        if self._pending_chunk is not None and self._pending_chunk.type == item_type:
            # Append to existing pending chunk
            self._pending_chunk = ConversationItem(
                type=item_type,
                content=str(self._pending_chunk.content) + text,
            )
        else:
            # Emit previous pending chunk and start a new one
            self._emit_pending()
            self._pending_chunk = ConversationItem(type=item_type, content=text)

    def _handle_tool_start(self, update: ToolCallStart) -> None:
        """Handle a tool call start, creating a new tool item."""
        self._emit_pending()  # Tool calls break consolidation
        item = ConversationItem(
            type="tool",
            content="",
            tool_id=update.tool_call_id,
            tool_kind=update.kind,
            tool_input=update.raw_input,
            tool_output=update.raw_output,
        )
        self._tool_items[update.tool_call_id] = item
        self.conversation.append(item)
        # Don't emit yet - wait for tool completion (raw_output) in _handle_tool_progress

    def _handle_tool_progress(self, update: ToolCallProgress) -> None:
        """Handle a tool call progress update, updating the existing tool item."""
        item = self._tool_items.get(update.tool_call_id)
        if item is None:
            return
        if update.raw_input is not None:
            item.tool_input = update.raw_input
        if update.raw_output is not None:
            item.tool_output = update.raw_output
        # Only emit once when tool has output and hasn't been emitted yet
        if item.tool_output is not None and update.tool_call_id not in self._emitted_tools:
            self._emitted_tools.add(update.tool_call_id)
            self._emit_item(item)

    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate,
        **kwargs: dict[str, Any],
    ) -> None:
        """Process a session update and build conversation in realtime."""
        logger.debug("ACP <- session_update: %s", update)
        del session_id, kwargs  # unused
        if (item_type := CHUNK_TYPE_MAP.get(type(update))) is not None:
            if isinstance(update.content, TextContentBlock):
                text = (
                    update.content.text
                    if update.content.type == "text"
                    else f"[{update.content.type}]"
                )
            else:
                raise ValueError(f"Unsupported content type: {type(update.content)}")
            self._handle_chunk(item_type, text)  # type: ignore[arg-type]
        elif isinstance(update, ToolCallStart):
            self._handle_tool_start(update)
        elif isinstance(update, ToolCallProgress):
            self._handle_tool_progress(update)

    async def _build_ssh_command(self) -> tuple[str, ...]:
        """Build SSH command for connecting to the agent process."""
        ssh = await self._sandbox.create_ssh_access()
        ssh_target = f"{ssh.token}@ssh.app.daytona.io"
        remote_cmd = f"cd {self._cwd} && {self._acp_command}".strip()
        ssh_opts = ("-tt", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null")
        return ("ssh", *ssh_opts, ssh_target, remote_cmd)

    def _json_rpc_observer(self, event: StreamEvent) -> None:
        """Log raw JSON-RPC messages."""
        direction = "<-" if event.direction.value == "incoming" else "->"
        logger.debug("ACP %s JSON-RPC: %s", direction, json.dumps(event.message, indent=2))

    async def _spawn_connection(self, ssh_cmd: tuple[str, ...]) -> ClientSideConnection:
        """Spawn agent process and return the connection."""
        ctx = spawn_agent_process(self, *ssh_cmd, transport_kwargs={"limit": 2**24})
        conn, _ = await self._exit_stack.enter_async_context(ctx)  # type: ignore[union-attr]
        conn._conn.add_observer(self._json_rpc_observer)  # noqa: SLF001
        return conn

    async def _initialize_session(self, conn: ClientSideConnection) -> None:
        """Initialize ACP session with authentication and model selection."""
        client_capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=True),
            terminal=True,
        )
        logger.debug(
            "ACP -> initialize(protocol_version=1, client_capabilities=%s)",
            client_capabilities,
        )
        init_resp = await conn.initialize(
            protocol_version=1,
            client_capabilities=client_capabilities,
        )
        logger.debug("ACP <- initialize: %s", init_resp)
        self.agent_info = AgentInfo.from_implementation(init_resp.agent_info)

        if self._auth_method is not None:
            logger.debug("ACP -> authenticate(method_id=%r)", self._auth_method)
            await conn.authenticate(method_id=self._auth_method)

        logger.debug("ACP -> new_session(cwd=%r)", self._cwd)
        session_resp = await conn.new_session(cwd=self._cwd, mcp_servers=[])
        logger.debug("ACP <- new_session: %s", session_resp)
        self._session_id = session_resp.session_id

        if session_resp.models is not None:
            self.available_models = session_resp.models.available_models
            self.current_model_id = session_resp.models.current_model_id

        await self._select_model(conn)

    async def _select_model(self, conn: ClientSideConnection) -> None:
        """Select the requested model if specified and available."""
        if not self._model or not self.available_models or not self._session_id:
            return

        available_ids = {m.model_id for m in self.available_models}
        if self._model in available_ids:
            logger.debug(
                "ACP -> set_session_model(session_id=%r, model_id=%r)",
                self._session_id,
                self._model,
            )
            await conn.set_session_model(session_id=self._session_id, model_id=self._model)
            self.current_model_id = self._model
        else:
            model_list = "\n".join(f"  • {m.model_id} ({m.name})" for m in self.available_models)
            logger.error("Model %r not available. Available:\n%s", self._model, model_list)
            raise ValueError(f"Model '{self._model}' not available")

    async def _cleanup_connection(self) -> None:
        """Gracefully close the connection and exit stack."""
        conn = self._conn
        self._conn = None
        self._session_id = None
        stack = self._exit_stack
        self._exit_stack = None

        if conn is not None:
            # Yield to let the receive loop process any pending items
            await asyncio.sleep(0)
            with contextlib.suppress(Exception):
                await conn.close()

        if stack is not None:
            # Small delay to let background tasks finish after close
            await asyncio.sleep(0.01)
            with contextlib.suppress(Exception):
                await stack.__aexit__(None, None, None)

    async def __aenter__(self) -> Self:
        """Establish SSH connection to the agent."""
        self._exit_stack = contextlib.AsyncExitStack()
        await self._exit_stack.__aenter__()
        try:
            ssh_cmd = await self._build_ssh_command()
            self._conn = await self._spawn_connection(ssh_cmd)
            await self._initialize_session(self._conn)
        except BaseException:
            await self._cleanup_connection()
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the SSH connection."""
        await self._cleanup_connection()

    def _ensure_connected(self) -> None:
        """Raise RuntimeError if not connected."""
        if self._conn is None or self._session_id is None:
            msg = "Connection not established. Use 'async with client.connect():' first."
            raise RuntimeError(msg)

    async def _send_prompt(self, message: str) -> None:
        """Send prompt and signal completion."""
        self._ensure_connected()
        logger.debug("ACP -> prompt(session_id=%r, message=%r)", self._session_id, message)
        await self._conn.prompt(  # type: ignore[union-attr]
            session_id=self._session_id,  # type: ignore[arg-type]
            prompt=[TextContentBlock(type="text", text=message)],
        )
        self._emit_pending()
        self._queue.put_nowait(_END_SENTINEL)

    async def _stream_items(self, task: asyncio.Task[None]) -> AsyncIterator[ConversationItem]:
        """Yield conversation items as they arrive, handling task failures."""
        while True:
            get_task = asyncio.create_task(self._queue.get())
            done, _ = await asyncio.wait([get_task, task], return_when=asyncio.FIRST_COMPLETED)
            if task in done and get_task not in done:
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                await task  # Re-raise if prompt had an exception
                break
            item = get_task.result()
            if item is _END_SENTINEL:
                break
            if isinstance(item, ConversationItem):
                yield item

    async def run(self, message: str) -> AsyncIterator[ConversationItem]:
        """Run agent with a user message and yield conversation items.

        Must be called within a connect() context. Can be called multiple times
        for follow-up messages within the same connection.
        """
        self._ensure_connected()
        self._handle_chunk("user", message)
        self._emit_pending()

        task = asyncio.create_task(self._send_prompt(message))
        try:
            async for item in self._stream_items(task):
                yield item
        except RequestError as e:
            if (
                e.code == RequestError.internal_error().code
                and "fetch failed" in str(e.data or "").lower()
            ):
                logger.exception(
                    "Daytona sandbox may be blocking network access to the agent's API. "
                    "Contact support@daytona.io to request whitelisting or upgrade to Tier 3+."
                )
            raise
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    def on_connect(self, conn: Agent) -> None:
        logger.debug("Connected to agent %r", conn)

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: dict[str, Any],
    ) -> RequestPermissionResponse:
        del session_id, tool_call, kwargs  # unused
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=options[0].option_id)
        )

    async def write_text_file(
        self,
        content: str,
        path: str,
        session_id: str,
        **kwargs: dict[str, Any],
    ) -> WriteTextFileResponse | None:
        del session_id, kwargs  # unused
        await self._sandbox.fs.upload_file(content.encode(), path)
        return WriteTextFileResponse()

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: dict[str, Any],
    ) -> ReadTextFileResponse:
        del session_id, kwargs  # unused
        data = await self._sandbox.fs.download_file(path)
        content = (data or b"").decode(errors="replace")
        if line is not None:
            lines = content.splitlines()
            start = max(0, line - 1)
            end = start + (limit or 100)
            content = "\n".join(lines[start:end])
        elif limit is not None:
            content = content[:limit]
        return ReadTextFileResponse(content=content)

    async def create_terminal(  # noqa: PLR0913
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: dict[str, Any],
    ) -> CreateTerminalResponse:
        del session_id, kwargs  # unused
        terminal_id = str(uuid.uuid4())
        env_dict = {e.name: e.value for e in (env or [])} if env else None

        def on_data(data: bytes) -> None:
            term = self._terminals.get(terminal_id)
            if term:
                term.output += data.decode(errors="replace")
                if term.output_limit and len(term.output) > term.output_limit:
                    term.output = term.output[: term.output_limit]

        handle = await self._sandbox.process.create_pty_session(
            id=terminal_id, on_data=on_data, cwd=cwd, envs=env_dict
        )
        self._terminals[terminal_id] = _Terminal(handle=handle, output_limit=output_byte_limit)
        # Send command to execute
        full_cmd = f"{command} {' '.join(args or [])}\n"
        await handle.send_input(full_cmd)
        return CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: dict[str, Any],
    ) -> TerminalOutputResponse:
        del session_id, kwargs  # unused
        term = self._terminals.get(terminal_id)
        if not term:
            return TerminalOutputResponse(output="", truncated=False)
        truncated = bool(term.output_limit and len(term.output) >= term.output_limit)
        exit_status = None
        if not term.handle.is_connected():
            result = await term.handle.wait()
            exit_status = TerminalExitStatus(exit_code=result.exit_code)
        return TerminalOutputResponse(
            output=term.output, truncated=truncated, exit_status=exit_status
        )

    async def release_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: dict[str, Any],
    ) -> ReleaseTerminalResponse | None:
        del session_id, kwargs  # unused
        term = self._terminals.pop(terminal_id, None)
        if term:
            await term.handle.disconnect()
        return ReleaseTerminalResponse()

    async def wait_for_terminal_exit(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: dict[str, Any],
    ) -> WaitForTerminalExitResponse:
        del session_id, kwargs  # unused
        term = self._terminals.get(terminal_id)
        if not term:
            return WaitForTerminalExitResponse()
        result = await term.handle.wait()
        return WaitForTerminalExitResponse(exit_code=result.exit_code)

    async def kill_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: dict[str, Any],
    ) -> KillTerminalCommandResponse | None:
        del session_id, kwargs  # unused
        term = self._terminals.pop(terminal_id, None)
        if term:
            await term.handle.kill()
        return KillTerminalCommandResponse()

    async def ext_method(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        del method, params  # unused
        return {}

    async def ext_notification(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        del method, params  # unused
