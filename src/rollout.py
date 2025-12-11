"""Generate a group of rollouts for a task and save them to a directory."""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, ClassVar

import hjson
from dotenv import load_dotenv
from pydantic import BaseModel, BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, CliPositionalArg
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.traceback import install

from acp_client import ConversationItem
from rollout_executor import RolloutConfig, RolloutResult, run_rollout
from swebench_live import sample_task

logger = logging.getLogger(__name__)

install(show_locals=True)
load_dotenv()


class AgentSpec(BaseModel):
    """Specification for an agent to run."""

    name: str
    model: str | None = None

    @classmethod
    def parse(cls, s: str) -> "AgentSpec":
        name, _, model = s.strip().partition(":")
        return cls(name=name, model=model or None)

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}" if self.model else self.name


def _parse_agents(v: str | list) -> list[AgentSpec]:
    """Parse 'agent:model,agent:model,...' into AgentSpec list."""
    if isinstance(v, str):
        return [AgentSpec.parse(p) for p in v.split(",")]
    return [AgentSpec.parse(p) if isinstance(p, str) else p for p in v]


AgentsList = Annotated[list[AgentSpec], BeforeValidator(_parse_agents)]


class GroupSettings(BaseSettings, cli_implicit_flags=True):
    """CLI settings for generating rollout groups."""

    task: CliPositionalArg[str | None] = Field(default=None, description="Task prompt")
    agents: AgentsList = Field(
        default=[AgentSpec(name="codex")],
        alias="a",
        description="Agents as 'agent:model' pairs, comma-separated (e.g. 'codex:o3,claude-code')",
    )
    num_rollouts: int = Field(
        default=1, alias="n", description="Number of rollouts per agent/model"
    )
    repo: str | None = Field(default=None, description="Git repo URL")
    commit: str | None = Field(default=None, description="Git commit to checkout")
    docker_image: str | None = Field(default=None, description="Docker image for sandbox")
    output: Path = Field(
        default_factory=lambda: Path(
            f"output/rollouts_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
        ),
        alias="o",
        description="Output directory for rollouts",
    )
    max_turns: int = Field(default=100, description="Maximum turns per rollout")
    test: bool = Field(default=False, description="Run in test mode with hardcoded values")
    swebench_live: bool = Field(
        default=False, alias="s", description="Sample a random task from SWE-bench Live dataset"
    )
    debug: bool = Field(default=False, description="Enable debug logging (shows ACP traffic)")

    TEST_TASK: ClassVar[str] = (
        "fix bug in test client when using follow_redirects the session shows the state "
        "from the first request instead of the last one. think the contexts are being "
        "restored in the wrong order after redirects. update test_redirect_keep_session "
        "to verify the fix and add changelog entry"
    )
    TEST_REPO: ClassVar[str] = "https://github.com/pallets/flask"
    TEST_COMMIT: ClassVar[str] = "5addaf833b2e8c7a616f89dd8ad5a44b07d7c000"

    @model_validator(mode="after")
    def apply_defaults(self) -> "GroupSettings":
        if self.swebench_live:
            # Sample a random task from SWE-bench Live
            task = sample_task()
            self.task = f"Address the following issue:\n{task.problem_statement}"
            self.repo = task.repo_url
            self.commit = task.base_commit
            self.docker_image = task.docker_image
        elif self.test:
            self.task = self.task or self.TEST_TASK
            self.repo = self.repo or self.TEST_REPO
            self.commit = self.commit or self.TEST_COMMIT
        elif not self.task:
            raise ValueError("task is required (or use --test or --swebench-live)")
        return self


def create_progress() -> Progress:
    """Create a Rich Progress instance for tracking rollout tasks."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[turns]}[/cyan] turns"),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=Console(),
        disable=logger.isEnabledFor(logging.DEBUG),
    )


async def generate_rollouts(
    args: GroupSettings,
) -> tuple[list[RolloutResult], list[Exception]]:
    """Generate N rollouts per agent in parallel with live progress display."""
    base_config = RolloutConfig(
        repo=args.repo,
        commit=args.commit,
        docker_image=args.docker_image,
        max_turns=args.max_turns,
    )
    progress = create_progress()
    configs = [
        (base_config.model_copy(update={"agent": a.name, "model": a.model}), a.label)
        for a in args.agents
        for _ in range(args.num_rollouts)
    ]
    task_ids = [
        progress.add_task(f"[green]{lbl}[/green]", total=args.max_turns, completed=0, turns=0)
        for _, lbl in configs
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "task.txt").write_text(args.task or "")

    async def run_one(idx: int) -> RolloutResult:
        config, label = configs[idx]
        task_id = task_ids[idx]
        turns = 0

        def on_msg(_: ConversationItem) -> None:
            nonlocal turns
            turns += 1
            progress.update(task_id, completed=turns, turns=turns)

        def on_status(status: str) -> None:
            progress.update(task_id, description=f"[dim]{status}[/dim] {label}")

        try:
            if args.task is None:
                raise ValueError("task is required")
            result = await run_rollout(args.task, config, on_message=on_msg, on_status=on_status)
        except Exception:
            if args.debug:
                Console().print_exception()
            progress.update(task_id, description=f"[bold red]✗ {label}[/bold red]")
            raise
        with (args.output / f"rollout_{idx}.hjson").open("w") as f:
            hjson.dump(result.model_dump(), f, indent=2)
        progress.update(
            task_id,
            description=f"[bold green]✓ {label}[/bold green]",
            completed=args.max_turns,
        )
        return result

    with progress:
        results = await asyncio.gather(
            *[run_one(i) for i in range(len(configs))], return_exceptions=True
        )

    rollouts = [r for r in results if isinstance(r, RolloutResult)]
    errors = [r for r in results if isinstance(r, Exception)]
    return rollouts, errors


def setup_logging(*, debug: bool) -> None:
    """Configure logging with optional debug level."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=debug)],
    )


async def main() -> None:
    console = Console()
    try:
        args = GroupSettings(_cli_parse_args=True)  # type: ignore[call-arg]
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    setup_logging(debug=args.debug)

    if args.swebench_live:
        console.print(
            Panel(
                f"[bold]Repo:[/bold] {args.repo}\n"
                f"[bold]Commit:[/bold] {args.commit[:12] if args.commit else 'N/A'}\n"
                f"[bold]Docker:[/bold] {args.docker_image}",
                title="[cyan]SWE-bench Live Task[/cyan]",
                border_style="cyan",
            )
        )
    elif args.test:
        console.print("[yellow]Running in test mode with hardcoded values[/yellow]")

    n, n_agents = args.num_rollouts, len(args.agents)
    console.print(f"[bold cyan]Generating {n * n_agents} rollouts ({n} x {n_agents} agents)[/]")
    rollouts, errors = await generate_rollouts(args)

    for err in errors:
        console.print(f"[red]Rollout failed: {err}[/red]")
    for r in rollouts:
        console.print(
            Panel(
                f"Turns: {r.turn_count}\nGit diff: {len(r.git_diff)} chars",
                title=f"{r.rollout_id}",
                border_style="blue",
            )
        )
    console.print(f"\n[bold green]Saved {len(rollouts)} rollouts to {args.output}[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
