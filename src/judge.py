"""Judge script: loads rollouts from a directory and ranks them using a judge agent."""

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path

import hjson
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, CliPositionalArg
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.traceback import install

from acp_client import ConversationItem
from metrics import ALL_METRICS, Metric
from rollout_executor import CodexRolloutExecutor, RolloutConfig, RolloutResult

METRICS_BY_NAME = {m.name.lower(): m for m in ALL_METRICS}

install(show_locals=True)
load_dotenv()

MAX_TURNS = 100
JUDGE_FILES_DIR = "~"
# Truncation limits for display
CONTENT_TRUNCATE_THRESHOLD = 2000
FINAL_MESSAGE_TRUNCATE_LENGTH = 200


class JudgeSettings(BaseSettings, cli_implicit_flags=True):
    """CLI settings for the judge."""

    rollouts_dir: CliPositionalArg[Path] = Field(description="Directory containing rollouts")
    metrics: str = Field(
        default=",".join(m.name for m in ALL_METRICS),
        alias="m",
        description="Comma-separated list of metric names to evaluate",
    )


class RolloutScore(BaseModel):
    """Score for a single rollout on a single metric."""

    rollout_id: str
    score: int = Field(ge=1, le=10)
    reasoning: str


class MetricResult(BaseModel):
    """Result of evaluating all rollouts on a single metric."""

    metric_name: str
    scores: list[RolloutScore]


class MetricStats(BaseModel):
    """Aggregated statistics for a single metric."""

    metric_name: str
    avg: float
    std: float


class AggregatedJudgeResult(BaseModel):
    """Aggregated results across all metrics."""

    metric_results: list[MetricResult]

    @computed_field
    @property
    def total_scores(self) -> dict[str, int]:
        """Sum of scores per rollout across all metrics."""
        totals: dict[str, int] = defaultdict(int)
        for r in self.metric_results:
            for s in r.scores:
                totals[s.rollout_id] += s.score
        return dict(totals)

    @computed_field
    @property
    def rankings(self) -> list[str]:
        """Rollout IDs ordered best to worst."""
        return sorted(self.total_scores, key=self.total_scores.get, reverse=True)  # type: ignore[arg-type]

    @computed_field
    @property
    def stats(self) -> list[MetricStats]:
        """Average and standard deviation for each metric."""
        result = []
        for mr in self.metric_results:
            scores = np.array([s.score for s in mr.scores])
            result.append(
                MetricStats(
                    metric_name=mr.metric_name,
                    avg=float(scores.mean()),
                    std=float(scores.std()),
                )
            )
        return result


def _truncate(text: str, max_lines: int = 50) -> str:
    """Truncate text in the middle, keeping first and last lines."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    n = max_lines // 2
    return "\n".join(
        [*lines[:n], f"... ({len(lines) - max_lines} lines truncated) ...", *lines[-n:]]
    )


def print_judge_message(console: Console, item: ConversationItem) -> None:
    """Print a judge conversation item with Rich formatting."""
    styles = {
        "user": ("bold cyan", "📋 PROMPT"),
        "agent": ("bold green", "🤖 JUDGE"),
        "thought": ("bold magenta", "💭 THINKING"),
        "tool": ("bold yellow", "🔧 TOOL"),
    }
    style, title = styles.get(item.type, ("bold white", item.type.upper()))

    if item.type == "tool":
        title = f"{title} ({item.tool_kind})" if item.tool_kind else title
        parts: list[Text | Syntax] = []
        if item.tool_input:
            parts += [
                Text("\n📥 INPUT:", style="bold"),
                Syntax(_truncate(hjson.dumps(item.tool_input, indent=2)), "text"),
            ]
        if item.tool_output:
            parts.append(Text("\n📤 OUTPUT:", style="bold"))
            if isinstance(item.tool_output, dict):
                parts.append(Syntax(_truncate(hjson.dumps(item.tool_output, indent=2)), "text"))
            else:
                parts.append(Text(_truncate(str(item.tool_output)), style="dim"))
        content = Group(*parts)
    else:
        text = str(item.content or "")
        if len(text) > CONTENT_TRUNCATE_THRESHOLD:
            content = text[:1000] + "\n\n... (truncated) ...\n\n" + text[-500:]
        else:
            content = text

    console.print(Panel(content, title=title, border_style=style, expand=True), "")


INITIAL_PROMPT = """\
You are a code review judge evaluating agent rollouts.

## Task
The original task prompt is in: {task_file}

## Rollouts
Rollout files are located at:
{rollout_files_list}

Each rollout HJSON file contains:
- rollout_id: Unique identifier for the rollout
- conversation: The full agent conversation (list of messages with type, content, tool info)
- git_diff: The code changes made by the agent
- final_message: The agent's final response

## Instructions
1. Read the task file to understand what was asked
2. Read all rollout files to understand what each agent produced
3. Thoroughly analyze the codebase to gather context before evaluating:
   - Explore the repository structure to understand the project
   - Read relevant source files referenced in the rollouts
   - Understand the existing code patterns and conventions
   - Check how the changes integrate with the existing codebase
4. Only after gathering sufficient context, I will ask you to evaluate each rollout
   on specific metrics
"""

METRIC_PROMPT = """\
Now evaluate all rollouts on this metric:

{metric}

## Instructions
1. Review each rollout's conversation and git_diff
2. Score each rollout from 1-10 on this specific metric
3. Provide brief reasoning for each score

Return your evaluation as JSON matching this schema:
```json
{schema}
```

Return ONLY the JSON object, no other text.
"""


def build_initial_prompt(rollout_ids: list[str]) -> str:
    """Build the initial prompt that introduces files to the judge."""
    files = "\n".join(
        f"- {JUDGE_FILES_DIR}/rollout_{i}.hjson (ID: {rid})" for i, rid in enumerate(rollout_ids)
    )
    return INITIAL_PROMPT.format(task_file=f"{JUDGE_FILES_DIR}/task.txt", rollout_files_list=files)


def build_metric_prompt(metric: Metric) -> str:
    """Build a prompt for evaluating a single metric."""
    return METRIC_PROMPT.format(
        metric=str(metric),
        schema=json.dumps(MetricResult.model_json_schema(), indent=2),
    )


def extract_json(response: str) -> dict:
    """Extract JSON from the judge agent's response."""
    for pattern in [None, r"```(?:json)?\s*([\s\S]*?)\s*```", r"\{[\s\S]*\}"]:
        try:
            if pattern is None:
                return json.loads(response)
            if match := re.search(pattern, response):
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not extract JSON from response: {response[:500]}")


def prepare_files(rollouts: list[RolloutResult], task_prompt: str) -> dict[Path, str | bytes]:
    """Prepare files to upload to the sandbox for judging."""
    base = Path(JUDGE_FILES_DIR)
    files: dict[Path, str | bytes] = {base / "task.txt": task_prompt}
    files |= {
        base / f"rollout_{i}.hjson": hjson.dumps(r.model_dump(), indent=2)
        for i, r in enumerate(rollouts)
    }
    return files


def parse_metrics(metrics_arg: str) -> list[Metric]:
    """Parse comma-separated metric names into a list of Metric objects."""
    names = [name.strip().lower() for name in metrics_arg.split(",")]
    metrics = []
    for name in names:
        if name not in METRICS_BY_NAME:
            available = ", ".join(m.name for m in ALL_METRICS)
            raise ValueError(f"Unknown metric '{name}'. Available: {available}")
        metrics.append(METRICS_BY_NAME[name])
    return metrics


async def run_judge_conversation(
    rollouts: list[RolloutResult],
    task_prompt: str,
    console: Console,
    metrics: list[Metric],
) -> AggregatedJudgeResult:
    """Run the judge agent in a continuous conversation, evaluating each metric."""
    files = prepare_files(rollouts, task_prompt)
    rollout_ids = [r.rollout_id for r in rollouts]
    metric_results: list[MetricResult] = []
    metric_iter = iter(metrics)
    current_metric: Metric | None = None

    def on_response(result: RolloutResult) -> str | None:
        """Process response and return next metric prompt, or None when done."""
        nonlocal current_metric

        if current_metric is not None and result.final_message:
            try:
                json_data = extract_json(result.final_message)
                json_data.setdefault("metric_name", current_metric.name)
                metric_results.append(MetricResult.model_validate(json_data))
                console.print(f"[green]✓ {current_metric.name} evaluation complete[/green]")
            except (ValueError, json.JSONDecodeError) as e:
                console.print(f"[red]Failed to parse result: {e}[/red]")

        current_metric = next(metric_iter, None)
        if current_metric is None:
            return None
        console.print(f"\n[bold cyan]📊 Evaluating: {current_metric.name}[/bold cyan]")
        return build_metric_prompt(current_metric)

    config = RolloutConfig(max_turns=MAX_TURNS, files_to_upload=files)
    executor = CodexRolloutExecutor(
        config,
        on_message=lambda item: print_judge_message(console, item),
        on_response=on_response,
    )
    await executor.execute(build_initial_prompt(rollout_ids))
    return AggregatedJudgeResult(metric_results=metric_results)


def load_rollouts(rollouts_dir: Path) -> tuple[list[RolloutResult], str]:
    """Load rollouts and task prompt from a directory."""
    if not rollouts_dir.exists():
        raise FileNotFoundError(f"Rollouts directory not found: {rollouts_dir}")
    task_file = rollouts_dir / "task.txt"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    rollouts = [
        RolloutResult.model_validate(hjson.load(f.open(encoding="utf-8")))
        for f in sorted(rollouts_dir.glob("rollout_*.hjson"))
    ]
    if not rollouts:
        raise ValueError(f"No rollout files found in {rollouts_dir}")
    return rollouts, task_file.read_text().strip()


def display_results(console: Console, result: AggregatedJudgeResult) -> None:
    """Display judge results with a table (rollouts as rows, metrics as columns)."""
    console.print("\n[bold green] Judge Results[/bold green]\n")

    rollout_ids = list(result.total_scores.keys())
    metric_names = [mr.metric_name for mr in result.metric_results]

    # Build table: rollouts as rows, metrics as columns
    table = Table(title="Metric Scores", show_header=True, header_style="bold cyan")
    table.add_column("Rollout", style="bold")
    for metric_name in metric_names:
        table.add_column(metric_name, justify="center")

    # Build score lookup: {rollout_id: {metric_name: score}}
    scores_by_rollout: dict[str, dict[str, int]] = {rid: {} for rid in rollout_ids}
    for mr in result.metric_results:
        for s in mr.scores:
            scores_by_rollout[s.rollout_id][mr.metric_name] = s.score

    for rid in rollout_ids:
        row_scores = [str(scores_by_rollout[rid].get(m, 0)) for m in metric_names]
        table.add_row(rid, *row_scores)

    # Stats rows (avg and std)
    table.add_section()
    metric_vals = {
        m: np.array([scores_by_rollout[rid].get(m, 0) for rid in rollout_ids]) for m in metric_names
    }

    avg_row = [f"{metric_vals[m].mean():.1f}" for m in metric_names]
    std_row = [f"{metric_vals[m].std():.1f}" for m in metric_names]
    table.add_row("[dim]avg[/dim]", *avg_row)
    table.add_row("[dim]std[/dim]", *std_row)

    console.print(table)

    console.print("\n[bold]Rankings:[/bold]")
    for i, rid in enumerate(result.rankings, 1):
        console.print(f"  {i}. {rid} ({result.total_scores[rid]} points)")


async def main() -> None:
    args = JudgeSettings(_cli_parse_args=True)  # type: ignore[call-arg]
    console = Console()

    # Parse metrics
    metrics = parse_metrics(args.metrics)
    metric_names = ", ".join(m.name for m in metrics)
    console.print(f"[bold cyan]Metrics: {metric_names}[/bold cyan]")

    console.print(f"[bold cyan]Loading rollouts from {args.rollouts_dir}...[/bold cyan]")
    rollouts, task_prompt = load_rollouts(args.rollouts_dir)
    console.print(f"[green]Loaded {len(rollouts)} rollouts[/green]")

    for r in rollouts:
        if r.final_message and len(r.final_message) > FINAL_MESSAGE_TRUNCATE_LENGTH:
            final_msg = r.final_message[:FINAL_MESSAGE_TRUNCATE_LENGTH] + "..."
        else:
            final_msg = r.final_message
        panel_content = (
            f"Turns: {r.turn_count}\n"
            f"Git diff size: {len(r.git_diff)} chars\n"
            f"Final message: {final_msg}"
        )
        console.print(Panel(panel_content, title=f"{r.rollout_id}", border_style="blue"))

    console.print("\n[bold cyan]Running judge agent...[/bold cyan]\n")
    result = await run_judge_conversation(rollouts, task_prompt, console, metrics)

    display_results(console, result)
    console.print("\n[bold]Structured Output:[/bold]")
    console.print(Syntax(result.model_dump_json(indent=2), "json", theme="monokai"))

    result_file = args.rollouts_dir / "judge_result.hjson"
    with result_file.open("w", encoding="utf-8") as f:
        hjson.dump(result.model_dump(), f, indent=2)
    console.print(f"\n[green]✓ Results saved to {result_file}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
