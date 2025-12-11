"""Minimal Codex runner via ACP in Daytona sandboxes."""

import asyncio

import yaml
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.traceback import install

from acp_client import ConversationItem
from rollout_executor import CodexRolloutExecutor, RolloutConfig, RolloutResult

install(show_locals=True)
load_dotenv()


def _truncate_middle(text: str, max_lines: int = 10) -> str:
    """Truncate text in the middle, keeping first and last lines."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    keep_each_side = max_lines // 2
    truncated_count = len(lines) - max_lines
    top = lines[:keep_each_side]
    bottom = lines[-keep_each_side:]
    return "\n".join([*top, f"... ({truncated_count} lines truncated) ...", *bottom])


def _print_conversation_item(console: Console, item: ConversationItem) -> None:
    """Print a conversation item with Rich formatting."""
    role_styles = {
        "user": ("bold cyan", "👤 USER"),
        "agent": ("bold green", "🤖 ASSISTANT"),
        "thought": ("bold magenta", "💭 THOUGHT"),
        "tool": ("bold yellow", "🔧 TOOL"),
    }

    style, title = role_styles.get(item.type, ("bold white", item.type.upper()))

    if item.type == "tool":
        # Build tool content with tool_kind in the title
        if item.tool_kind:
            title = f"{title} ({item.tool_kind})"
        content_parts = []
        if item.tool_input:
            content_parts.append(Text("\n\n📥 INPUT:", style="bold"))
            yaml_str = yaml.dump(item.tool_input, default_flow_style=False, indent=2)
            yaml_str = _truncate_middle(yaml_str)
            content_parts.append(Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False))

        if item.tool_output:
            content_parts.append(Text("\n📤 OUTPUT:", style="bold"))
            if isinstance(item.tool_output, dict):
                yaml_str = yaml.dump(item.tool_output, default_flow_style=False, indent=2)
                yaml_str = _truncate_middle(yaml_str)
                content_parts.append(Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False))
            elif isinstance(item.tool_output, str):
                truncated = _truncate_middle(item.tool_output)
                content_parts.append(Text(truncated, style="dim"))

        panel = Panel(Group(*content_parts), title=title, border_style=style, expand=True)
    else:
        content = str(item.content) if item.content else ""
        panel = Panel(content, title=title, border_style=style, expand=True)

    console.print(panel)
    console.print()


async def main() -> None:
    console = Console()
    repo = "https://github.com/pallets/flask"

    console.print(f"Cloning {repo}...")
    console.print("Creating sandbox and running agent...\n")

    prompts = [
        "which model are you?",
        "what tools do you have?",
        "Make the example in the README.md more complex.",
    ]

    # Multi-turn conversation via on_response hook
    prompt_iter = iter(prompts[1:])  # Skip first, it's the initial prompt

    def next_prompt(_: RolloutResult) -> str | None:
        return next(prompt_iter, None)

    config = RolloutConfig(repo=repo)
    executor = CodexRolloutExecutor(
        config,
        on_message=lambda item: _print_conversation_item(console, item),
        on_response=next_prompt,
    )

    await executor.execute(prompts[0])


if __name__ == "__main__":
    asyncio.run(main())
