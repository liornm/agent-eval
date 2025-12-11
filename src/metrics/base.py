"""Base classes for metrics."""

from typing import Literal

from pydantic import BaseModel, Field

ArtifactType = Literal["conversation", "git_diff"]


class ComparisonExample(BaseModel):
    """A comparison example showing two outputs for the same task."""

    task: str = Field(description="The task prompt given to the agent")
    better: str = Field(description="Example output - a diff or conversation")
    worse: str = Field(description="Example output - a diff or conversation")
    explanation: str = Field(description="How the outputs differ for this metric")

    def __str__(self) -> str:
        """Format example for judge prompts."""
        return (
            f"Task: {self.task}\n\n"
            f"Better:\n```\n{self.better}\n```\n\n"
            f"Worse:\n```\n{self.worse}\n```\n\n"
            f"Why: {self.explanation}"
        )


class Metric(BaseModel):
    """A metric for evaluating agent rollouts."""

    name: str = Field(description="Short name of the metric")
    description: str = Field(description="What this metric measures")
    required_artifacts: list[ArtifactType] = Field(
        description="Artifact types needed to evaluate this metric"
    )
    examples: list[ComparisonExample] = Field(
        default_factory=list, description="Comparison examples showing different outputs"
    )

    def __str__(self) -> str:
        """Format metric for judge prompts."""
        if self.examples:
            examples = "\n".join(f"**Example {i}:**\n{ex}\n" for i, ex in enumerate(self.examples, 1))
        else:
            examples = "(No examples provided)"
        return f"## Metric: {self.name}\n{self.description}\n\n{examples}"
