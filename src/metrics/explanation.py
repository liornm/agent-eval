from metrics.base import ComparisonExample, Metric

EXPLANATION_STYLE = Metric(
    name="Explanation Style",
    description=(
        "Whether the agent communicates clearly: explains its approach before implementing, "
        "provides context for decisions, and structures responses in a logical, readable way."
    ),
    required_artifacts=["conversation", "git_diff"],
    examples=[
        ComparisonExample(
            task="fix typo in docs/appcontext.rst formatting issue in docs/design.rst",
            better="""\
Corrected the request context typo and fixed the "Lifecycle of the Context" \
heading in `docs/appcontext.rst`.

Repaired the RST role formatting for proxy names in `docs/design.rst` so the \
context locals render correctly.

Tests not run (docs-only change).""",
            worse="Fixed the typos you mentioned in the docs.",
            explanation=(
                "The better response summarizes each change made, explains why tests weren't "
                "run, and gives the user confidence the task was completed correctly. "
                "The worse response provides no detail, leaving the user unsure what was "
                "actually fixed or whether all issues were addressed."
            ),
        ),
    ],
)
