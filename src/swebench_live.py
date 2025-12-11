"""SWE-bench Live dataset utilities."""

import random
import secrets
from functools import cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset
from pydantic import BaseModel, Field, computed_field

DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_DIR / "swebench_live_all.parquet"
DOCKER_PREFIX = "starryzhang/sweb.eval.x86_64"


@cache
def _load_dataset() -> tuple[dict[str, Any], ...]:
    """Load dataset from local cache, downloading from HuggingFace if needed."""
    if not DATASET_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ds = load_dataset("SWE-bench-Live/SWE-bench-Live", split="all")
        pq.write_table(ds.data.table, DATASET_PATH, compression="zstd")
    return tuple(pq.read_table(DATASET_PATH).to_pylist())


class SWEBenchLiveTask(BaseModel):
    """A task from the SWE-bench Live dataset."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str = ""
    test_patch: str = ""
    test_cmds: list[str] = Field(default_factory=list)
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def docker_image(self) -> str:
        """Docker image on DockerHub for this task."""
        return f"{DOCKER_PREFIX}.{self.instance_id.lower().replace('__', '_1776_')}:latest"

    @computed_field
    @property
    def repo_url(self) -> str:
        """GitHub URL for the repository."""
        return f"https://github.com/{self.repo}"


def _to_task(row: dict) -> SWEBenchLiveTask:
    return SWEBenchLiveTask(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        patch=row.get("patch", ""),
        test_patch=row.get("test_patch", ""),
        test_cmds=row.get("test_cmds", []),
        fail_to_pass=row.get("FAIL_TO_PASS", []),
        pass_to_pass=row.get("PASS_TO_PASS", []),
    )


def sample_task(seed: int | None = None) -> SWEBenchLiveTask:
    """Sample a random task, optionally with a seed for reproducibility."""
    dataset = _load_dataset()
    if seed is not None:
        return _to_task(random.Random(seed).choice(dataset))
    return _to_task(secrets.choice(dataset))


def get_task(instance_id: str) -> SWEBenchLiveTask:
    """Get a task by instance ID."""
    for row in _load_dataset():
        if row["instance_id"] == instance_id:
            return _to_task(row)
    raise ValueError(f"Task not found: {instance_id}")
