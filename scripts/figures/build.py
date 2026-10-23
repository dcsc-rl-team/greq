"""Build a reusable D3 learning-curve page from explicit CSV inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import tyro


@dataclass(frozen=True)
class Config:
    """CSV inputs and an unused figure directory under runs/."""

    csvs: tuple[Path, ...]  # CSV files to validate and include.
    output: Path = Path("runs/figures")  # Unused figure directory under runs/.
    kind: Literal["learning", "runtime"] = "learning"
    labels: tuple[str, ...] = ()  # One per CSV; repeated labels group replicates.
    metrics: tuple[
        str, ...
    ] = ()  # Numeric metric columns; default all non-axis columns.
    step_column: str = "step"  # Sample-count column.
    time_column: str | None = "train_seconds"  # Time column; None makes a single panel.
    include_failed: bool = False  # Permit explicitly labeled partial or failed runs.
    title: str | None = None  # Override the title inferred from run metadata.
    time_label: str = "run time (s)"  # Concise axis title.
    time_description: str | None = (
        None  # Timing scope retained in metadata and exports.
    )


@dataclass(frozen=True)
class CsvRun:
    """One snapshotted CSV and its available run metadata."""

    file: str
    label: str
    source: str
    sha256: str
    status: str
    seed: int | None
    environment: str | None = None  # Runtime facet row label.
    gamma: float | None = None  # Runtime facet column.


@dataclass(frozen=True)
class Metric:
    """Metric column and its on-plot label."""

    key: str
    label: str


@dataclass(frozen=True)
class Manifest:
    """Data-only interface consumed by the reusable browser renderer."""

    kind: str
    step_column: str
    time_column: str | None
    metrics: tuple[Metric, ...]
    runs: tuple[CsvRun, ...]
    title: str | None = None
    time_label: str = "run time (s)"
    time_description: str | None = None
    band: str = "empirical percentile band"
    percentile_bounds: tuple[int, int] = (25, 75)
    primary_algorithm: str = "GReQ"
    time_alignment: str = "linear interpolation on shared support; no extrapolation"


_LABELS = {"greq": "GReQ", "ql": "QL", "zap": "Zap", "zap-batched": "Zap-batched"}
_METRICS = {
    "bellman_linf": "Bellman residual",
    "value_linf": "Value error",
    "policy_return": "Policy return",
    "policy_gap": "Policy gap",
}


def build(config: Config) -> Path:
    """Validate inputs and create a self-contained page at an unused path."""
    root = Path("runs").resolve()
    output = config.output.resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("figure output must be a new subdirectory of runs/")
    if not config.csvs or len(set(path.resolve() for path in config.csvs)) != len(
        config.csvs
    ):
        raise ValueError("provide distinct CSV inputs")
    if config.labels and len(config.labels) != len(config.csvs):
        raise ValueError("labels must have one entry per CSV")
    axes = [config.step_column]
    if config.time_column is not None:
        axes.append(config.time_column)
    if config.kind not in ("learning", "runtime"):
        raise ValueError("kind must be learning or runtime")
    if config.kind == "runtime" and config.time_column is None:
        raise ValueError("runtime figures require a time column")
    metrics = config.metrics
    runs = []
    runtime_cells = set()
    runtime_seeds = {}
    runtime_settings = {}
    runtime_protocols = set()
    runtime_environments = {}
    environments = []
    group_settings = {}
    group_seeds = set()
    for index, path in enumerate(config.csvs):
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            if len(set(headers)) != len(headers):
                raise ValueError(f"duplicate CSV columns: {path}")
            if not metrics:
                metrics = tuple(name for name in headers if name not in axes)
            required = [*axes, *metrics]
            if not metrics or any(name not in headers for name in required):
                raise ValueError(f"missing axis or metric columns: {path}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"empty CSV: {path}")
        previous_step, previous_time = -1.0, -1.0
        for row in rows:
            values = [float(row[name]) for name in required]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite CSV value: {path}")
            step = float(row[config.step_column])
            if step < 0 or not step.is_integer() or step <= previous_step:
                raise ValueError(
                    f"sample counts must be increasing nonnegative integers: {path}"
                )
            previous_step = step
            if config.time_column is not None:
                elapsed = float(row[config.time_column])
                if elapsed < 0 or elapsed < previous_time:
                    raise ValueError(
                        f"time must be nonnegative and nondecreasing: {path}"
                    )
                previous_time = elapsed
        metadata_path = path.parent / "config.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        status_path = path.parent / "status.json"
        status_data = (
            json.loads(status_path.read_text()) if status_path.exists() else {}
        )
        status = status_data.get("status", "unknown")
        if status not in ("completed", "unknown") and not config.include_failed:
            raise ValueError(
                f"run is {status}: {path}; use --include-failed explicitly"
            )
        label = (
            config.labels[index]
            if config.labels
            else _LABELS.get(metadata.get("method"), path.parent.name)
        )
        if not label.strip():
            raise ValueError("labels must not be empty")
        seed = metadata.get("seed")
        environment, gamma = None, None
        if config.kind == "runtime":
            execution = metadata.get("execution", {})
            if (
                execution.get("mode") != "scalar-jit-staged"
                or execution.get("batch_size") != 1
            ):
                raise ValueError(
                    "runtime figures require single-run scalar JIT measurements"
                )
            budget = metadata.get("n_steps")
            if (
                status != "completed"
                or status_data.get("completed_steps") != budget
                or float(rows[-1][config.step_column]) != budget
            ):
                raise ValueError("runtime figures require completed fixed-budget runs")
            if float(rows[-1][config.time_column]) <= 0:
                raise ValueError("runtime must be positive for a logarithmic axis")
            if metadata.get("method") not in _LABELS or seed is None:
                raise ValueError("runtime figures require method and seed metadata")
            gamma = metadata["gamma"]
            if not isinstance(gamma, (int, float)) or not 0 < gamma < 1:
                raise ValueError("invalid runtime discount")
            environment = (
                "Graph (6×6 padded)"
                if metadata.get("environment", "garnet") == "graph"
                else f"Garnet ({metadata['n_states']}×{metadata['n_actions']})"
            )
            identity = tuple(
                metadata.get(key)
                for key in ("n_states", "n_actions", "branching", "env_seed")
            )
            if (
                environment in runtime_environments
                and runtime_environments[environment] != identity
            ):
                raise ValueError("runtime facet mixes different MDP instances")
            runtime_environments[environment] = identity
            runtime_protocols.add((budget, metadata.get("eval_every")))
            cell = (environment, gamma, label)
            seeds = runtime_seeds.setdefault(cell, set())
            if seed in seeds:
                raise ValueError("runtime cells require distinct seeds")
            seeds.add(seed)
            settings = {
                key: value
                for key, value in metadata.items()
                if key not in ("seed", "output", "execution")
            }
            if cell in runtime_settings and runtime_settings[cell] != settings:
                raise ValueError(
                    "runtime replicates require identical algorithm settings"
                )
            runtime_settings[cell] = settings
            runtime_cells.add(cell)
        if metadata and config.kind == "learning":
            environments.append(
                tuple(
                    metadata.get(key)
                    for key in (
                        "gamma",
                        "n_states",
                        "n_actions",
                        "branching",
                        "env_seed",
                    )
                )
                + (metadata.get("environment", "garnet"),)
            )
            settings = {
                key: value
                for key, value in metadata.items()
                if key not in ("seed", "output")
            }
            if label in group_settings and group_settings[label] != settings:
                raise ValueError(f"different algorithm settings share label {label!r}")
            group_settings[label] = settings
            if (label, seed) in group_seeds:
                raise ValueError(f"duplicate seed for label {label!r}")
            group_seeds.add((label, seed))
        runs.append(
            CsvRun(
                file=f"data/run-{index:03d}.csv",
                label=label,
                source=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                status=status,
                seed=seed,
                environment=environment,
                gamma=gamma,
            )
        )
    if config.kind == "runtime":
        if (
            len(runtime_protocols) != 1
            or len({frozenset(seeds) for seeds in runtime_seeds.values()}) != 1
        ):
            raise ValueError(
                "runtime comparisons require matched seed sets, budget and block size"
            )
        expected = {
            (env, gamma, run.label)
            for env in runtime_environments
            for gamma in {r.gamma for r in runs}
            for run in runs
        }
        if runtime_cells != expected:
            raise ValueError(
                "runtime comparison has missing environment/discount/method cells"
            )
    if len(set(environments)) > 1:
        raise ValueError("CSV environments or discounts differ; use separate pages")
    if len(set(run.label for run in runs)) > 6:
        raise ValueError("at most six groups per page; split larger comparisons")
    title = config.title
    if (
        title is None
        and len(environments) == len(runs)
        and all(value is not None for value in environments[0][:4])
    ):
        name = "Graph (6×6 padded)" if environments[0][-1] == "graph" else "Garnet"
        title = f"{name} — γ = {environments[0][0]:g}"
    manifest = Manifest(
        kind=config.kind,
        step_column=config.step_column,
        time_column=config.time_column,
        metrics=tuple(Metric(key=key, label=_METRICS.get(key, key)) for key in metrics),
        runs=tuple(runs),
        title=title,
        time_label=config.time_label,
        time_description=config.time_description,
    )
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("assets")
    shutil.copytree(source / "viz", output / "viz")
    for name in ("index.html", "figures.mjs"):
        shutil.copyfile(source / name, output / name)
    (output / "data").mkdir()
    for path, run in zip(config.csvs, runs, strict=True):
        shutil.copyfile(path, output / run.file)
    (output / "data/manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, allow_nan=False) + "\n"
    )
    print(f"node {Path(__file__).with_name('serve.mjs')} 8000 {output}")
    return output


def main() -> None:
    """Build a figure page from tyro command-line arguments."""
    build(tyro.cli(Config))


if __name__ == "__main__":
    main()
