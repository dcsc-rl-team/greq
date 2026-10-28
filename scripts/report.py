"""Frozen selection and eleven-seed evaluation, optionally with scalar timing."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from greq import checkpoint
from scripts.finetune import Config as TuningConfig
from scripts.finetune import (
    Row,
    archive_provenance,
    execute,
    observation,
    read_cached,
    request,
    validate_table,
    write_json,
    write_rows,
)
from scripts.finetune import run as tune
from scripts.run import Config as RunConfig
from scripts.run import build


@dataclass(frozen=True)
class Config(TuningConfig):
    """Evaluation protocol with optional independent runtime measurements."""

    runtime: bool = False  # Produce runtime.csv from scalar measurements.


@dataclass(frozen=True)
class Winner:
    """Selected setting with every selection-seed observation and its score."""

    score: float
    runs: tuple[Row, ...]


def select(rows: tuple[Row, ...], seeds: tuple[int, ...]) -> tuple[Winner, ...]:
    """Select by median final error among settings with every seed successful."""
    groups = {}
    for row in rows:
        if row.step != row.n_steps:
            raise ValueError("selection requires final observations only")
        groups.setdefault((row.cell, row.setting), []).append(row)
    candidates = {}
    for (cell, _), members in groups.items():
        candidates.setdefault(cell, [])
        if len(members) != len(seeds) or {r.seed for r in members} != set(seeds):
            raise ValueError("tuning candidate has missing or unexpected seeds")
        values = [
            r.value_linf
            for r in members
            if r.value_linf is not None and r.status == "completed"
        ]
        if len(values) == len(seeds):
            candidates[cell].append(
                Winner(score=float(np.quantile(values, 0.5)), runs=tuple(members))
            )
    if not candidates or any(not values for values in candidates.values()):
        raise ValueError("no eligible setting for at least one comparison cell")
    return tuple(
        min(values, key=lambda w: (w.score, w.runs[0].setting))
        for _, values in sorted(candidates.items())
    )


def _evaluation_table(
    winners: tuple[Winner, ...], environment: str, method: str, seeds: tuple[int, ...]
) -> np.ndarray:
    """Expand fixed winning settings over the evaluation trajectory seeds."""
    return np.asarray(
        [
            (
                seed,
                r.gamma,
                r.eta,
                r.k0 if method in ("ql", "nesa") else r.rho,
                *((r.zeta,) if method == "nesa" else ()),
            )
            for winner in winners
            for r in winner.runs[:1]
            if r.environment == environment and r.method == method
            for seed in seeds
        ],
        dtype=np.float64,
    )


def benchmark(cfg: RunConfig, table: np.ndarray) -> tuple[Row, ...]:
    """Time each seed in one scalar sampling-and-learning call."""
    jax.config.update("jax_enable_x64", True)
    cfg = replace(cfg, eval_every=cfg.n_steps)
    validate_table(table, np.unique(table[:, 1]), cfg.method)
    cfg.output.mkdir(parents=True, exist_ok=False)
    archive_provenance(cfg.output)
    np.save(cfg.output / "parameters.npy", table, allow_pickle=False)
    specification = {
        "config": {**asdict(cfg), "output": str(cfg.output)},
        "parameters": table.tolist(),
        "execution": "scalar",
    }
    write_json(cfg.output / "config.json", specification)
    write_json(cfg.output / "status.json", {"status": "running", "finished_rows": 0})
    rows = []
    try:
        runner = build(cfg, jnp.asarray(np.unique(table[:, 1])))
        initial = jax.block_until_ready(runner.initialize(jnp.asarray(table[0])))
        compiled = jax.jit(runner.advance).lower(initial).compile()
        jax.block_until_ready(compiled(initial))
        for index, row in enumerate(table):
            state = jax.block_until_ready(runner.initialize(jnp.asarray(row)))
            started = time.perf_counter()
            state = jax.block_until_ready(compiled(state))
            seconds = time.perf_counter() - started
            metrics, finite = jax.block_until_ready(runner.observe(state))
            rows.append(
                observation(
                    cfg,
                    row,
                    cfg.n_steps,
                    metrics if bool(finite) else None,
                    "" if bool(finite) else "nonfinite learner or diagnostics",
                    seconds,
                    execution="scalar",
                )
            )
            write_json(
                cfg.output / "progress.json",
                {"finished_rows": len(rows), "rows": [asdict(r) for r in rows]},
            )
            checkpoint.save(cfg.output / f"checkpoint-{index:04d}.npz", state)
        write_rows(cfg.output / "runtime.csv", tuple(rows), specification)
    except Exception as error:
        write_json(
            cfg.output / "status.json",
            {
                "status": "failed",
                "finished_rows": len(rows),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    write_json(
        cfg.output / "status.json",
        {
            "status": "completed",
            "execution": "scalar",
            "runs": len(table),
            "failed_runs": sum(r.status == "failed" for r in rows),
            "training_calls_per_run": 1,
            "excluded": [
                "initialization",
                "compilation",
                "discarded warmup",
                "evaluation",
                "I/O",
            ],
        },
    )
    return tuple(rows)


def validate_evaluation(
    study: TuningConfig,
    winners: tuple[Winner, ...],
    rows: tuple[Row, ...],
    *,
    runtime: bool = False,
) -> None:
    """Require all frozen settings, seeds, and observations in a cached report."""
    settings = {winner.runs[0].cell: winner.runs[0].setting for winner in winners}
    steps = (
        (study.n_steps,) if runtime else range(0, study.n_steps + 1, study.eval_every)
    )
    expected = {
        (cell, seed, step)
        for cell in settings
        for seed in study.evaluation_seeds
        for step in steps
    }
    actual = {(row.cell, row.seed, row.step) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("evaluation CSV has missing or unexpected observations")
    configs: dict[tuple[str, str], RunConfig] = {
        (e, m): study.config(e, m, "validation")
        for e in study.environments
        for m in study.methods
    }
    for row in rows:
        config = configs[row.environment, row.method]
        if row.setting != settings[row.cell] or any(
            getattr(row, key) != getattr(config, key)
            for key in ("n_states", "n_actions", "branching", "env_seed", "n_steps")
        ):
            raise ValueError("evaluation settings differ from the frozen selection")
        if (
            row.execution != ("scalar" if runtime else "vmap")
            or row.eval_every != (study.n_steps if runtime else study.eval_every)
            or (not runtime and row.train_seconds is not None)
        ):
            raise ValueError("evaluation timing scope does not match its request")
        if runtime and row.train_seconds is None:
            raise ValueError("runtime observations require measured times")


def run(study: Config) -> Path:
    """Ensure tuning exists, freeze winners, and evaluate eleven seeds into one CSV."""
    tuning = tune(study)
    winners = select(
        read_cached(tuning, request(study, "tuning")), study.selection_seeds
    )
    expected = {
        (e, m, g)
        for e in study.environments
        for m in study.methods
        for g in study.gammas
    }
    if {w.runs[0].cell for w in winners} != expected:
        raise ValueError("tuning CSV has missing or unexpected comparison cells")
    specification = {
        **request(study, "evaluation"),
        "tuning_sha256": hashlib.sha256(tuning.read_bytes()).hexdigest(),
    }
    path = study.output / "evaluation.csv"
    if path.exists():
        validate_evaluation(study, winners, read_cached(path, specification))
    else:
        write_json(
            study.output / "selection.json",
            {
                "criterion": "median final value_linf",
                "winners": [asdict(w) for w in winners],
            },
        )
        rows = tuple(
            row
            for environment in study.environments
            for method in study.methods
            for row in execute(
                study.config(environment, method, "evaluation"),
                _evaluation_table(winners, environment, method, study.evaluation_seeds),
                mapped=True,
                history=True,
            )
        )
        validate_evaluation(study, winners, rows)
        write_rows(path, rows, specification)
    if study.runtime:
        runtime_path = study.output / "runtime.csv"
        runtime_request = {
            **specification,
            "phase": "runtime",
            "timing": "scalar single-call",
        }
        if runtime_path.exists():
            validate_evaluation(
                study, winners, read_cached(runtime_path, runtime_request), runtime=True
            )
        else:
            rows = tuple(
                row
                for environment in study.environments
                for method in study.methods
                for row in benchmark(
                    study.config(environment, method, "runtime"),
                    _evaluation_table(
                        winners, environment, method, study.evaluation_seeds
                    ),
                )
            )
            validate_evaluation(study, winners, rows, runtime=True)
            write_rows(runtime_path, rows, runtime_request)
    return path


def main() -> None:
    """Ensure tuning exists and print the evaluation CSV path."""
    print(run(tyro.cli(Config)))


if __name__ == "__main__":
    main()
