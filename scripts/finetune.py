"""Flat-vmapped tuning, explicit CSV records, and validated result caches."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from greq import checkpoint
from greq.evaluation import Evaluation
from scripts.run import Config as RunConfig
from scripts.run import build


@dataclass(frozen=True)
class Dependency:
    """Installed version and package source URL, including Git revision."""

    name: str
    version: str
    direct_url: str | None


@dataclass(frozen=True)
class Source:
    """Relative source path and SHA-256 of the archived bytes."""

    path: str
    sha256: str


@dataclass(frozen=True)
class Provenance:
    """Runtime and source identity for one executed configuration group."""

    python: str
    platform: str
    cpu: str
    backend: str
    devices: tuple[str, ...]
    jax_x64: bool
    prng_impl: str
    dependencies: tuple[Dependency, ...]
    source: tuple[Source, ...]


def write_json(path: Path, value: object) -> None:
    """Publish complete JSON atomically, rejecting nonfinite numerical values."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cpu_description() -> str:
    """Read the processor model from the OS, or record it as unavailable."""
    try:
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                key, _, value = line.partition(":")
                if key.strip().lower() in ("model name", "hardware", "model"):
                    return value.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return f"unavailable ({platform.machine()})"


def archive_provenance(directory: Path) -> None:
    """Archive project sources and the dependency lock from the working directory."""
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "pyproject.toml",
        root / "uv.lock",
        root / ".python-version",
        *sorted((root / "src/greq").rglob("*.py")),
        *sorted((root / "scripts").rglob("*.py")),
    ]
    sources = []
    for path in files:
        relative = path.relative_to(root)
        destination = directory / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources.append(
            Source(
                path=str(relative), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        )
    dependencies = []
    for name in ("greq", "jax", "jaxlib", "jaxtor", "jaxdp", "chex", "numpy", "tyro"):
        distribution = importlib.metadata.distribution(name)
        dependencies.append(
            Dependency(
                name=name,
                version=distribution.version,
                direct_url=distribution.read_text("direct_url.json"),
            )
        )
    provenance = Provenance(
        python=platform.python_version(),
        platform=platform.platform(),
        cpu=cpu_description(),
        backend=jax.default_backend(),
        devices=tuple(str(device) for device in jax.devices()),
        jax_x64=bool(jax.config.read("jax_enable_x64")),
        prng_impl="threefry2x32",
        dependencies=tuple(dependencies),
        source=tuple(sources),
    )
    (directory / "provenance.json").write_text(
        json.dumps(asdict(provenance), indent=2, allow_nan=False) + "\n"
    )


@dataclass(frozen=True)
class Row:
    """One observation with explicit environment, setting, and trajectory identity."""

    environment: str
    method: str
    execution: str
    n_states: int
    n_actions: int
    branching: int
    env_seed: int
    n_steps: int
    eval_every: int
    gamma: float
    eta: float
    rho: float | None
    k0: float | None
    batch_size: int | None
    seed: int
    step: int
    status: str
    error: str
    train_seconds: float | None
    bellman_linf: float | None
    value_linf: float | None
    policy_return: float | None
    policy_gap: float | None
    zeta: float | None = None  # NeSA only; absent in legacy CSV schemas.

    @property
    def cell(self) -> tuple[str, str, float]:
        """Identify the environment, algorithm, and discount being compared."""
        return self.environment, self.method, self.gamma

    @property
    def setting(self) -> tuple[float, float, float, int, float]:
        """Return a lexicographic key including NeSA's momentum multiplier."""
        return (
            self.eta,
            self.rho or 0.0,
            self.k0 or 0.0,
            self.batch_size or 0,
            self.zeta or 0.0,
        )

    def __post_init__(self) -> None:
        """Reject malformed identities and incomplete successful observations."""
        if self.environment not in ("garnet", "graph") or self.method not in (
            "greq",
            "ql",
            "nesa",
            "zap",
            "zap-batched",
        ):
            raise ValueError("unknown environment or algorithm")
        if self.status not in ("completed", "failed"):
            raise ValueError("invalid observation status")
        if not 0 < self.gamma < 1 or self.eta <= 0:
            raise ValueError("invalid discount or learning exponent")
        if self.method in ("ql", "nesa"):
            if self.k0 is None or self.k0 <= 0 or self.rho is not None:
                raise ValueError("QL and NeSA require k0 and no rho")
        elif (
            self.rho is None
            or not 0.5 < self.rho < self.eta <= 1
            or self.k0 is not None
        ):
            raise ValueError("matrix methods require 1/2 < rho < eta <= 1")
        if self.method == "nesa":
            if self.zeta is None or self.zeta <= 0:
                raise ValueError("NeSA requires a positive finite zeta")
        elif self.zeta is not None:
            raise ValueError("only NeSA accepts zeta")
        if (self.method == "zap-batched") != (self.batch_size is not None):
            raise ValueError("only batched Zap requires a batch size")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if (
            min(self.n_states, self.n_actions, self.n_steps) < 1
            or not 0 <= self.env_seed < 2**32
        ):
            raise ValueError("invalid environment or sample budget")
        if self.status == "failed" and not self.error:
            raise ValueError("failed observations require a reason")
        if not 0 <= self.seed < 2**32 or not 0 <= self.step <= self.n_steps:
            raise ValueError("invalid seed or sample index")
        if (
            self.eval_every < 1
            or self.n_steps % self.eval_every
            or self.step % self.eval_every
        ):
            raise ValueError("invalid observation interval")
        metrics = (
            self.bellman_linf,
            self.value_linf,
            self.policy_return,
            self.policy_gap,
        )
        if self.status == "completed" and (
            self.error or any(v is None for v in metrics)
        ):
            raise ValueError(
                "completed observations require finite metrics and no error"
            )
        if any(
            value is not None and value < 0
            for value in (self.bellman_linf, self.value_linf)
        ):
            raise ValueError("error norms cannot be negative")
        for value in asdict(self).values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("CSV observations must not contain nonfinite numbers")
        if self.execution not in ("scalar", "vmap"):
            raise ValueError("execution must be scalar or vmap")
        if self.train_seconds is not None and (
            self.train_seconds <= 0 or self.execution != "scalar"
        ):
            raise ValueError("training times require positive scalar measurements")


def _optional(value: str) -> float | None:
    """Parse a numeric CSV field, mapping blanks to None."""
    return float(value) if value else None


def read_rows(path: Path) -> tuple[Row, ...]:
    """Read the common schema and reject duplicate observation identities."""
    result = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        names = [f.name for f in fields(Row)]
        if reader.fieldnames not in (names, names[:-1]):
            raise ValueError(f"unsupported CSV schema: {path}")
        for r in reader:
            result.append(
                Row(
                    environment=r["environment"],
                    method=r["method"],
                    execution=r["execution"],
                    n_states=int(r["n_states"]),
                    n_actions=int(r["n_actions"]),
                    branching=int(r["branching"]),
                    env_seed=int(r["env_seed"]),
                    n_steps=int(r["n_steps"]),
                    eval_every=int(r["eval_every"]),
                    gamma=float(r["gamma"]),
                    eta=float(r["eta"]),
                    rho=_optional(r["rho"]),
                    k0=_optional(r["k0"]),
                    batch_size=int(r["batch_size"]) if r["batch_size"] else None,
                    seed=int(r["seed"]),
                    step=int(r["step"]),
                    status=r["status"],
                    error=r["error"],
                    train_seconds=_optional(r["train_seconds"]),
                    bellman_linf=_optional(r["bellman_linf"]),
                    value_linf=_optional(r["value_linf"]),
                    policy_return=_optional(r["policy_return"]),
                    policy_gap=_optional(r["policy_gap"]),
                    zeta=_optional(r.get("zeta", "")),
                )
            )
    identities = {(r.cell, r.setting, r.seed, r.step) for r in result}
    if not result or len(identities) != len(result):
        raise ValueError("CSV must contain distinct, nonempty observations")
    return tuple(result)


def normalize(value: object) -> object:
    """Normalize host requests to their portable JSON representation."""
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def write_rows(path: Path, rows: tuple[Row, ...], request: object) -> Path:
    """Publish a CSV and request/hash sidecar at unused paths."""
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(Row)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    read_rows(temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.rename(path)
    write_json(sidecar, {"request": normalize(request), "sha256": digest})
    return path


def read_cached(path: Path, request: object) -> tuple[Row, ...]:
    """Reuse only complete, unmodified CSVs produced for the same request."""
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError(f"incomplete CSV cache: {path}; use a new output directory")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("request") != normalize(request):
        raise ValueError(
            f"cached CSV belongs to a different protocol: {path}; "
            "use a new output directory"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
        raise ValueError(f"cached CSV checksum mismatch: {path}")
    return read_rows(path)


def validate_table(
    table: np.ndarray,
    gammas: np.ndarray,
    method: str = "ql",
) -> None:
    """Validate flat rows; NeSA appends zeta to seed, gamma, eta, k0."""
    if method not in ("ql", "nesa", "greq", "zap", "zap-batched"):
        raise ValueError("unknown method")
    width = 5 if method == "nesa" else 4
    if table.ndim != 2 or table.shape[1] != width or len(table) == 0:
        raise ValueError(f"expected a nonempty H×{width} parameter table")
    if table.dtype != np.float64 or not np.isfinite(table).all():
        raise ValueError(
            "require finite float64 rows to preserve uint32 seed precision"
        )
    seeds, discounts, etas, other = table[:, :4].T
    if np.any((seeds < 0) | (seeds > 2**32 - 1) | (seeds != np.floor(seeds))):
        raise ValueError("seeds must be uint32 integers")
    if np.any((discounts <= 0) | (discounts >= 1)):
        raise ValueError("discounts must lie in (0, 1)")
    if not np.isin(discounts, gammas).all():
        raise ValueError("every discount needs a certified reference")
    if method in ("ql", "nesa"):
        if np.any(etas <= 0) or np.any(other <= 0):
            raise ValueError("eta and k0 must be positive")
        if method == "nesa" and np.any(table[:, 4] <= 0):
            raise ValueError("zeta must be positive")
    elif np.any((other <= 0.5) | (other >= etas) | (etas > 1)):
        raise ValueError("matrix-gain exponents must satisfy 1/2 < rho < eta <= 1")
    if len(np.unique(table, axis=0)) != len(table):
        raise ValueError("duplicate run configurations")


@dataclass(frozen=True)
class Config:
    """Comparison protocol with disjoint tuning and evaluation seed sets."""

    output: Path = Path("runs/results")
    environments: tuple[Literal["garnet", "graph"], ...] = ("garnet", "graph")
    methods: tuple[Literal["greq", "ql", "nesa", "zap", "zap-batched"], ...] = (
        "greq",
        "ql",
        "zap",
        "zap-batched",
    )
    gammas: tuple[float, ...] = (0.95, 0.99, 0.995, 0.999)
    selection_seeds: tuple[int, ...] = (0, 1, 2)
    evaluation_seeds: tuple[int, ...] = tuple(range(3, 14))
    etas: tuple[float, ...] = (1, 1, 1, 1, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7)
    rhos: tuple[float, ...] = (0.9, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.6, 0.6)
    ql_etas: tuple[float, ...] = (0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
    ql_k0s: tuple[float, ...] = (1, 10, 100, 1000)
    # 24 NeSA settings, including the source point eta=k0=zeta=1.
    nesa_etas: tuple[float, ...] = (1.0, 0.8, 0.6)
    nesa_k0s: tuple[float, ...] = (1.0, 10.0)
    nesa_zetas: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    n_states: int = 50  # Garnet states; Graph retains its fixed representation.
    n_actions: int = 5
    branching: int = 5
    env_seed: int = 0
    n_steps: int = 10_000
    eval_every: int = 100

    def __post_init__(self) -> None:
        """Validate the complete protocol before producing any artifacts."""
        for values in (
            self.environments,
            self.methods,
            self.gammas,
            self.selection_seeds,
            self.evaluation_seeds,
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError("protocol identities must be nonempty and distinct")
        if len(self.evaluation_seeds) != 11 or set(self.selection_seeds) & set(
            self.evaluation_seeds
        ):
            raise ValueError(
                "evaluation requires 11 distinct seeds disjoint from selection"
            )
        if len(self.etas) != len(self.rhos):
            raise ValueError("etas and rhos must be paired")
        for environment in self.environments:
            for method in self.methods:
                cfg = self.config(environment, method, "validation")
                table = self.table(method)
                validate_table(table, np.asarray(self.gammas), method)
                for seed in self.evaluation_seeds:
                    replace(cfg, seed=seed)

    def config(
        self,
        environment: Literal["garnet", "graph"],
        method: Literal["greq", "ql", "nesa", "zap", "zap-batched"],
        phase: str,
    ) -> RunConfig:
        """Configure one static environment/algorithm group."""
        return RunConfig(
            environment=environment,
            method=method,
            output=self.output / "_artifacts" / phase / environment / method,
            n_states=self.n_states,
            n_actions=self.n_actions,
            branching=self.branching,
            env_seed=self.env_seed,
            n_steps=self.n_steps,
            eval_every=self.eval_every,
            batch_size=(36 if environment == "graph" else 770)
            if method == "zap-batched"
            else None,
        )

    def table(self, method: str) -> np.ndarray:
        """Construct flat scalar configurations; only NeSA has a fifth column."""
        if method == "nesa":
            settings = tuple(
                itertools.product(self.nesa_etas, self.nesa_k0s, self.nesa_zetas)
            )
        elif method == "ql":
            settings = tuple(itertools.product(self.ql_etas, self.ql_k0s))
        else:
            settings = tuple(zip(self.etas, self.rhos, strict=True))
        return np.asarray(
            [
                (seed, gamma, *setting)
                for gamma in self.gammas
                for setting in settings
                for seed in self.selection_seeds
            ],
            dtype=np.float64,
        ).reshape(-1, 5 if method == "nesa" else 4)


def request(study: Config, phase: str) -> dict:
    """Identify a cache independently of output location and plotting changes."""
    values = {
        f.name: getattr(study, f.name)
        for f in fields(Config)
        if f.name not in ("output", "evaluation_seeds")
    }
    if phase != "tuning":
        values["evaluation_seeds"] = study.evaluation_seeds
    source = Path(__file__).resolve().parents[1]
    files = [
        *sorted((source / "src/greq").glob("*.py")),
        *(source / "scripts" / name for name in ("run.py", "finetune.py", "report.py")),
        source / "uv.lock",
    ]
    return {
        "phase": phase,
        "protocol": values,
        "execution": {
            "dtype": "float64",
            "prng": "threefry2x32",
            "backend": jax.default_backend(),
            "devices": [d.device_kind for d in jax.devices()],
            "platform": platform.platform(),
            "cpu": cpu_description(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "jaxtor", "jaxdp", "chex", "numpy")
            },
        },
        "selection": "median final value_linf; all selection seeds finite; "
        "ties by eta,rho,k0,batch_size,zeta",
        "kernels": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }


def observation(
    cfg: RunConfig,
    row: np.ndarray,
    step: int,
    metrics: Evaluation.Metrics | None,
    error: str = "",
    seconds: float | None = None,
    *,
    execution: Literal["scalar", "vmap"] = "vmap",
) -> Row:
    """Build an observation row with optional measured scalar runtime."""
    return Row(
        environment=cfg.environment,
        method=cfg.method,
        execution=execution,
        n_states=cfg.n_states,
        n_actions=cfg.n_actions,
        branching=cfg.branching,
        env_seed=cfg.env_seed,
        n_steps=cfg.n_steps,
        eval_every=cfg.eval_every,
        gamma=float(row[1]),
        eta=float(row[2]),
        rho=float(row[3]) if cfg.method not in ("ql", "nesa") else None,
        k0=float(row[3]) if cfg.method in ("ql", "nesa") else None,
        zeta=float(row[4]) if cfg.method == "nesa" else None,
        batch_size=(cfg.batch_size or cfg.n_states * cfg.n_actions)
        if cfg.method == "zap-batched"
        else None,
        seed=int(row[0]),
        step=step,
        status="failed" if error else "completed",
        error=error,
        train_seconds=seconds,
        bellman_linf=float(metrics.bellman_linf) if metrics is not None else None,
        value_linf=float(metrics.value_linf) if metrics is not None else None,
        policy_return=float(metrics.policy_return) if metrics is not None else None,
        policy_gap=float(metrics.policy_gap) if metrics is not None else None,
    )


def execute(
    cfg: RunConfig, table: np.ndarray, *, mapped: bool, history: bool
) -> tuple[Row, ...]:
    """Execute the same run function, scalar or vmapped, retaining failed rows."""
    jax.config.update("jax_enable_x64", True)
    gammas = np.unique(table[:, 1])
    validate_table(table, gammas, cfg.method)
    if not mapped and len(table) != 1:
        raise ValueError("scalar execution requires one configuration")
    cfg.output.mkdir(parents=True, exist_ok=False)
    archive_provenance(cfg.output)
    np.save(cfg.output / "parameters.npy", table, allow_pickle=False)
    write_json(
        cfg.output / "config.json",
        {
            **asdict(cfg),
            "output": str(cfg.output),
            "execution": "vmap" if mapped else "scalar",
            "status": "running",
        },
    )
    try:
        runner = build(cfg, jnp.asarray(gammas))
        inputs = jnp.asarray(table if mapped else table[0])
        compiled = (
            jax.jit(jax.vmap(runner.run) if mapped else runner.run)
            .lower(inputs)
            .compile()
        )
        started = time.perf_counter()
        result = jax.block_until_ready(compiled(inputs))
        elapsed = time.perf_counter() - started
        records = []
        for index, row in enumerate(table):
            individual = (
                jax.tree.map(lambda value, index=index: value[index], result)
                if mapped
                else result
            )
            finite = np.asarray(individual.finite)
            failures = np.flatnonzero(~finite)
            error = (
                "nonfinite learner or metrics at sample "
                f"{int(failures[0]) * cfg.eval_every}"
                if len(failures)
                else ""
            )
            indices = range(len(finite)) if history else (len(finite) - 1,)
            for sample in indices:
                metrics = (
                    jax.tree.map(
                        lambda value, sample=sample: value[sample], individual.metrics
                    )
                    if finite[sample]
                    else None
                )
                records.append(
                    observation(
                        cfg,
                        row,
                        sample * cfg.eval_every,
                        metrics,
                        error,
                        execution="vmap" if mapped else "scalar",
                    )
                )
        write_rows(
            cfg.output / ("evaluation.csv" if history else "tuning.csv"),
            tuple(records),
            {
                "config": asdict(cfg),
                "parameters": table.tolist(),
                "execution": "vmap" if mapped else "scalar",
            },
        )
        for index in range(len(table)):
            individual = (
                jax.tree.map(lambda value, index=index: value[index], result)
                if mapped
                else result
            )
            checkpoint.save(
                cfg.output / f"checkpoint-{index:04d}.npz", individual.state
            )
        write_json(
            cfg.output / "status.json",
            {
                "status": "completed",
                "runs": len(table),
                "failed_runs": sum(
                    r.status == "failed" for r in records if r.step == cfg.n_steps
                ),
                "batch_compute_seconds": elapsed,
                "timing_scope": "whole computation including observations; "
                "not individual training latency",
            },
        )
        return tuple(records)
    except Exception as error:
        write_json(
            cfg.output / "status.json",
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def finetune(config: RunConfig, table: np.ndarray) -> tuple[Row, ...]:
    """Vmap the scalar run over any validated flat table and report final scores."""
    return execute(config, table, mapped=True, history=False)


def run(study: Config) -> Path:
    """Produce tuning.csv, or validate its existing cache."""
    path = study.output / "tuning.csv"
    specification = request(study, "tuning")
    if path.exists():
        validate_tuning(study, read_cached(path, specification))
        return path
    rows = tuple(
        row
        for environment in study.environments
        for method in study.methods
        for row in finetune(
            study.config(environment, method, "tuning"), study.table(method)
        )
    )
    validate_tuning(study, rows)
    return write_rows(path, rows, specification)


def validate_tuning(study: Config, rows: tuple[Row, ...]) -> None:
    """Require observations for every requested setting and selection seed."""
    expected = {
        (environment, method, *row)
        for environment in study.environments
        for method in study.methods
        for row in study.table(method)
    }
    actual = {
        (
            r.environment,
            r.method,
            r.seed,
            r.gamma,
            r.eta,
            r.k0 if r.method in ("ql", "nesa") else r.rho,
            *((r.zeta,) if r.method == "nesa" else ()),
        )
        for r in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise ValueError("tuning CSV does not contain the complete requested grid")
    configs: dict[tuple[str, str], RunConfig] = {
        (e, m): study.config(e, m, "validation")
        for e in study.environments
        for m in study.methods
    }
    for row in rows:
        config = configs[row.environment, row.method]
        if (
            any(
                getattr(row, key) != getattr(config, key)
                for key in (
                    "n_states",
                    "n_actions",
                    "branching",
                    "env_seed",
                    "n_steps",
                    "eval_every",
                    "batch_size",
                )
            )
            or row.step != config.n_steps
        ):
            raise ValueError(
                "tuning row metadata does not match the requested protocol"
            )


def main() -> None:
    """Create or validate tuning.csv for the configured selection seeds."""
    print(run(tyro.cli(Config)))


if __name__ == "__main__":
    main()
