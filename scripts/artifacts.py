"""Host-side experiment records and source provenance."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import jax


@dataclass(frozen=True)
class Record:
    """One CSV row of exact diagnostics and synchronized training time."""

    step: int
    train_seconds: float
    bellman_linf: float
    value_linf: float
    policy_return: float
    policy_gap: float


@dataclass(frozen=True)
class Status:
    """Run lifecycle record retained for successful and failed runs."""

    status: Literal["running", "completed", "failed"]
    completed_steps: int = 0
    compile_seconds: float = 0.0
    warmup_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class Dependency:
    """Installed version and original package URL, including Git revision."""

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
    """Runtime and source identity sufficient to audit a pilot run."""

    schema_version: int
    python: str
    platform: str
    backend: str
    devices: tuple[str, ...]
    jax_x64: bool
    prng_impl: str
    dependencies: tuple[Dependency, ...]
    source: tuple[Source, ...]


def write_status(directory: Path, status: Status) -> None:
    """Atomically replace the run status, without hiding failures."""
    temporary = directory / "status.json.tmp"
    temporary.write_text(json.dumps(asdict(status), indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "status.json")


def append_record(directory: Path, record: Record) -> None:
    """Append a flushed CSV row, creating its header on first use."""
    path = directory / "metrics.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=[field.name for field in fields(record)]
        )
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(record))


def archive_provenance(directory: Path) -> None:
    """Archive executed project sources and the dependency lock without invoking Git."""
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "pyproject.toml",
        root / "uv.lock",
        root / ".python-version",
        *sorted((root / "src/greq").glob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
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
        schema_version=1,
        python=platform.python_version(),
        platform=platform.platform(),
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
