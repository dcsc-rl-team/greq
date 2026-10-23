"""Package report CSVs and interactive figures into one offline HTML file."""

from __future__ import annotations

import csv
import hashlib
import json
import shlex
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from tempfile import TemporaryDirectory

import tyro

from scripts.figures.build import CsvRun, Manifest, Metric
from scripts.figures.gallery import Config as GalleryConfig
from scripts.figures.gallery import build as gallery
from scripts.finetune import Config as Study
from scripts.finetune import Row, normalize, read_cached, validate_tuning
from scripts.report import select, validate_evaluation


@dataclass(frozen=True)
class Config:
    """Report CSV directory and standalone HTML destination."""

    data: Path = Path("runs/results")  # Directory containing report CSVs and sidecars.
    output: Path | None = None  # Unused HTML path; defaults to DATA/figures.html.


@dataclass(frozen=True)
class Report:
    """Checksum-verified CSV rows and their recorded experiment request."""

    path: Path
    rows: tuple[Row, ...]
    request: dict
    sha256: str


def load(directory: Path, phase: str) -> Report:
    """Read one report and verify its phase, row schema, and checksum."""
    path = directory / f"{phase}.csv"
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(
            f"missing {path} or {sidecar}; generate the CSVs with "
            "uv run --locked python -m scripts.report --output "
            f"{shlex.quote(str(directory))}"
            + (" --runtime" if phase == "runtime" else "")
        )
    metadata = json.loads(sidecar.read_text())
    request = metadata.get("request", {})
    if request.get("phase") != phase:
        raise ValueError(f"CSV phase does not match {path}")
    return Report(
        path=path,
        rows=read_cached(path, request),
        request=request,
        sha256=metadata["sha256"],
    )


def validate(tuning: Report, evaluation: Report, runtime: Report | None) -> None:
    """Require complete comparisons at the settings selected by the tuning data."""
    study = Study(**evaluation.request["protocol"])
    protocol = asdict(study)
    protocol.pop("output")
    protocol.pop("evaluation_seeds")
    if tuning.request.get("protocol") != normalize(protocol):
        raise ValueError("tuning and evaluation protocols differ")
    if evaluation.request.get("tuning_sha256") != tuning.sha256:
        raise ValueError("evaluation refers to a different tuning CSV")
    validate_tuning(study, tuning.rows)
    winners = select(tuning.rows, study.selection_seeds)
    validate_evaluation(study, winners, evaluation.rows)
    if any(row.status != "completed" for row in evaluation.rows):
        raise ValueError(
            "learning figures require successful observations for every seed"
        )
    if runtime is not None:
        expected = {
            **evaluation.request,
            "phase": "runtime",
            "timing": "scalar single-call",
        }
        if runtime.request != expected:
            raise ValueError("runtime and evaluation requests differ")
        validate_evaluation(study, winners, runtime.rows, runtime=True)
        if any(row.status != "completed" for row in runtime.rows):
            raise ValueError(
                "runtime figures require successful observations for every seed"
            )


def page(root: Path, report: Report, rows: tuple[Row, ...], title: str) -> Path:
    """Snapshot complete trajectories into the shared renderer's page format."""
    root.mkdir()
    assets = Path(__file__).with_name("assets")
    shutil.copytree(assets / "viz", root / "viz")
    for name in ("index.html", "figures.mjs"):
        shutil.copyfile(assets / name, root / name)
    (root / "data").mkdir()
    groups: dict[tuple, list[Row]] = {}
    for row in rows:
        groups.setdefault((*row.cell, row.seed), []).append(row)
    labels = {"greq": "GReQ", "ql": "QL", "zap": "Zap", "zap-batched": "Zap-batched"}
    runs = []
    for index, (_, members) in enumerate(sorted(groups.items())):
        members.sort(key=lambda row: row.step)
        row = members[0]
        path = root / "data" / f"run-{index:04d}.csv"
        with path.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=[field.name for field in fields(Row)]
            )
            writer.writeheader()
            writer.writerows(asdict(member) for member in members)
        runs.append(
            CsvRun(
                file=path.relative_to(root).as_posix(),
                label=labels[row.method],
                source=str(report.path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                status="completed",
                seed=row.seed,
                environment="Graph (6×6 padded)"
                if row.environment == "graph"
                else f"Garnet ({row.n_states}×{row.n_actions})",
                gamma=row.gamma,
            )
        )
    runtime = report.request["phase"] == "runtime"
    manifest = asdict(
        Manifest(
            kind="runtime" if runtime else "learning",
            step_column="step",
            time_column="train_seconds" if runtime else None,
            metrics=(
                Metric(key="value_linf", label="‖Q − Q*‖∞"),
                Metric(key="bellman_linf", label="Bellman error"),
                Metric(key="policy_gap", label="Policy gap"),
                Metric(key="policy_return", label="Policy return"),
            ),
            runs=tuple(runs),
            title=title,
            time_description=(
                "Median of individually measured scalar sampling-and-learning times. "
                "Compilation, warmup, evaluation and I/O excluded."
            )
            if runtime
            else None,
        )
    )
    source = report.path.resolve()
    manifest["source_table"] = (
        source.relative_to(Path.cwd()).as_posix()
        if source.is_relative_to(Path.cwd())
        else source.name
    )
    manifest["source_sha256"] = report.sha256
    (root / "data/manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, allow_nan=False) + "\n"
    )
    return root / "index.html"


def run(config: Config) -> Path:
    """Validate results and write one HTML with embedded data and plotting assets."""
    output = (config.output or config.data / "figures.html").resolve()
    if not output.is_relative_to(Path("runs").resolve()) or output.suffix != ".html":
        raise ValueError("figure output must be an HTML file under runs/")
    if output.exists():
        raise FileExistsError(output)
    tuning = load(config.data, "tuning")
    evaluation = load(config.data, "evaluation")
    runtime = (
        load(config.data, "runtime")
        if any(
            (config.data / f"runtime.{suffix}").exists() for suffix in ("csv", "json")
        )
        else None
    )
    validate(tuning, evaluation, runtime)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".figures-", dir=output.parent) as temporary:
        root = Path(temporary)
        pages = []
        for environment, gamma in sorted(
            {(r.environment, r.gamma) for r in evaluation.rows}
        ):
            rows = tuple(
                r
                for r in evaluation.rows
                if (r.environment, r.gamma) == (environment, gamma)
            )
            first = rows[0]
            name = (
                "Graph"
                if environment == "graph"
                else f"Garnet ({first.n_states}×{first.n_actions})"
            )
            pages.append(
                page(
                    root / f"learning-{len(pages)}",
                    evaluation,
                    rows,
                    f"{name} — γ = {gamma:g} — 11 seeds",
                )
            )
        if runtime is not None:
            pages.append(
                page(
                    root / "runtime",
                    runtime,
                    runtime.rows,
                    "Median JIT time — 11 seeds",
                )
            )
        return gallery(
            GalleryConfig(pages=tuple(pages), output=output, standalone=True)
        )


def main() -> None:
    """Create an offline HTML figure gallery from report CSVs."""
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
