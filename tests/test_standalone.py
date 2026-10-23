"""Standalone report packaging preserves data and validates complete comparisons."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from scripts.figures.__main__ import Config, load, run, validate
from scripts.finetune import Config as Study
from scripts.finetune import Row, write_rows


@pytest.fixture
def reports(tmp_path, monkeypatch):
    """Write synthetic complete reports with the experiment CSV schema."""
    monkeypatch.chdir(tmp_path)
    study = Study(
        n_states=2,
        n_actions=2,
        branching=2,
        gammas=(0.9, 0.99),
        etas=(1.0,),
        rhos=(0.7,),
        ql_etas=(0.5,),
        ql_k0s=(10.0,),
        n_steps=4,
        eval_every=2,
    )
    protocol = asdict(study)
    protocol.pop("output")
    evaluation_request = {}
    for phase in ("tuning", "evaluation", "runtime"):
        rows = []
        seeds = study.selection_seeds if phase == "tuning" else study.evaluation_seeds
        steps = (study.n_steps,) if phase != "evaluation" else (0, 2, 4)
        for environment in study.environments:
            for method in study.methods:
                config = study.config(environment, method, phase)
                for gamma in study.gammas:
                    for seed in seeds:
                        for step in steps:
                            rows.append(
                                Row(
                                    environment=environment,
                                    method=method,
                                    execution="scalar"
                                    if phase == "runtime"
                                    else "vmap",
                                    n_states=config.n_states,
                                    n_actions=config.n_actions,
                                    branching=config.branching,
                                    env_seed=config.env_seed,
                                    n_steps=study.n_steps,
                                    eval_every=study.n_steps
                                    if phase == "runtime"
                                    else study.eval_every,
                                    gamma=gamma,
                                    eta=0.5 if method == "ql" else 1.0,
                                    rho=None if method == "ql" else 0.7,
                                    k0=10.0 if method == "ql" else None,
                                    batch_size=config.batch_size,
                                    seed=seed,
                                    step=step,
                                    status="completed",
                                    error="",
                                    train_seconds=0.1 + seed / 1000
                                    if phase == "runtime"
                                    else None,
                                    bellman_linf=0.5,
                                    value_linf=(1 + seed / 100) / (step + 1),
                                    policy_gap=0.3,
                                    policy_return=1.0,
                                )
                            )
        if phase == "tuning":
            specification = {
                "phase": phase,
                "protocol": {
                    k: v for k, v in protocol.items() if k != "evaluation_seeds"
                },
            }
        elif phase == "evaluation":
            evaluation_request = {
                "phase": phase,
                "protocol": protocol,
                "tuning_sha256": hashlib.sha256(
                    (study.output / "tuning.csv").read_bytes()
                ).hexdigest(),
            }
            specification = evaluation_request
        else:
            specification = {
                **evaluation_request,
                "phase": phase,
                "timing": "scalar single-call",
            }
        write_rows(study.output / f"{phase}.csv", tuple(rows), specification)
    return study.output


def test_standalone_contains_all_comparisons_and_exact_observations(reports):
    before = {p.name: p.read_bytes() for p in reports.iterdir()}
    output = run(Config())
    assert output == (reports / "figures.html").resolve()
    assert {p.name for p in reports.iterdir()} == {*before, "figures.html"}
    assert all((reports / name).read_bytes() == data for name, data in before.items())
    text = output.read_text()
    match = re.search(
        r'id="gallery-bundle" type="application/json">(.*?)</script>', text
    )
    assert match is not None
    bundle = json.loads(match[1])
    assert len(bundle["pages"]) == 5
    for phase, kind in (("evaluation", "learning"), ("runtime", "runtime")):
        embedded = []
        for page in bundle["pages"]:
            if page["manifest"]["kind"] != kind:
                continue
            assert page["manifest"]["source_table"] == f"runs/results/{phase}.csv"
            assert (
                page["manifest"]["source_sha256"]
                == hashlib.sha256(before[f"{phase}.csv"]).hexdigest()
            )
            for record in page["manifest"]["runs"]:
                data = page["csvs"][record["file"]]
                assert record["sha256"] == hashlib.sha256(data.encode()).hexdigest()
                embedded.extend(csv.DictReader(io.StringIO(data)))
        original = list(csv.DictReader(io.StringIO(before[f"{phase}.csv"].decode())))

        def key(row):
            return (
                row["environment"],
                row["method"],
                row["gamma"],
                row["seed"],
                row["step"],
            )

        assert sorted(embedded, key=key) == sorted(original, key=key)
    assert "greq-offline/viz/fig.mjs" in bundle["assets"]["figures.mjs"]
    assert "connect-src blob:" in text
    assert ".figures-" not in text
    assert str(Path.cwd()) not in text
    with pytest.raises(FileExistsError):
        run(Config())


def test_standalone_supports_learning_only_and_custom_paths(reports):
    for suffix in ("csv", "json"):
        (reports / f"runtime.{suffix}").unlink()
    data = reports.with_name("custom")
    reports.rename(data)
    output = run(Config(data=data, output=Path("runs/share/results.html")))
    assert output.is_file()
    assert '"kind": "runtime"' not in output.read_text()
    assert not list(output.parent.glob(".figures-*"))


@pytest.mark.parametrize(
    "change, message",
    [
        ("missing", "missing or unexpected"),
        ("setting", "frozen selection"),
        ("failed", "successful observations"),
        ("tuning_hash", "different tuning"),
        ("protocol", "protocols differ"),
        ("runtime_request", "requests differ"),
        ("runtime_failed", "successful observations"),
    ],
)
def test_standalone_rejects_inconsistent_comparisons(reports, change, message):
    tuning, evaluation, runtime = (
        load(reports, phase) for phase in ("tuning", "evaluation", "runtime")
    )
    if change == "missing":
        evaluation = replace(evaluation, rows=evaluation.rows[:-1])
    elif change == "setting":
        evaluation = replace(
            evaluation,
            rows=(replace(evaluation.rows[0], eta=0.9), *evaluation.rows[1:]),
        )
    elif change == "failed":
        evaluation = replace(
            evaluation,
            rows=(
                replace(evaluation.rows[0], status="failed", error="nonfinite"),
                *evaluation.rows[1:],
            ),
        )
    elif change == "tuning_hash":
        evaluation = replace(
            evaluation, request={**evaluation.request, "tuning_sha256": "wrong"}
        )
    elif change == "protocol":
        tuning = replace(tuning, request={**tuning.request, "protocol": {}})
    elif change == "runtime_request":
        runtime = replace(runtime, request={**runtime.request, "timing": "batch"})
    else:
        runtime = replace(
            runtime,
            rows=(
                replace(runtime.rows[0], status="failed", error="nonfinite"),
                *runtime.rows[1:],
            ),
        )
    with pytest.raises(ValueError, match=message):
        validate(tuning, evaluation, runtime)


@pytest.mark.parametrize(
    "failure", ["checksum", "phase", "sidecar", "runtime", "evaluation"]
)
def test_standalone_rejects_damaged_or_missing_files(reports, failure):
    path = reports / "evaluation.csv"
    if failure == "checksum":
        path.write_bytes(path.read_bytes() + b"\n")
    elif failure == "phase":
        metadata = json.loads(path.with_suffix(".json").read_text())
        metadata["request"]["phase"] = "runtime"
        path.with_suffix(".json").write_text(json.dumps(metadata))
    elif failure == "sidecar":
        path.with_suffix(".json").unlink()
    elif failure == "runtime":
        (reports / "runtime.json").unlink()
    else:
        path.unlink()
    with pytest.raises(ValueError):
        run(Config())
    assert not (reports / "figures.html").exists()
    assert not list(reports.glob(".figures-*"))


@pytest.mark.parametrize("output", [Path("outside.html"), Path("runs/figures.txt")])
def test_standalone_requires_an_html_destination_under_runs(reports, output):
    with pytest.raises(ValueError, match="HTML file under runs"):
        run(Config(output=output))
