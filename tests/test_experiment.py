"""Pilot artifacts, paired seeds, failure retention, and overwrite protection."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts import run as experiment
from scripts.run import Config, run


def config(method="greq", output="runs/first", **kwargs):
    """Small fully connected MDP suitable for integration tests."""
    return Config(
        method=method,
        output=Path(output),
        n_steps=10,
        eval_every=5,
        n_states=2,
        n_actions=2,
        branching=2,
        **kwargs,
    )


def records(path):
    """Load learning metrics without nondeterministic timing measurements."""
    with (path / "metrics.csv").open() as file:
        result = list(csv.DictReader(file))
    for row in result:
        assert float(row.pop("train_seconds")) >= 0
    return result


@pytest.mark.parametrize("method", ["greq", "ql"])
def test_repeatable_run_and_complete_artifacts(method, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config(method)
    first = run(cfg)
    second = run(replace(cfg, output=Path("runs/second")))
    assert records(first) == records(second)
    assert [int(row["step"]) for row in records(first)] == [0, 5, 10]
    for filename in ("checkpoint.npz", "reference.npz", "environment.npz"):
        with (
            np.load(first / filename, allow_pickle=False) as a,
            np.load(second / filename, allow_pickle=False) as b,
        ):
            assert set(a.files) == set(b.files)
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])
    saved = json.loads((first / "config.json").read_text())
    assert saved["eta"] == (1.0 if method == "greq" else 0.5)
    status = json.loads((first / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["completed_steps"] == 10
    assert status["compile_seconds"] >= 0
    assert status["warmup_seconds"] >= 0
    provenance = json.loads((first / "provenance.json").read_text())
    assert provenance["jax_x64"] is True
    assert provenance["prng_impl"] == "threefry2x32"
    for source in provenance["source"]:
        data = (first / "source" / source["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source["sha256"]
    dependencies = {dep["name"]: dep for dep in provenance["dependencies"]}
    for name in ("jaxdp", "jaxtor"):
        direct = json.loads(dependencies[name]["direct_url"])
        assert len(direct["vcs_info"]["commit_id"]) == 40
    before = (first / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(cfg)
    assert (first / "config.json").read_bytes() == before


def test_failed_run_is_retained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config(rho=1.0)
    with pytest.raises(ValueError, match="exponents"):
        run(cfg)
    assert (cfg.output / "config.json").exists()
    assert (cfg.output / "environment.npz").exists()
    status = json.loads((cfg.output / "status.json").read_text())
    assert status["status"] == "failed"
    assert "ValueError" in status["error"]


def test_diagnostic_failure_retains_current_checkpoint_and_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = experiment._record

    def fail_after_initial(step, seconds, metrics):
        if step:
            raise FloatingPointError("injected diagnostic failure")
        return original(step, seconds, metrics)

    monkeypatch.setattr(experiment, "_record", fail_after_initial)
    cfg = config()
    with pytest.raises(FloatingPointError, match="injected"):
        run(cfg)
    status = json.loads((cfg.output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["completed_steps"] == 5
    assert (cfg.output / "checkpoint.npz").is_file()
    assert len(records(cfg.output)) == 1


def test_nonfinite_config_is_recorded_as_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config(gamma=float("nan"))
    with pytest.raises(ValueError):
        run(cfg)
    assert json.loads((cfg.output / "status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("output", ["results/run", "runs", "runs/../escape"])
def test_output_must_be_a_run_subdirectory(output, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="runs/"):
        config(output=output)


def test_eval_frequency_does_not_change_learning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = run(config())
    second = run(replace(config(output="runs/second"), eval_every=10))
    with (
        np.load(first / "checkpoint.npz", allow_pickle=False) as a,
        np.load(second / "checkpoint.npz", allow_pickle=False) as b,
    ):
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
    assert records(first)[-1] == records(second)[-1]
