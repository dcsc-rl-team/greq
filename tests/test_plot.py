"""CSV validation and reusable figure-page generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.figures.build import Config, build


def write_csv(directory, content=None, metadata=None, status=None):
    directory.mkdir(parents=True)
    path = directory / "metrics.csv"
    path.write_text(content or "step,train_seconds,error\n0,0,1\n10,0.1,0.5\n")
    if metadata is not None:
        (directory / "config.json").write_text(json.dumps(metadata))
    if status is not None:
        (directory / "status.json").write_text(json.dumps({"status": status}))
    return path


def test_reusable_page_and_input_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = write_csv(Path("runs/a"))
    b = write_csv(Path("runs/b"), "step,train_seconds,error\n0,0,2\n10,0.2,1\n")
    output = build(Config(csvs=(a, b), labels=("Method", "Method")))
    manifest = json.loads((output / "data/manifest.json").read_text())
    assert [run["label"] for run in manifest["runs"]] == ["Method", "Method"]
    assert manifest["metrics"] == [{"key": "error", "label": "error"}]
    assert manifest["title"] is None
    for path, run in zip((a, b), manifest["runs"], strict=True):
        copied = (output / run["file"]).read_bytes()
        assert copied == path.read_bytes()
        assert hashlib.sha256(copied).hexdigest() == run["sha256"]
    assert (output / "viz/vendor/d3.v7.min.js").is_file()
    assert (output / "viz/ui.css").is_file()
    assert (output / "figures.mjs").is_file()
    original_script = (output / "figures.mjs").read_bytes()
    second = build(Config(csvs=(b,), output=Path("runs/another")))
    assert (second / "figures.mjs").read_bytes() == original_script
    with pytest.raises(FileExistsError):
        build(Config(csvs=(a,), output=output))


def test_custom_columns_and_single_panel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_csv(Path("runs/custom"), "samples,loss,tag\n0,1,foo\n1,0.5,bar\n")
    output = build(
        Config(csvs=(path,), step_column="samples", time_column=None, metrics=("loss",))
    )
    manifest = json.loads((output / "data/manifest.json").read_text())
    assert manifest["step_column"] == "samples"
    assert manifest["time_column"] is None
    assert manifest["metrics"][0]["key"] == "loss"


@pytest.mark.parametrize(
    "content",
    [
        "step,train_seconds,error\n",
        "step,train_seconds,error\n0,0,nan\n",
        "step,train_seconds,error\n0,0,1\n0,1,2\n",
        "step,train_seconds,error\n0,1,1\n1,0,2\n",
        "step,step,error\n0,0,1\n",
        "wrong,error\n0,1\n",
    ],
)
def test_invalid_csv_rejected(content, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_csv(Path("runs/input"), content)
    with pytest.raises(ValueError):
        build(Config(csvs=(path,)))
    assert not Path("runs/figures").exists()


def test_failed_runs_require_explicit_inclusion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_csv(Path("runs/input"), status="failed")
    with pytest.raises(ValueError, match="failed"):
        build(Config(csvs=(path,)))
    output = build(Config(csvs=(path,), include_failed=True))
    manifest = json.loads((output / "data/manifest.json").read_text())
    assert manifest["runs"][0]["status"] == "failed"


@pytest.mark.parametrize("change", [{"gamma": 0.99}, {"seed": 1, "rho": 0.6}, {}])
def test_ambiguous_replications_rejected(change, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metadata = {"method": "zap", "gamma": 0.9, "seed": 0, "rho": 0.7}
    a = write_csv(Path("runs/a"), metadata=metadata)
    b = write_csv(Path("runs/b"), metadata={**metadata, **change})
    with pytest.raises(ValueError):
        build(Config(csvs=(a, b)))


def test_output_is_under_runs_and_inputs_are_distinct(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_csv(Path("runs/input"))
    with pytest.raises(ValueError, match="runs/"):
        build(Config(csvs=(path,), output=Path("figures")))
    with pytest.raises(ValueError, match="distinct"):
        build(Config(csvs=(path, path)))


@pytest.mark.parametrize("gamma", [0.9, 0.99])
def test_garnet_title_from_metadata(gamma, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_csv(
        Path("runs/input"),
        metadata={"gamma": gamma, "n_states": 2, "n_actions": 2, "branching": 2},
    )
    output = build(Config(csvs=(path,)))
    manifest = json.loads((output / "data/manifest.json").read_text())
    assert manifest["title"] == f"Garnet — γ = {gamma:g}"


def test_graph_title_and_environment_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metadata = {
        "environment": "graph",
        "gamma": 0.95,
        "n_states": 6,
        "n_actions": 6,
        "branching": 0,
        "seed": 0,
        "method": "greq",
    }
    a = write_csv(Path("runs/graph"), metadata=metadata)
    output = build(Config(csvs=(a,)))
    manifest = json.loads((output / "data/manifest.json").read_text())
    assert manifest["title"] == "Graph (6×6 padded) — γ = 0.95"
    b = write_csv(
        Path("runs/different"),
        metadata={**metadata, "method": "ql", "environment": "garnet"},
    )
    with pytest.raises(ValueError, match="environments"):
        build(Config(csvs=(a, b), output=Path("runs/mixed")))


def runtime_inputs():
    paths = []
    for environment in ("garnet", "graph"):
        for gamma in (0.95, 0.99):
            for method in ("greq", "ql", "nesa", "zap", "zap-batched"):
                directory = Path(f"runs/{environment}/{gamma}/{method}")
                path = write_csv(
                    directory,
                    metadata={
                        "environment": environment,
                        "n_states": 6,
                        "n_actions": 6,
                        "branching": 0 if environment == "graph" else 3,
                        "env_seed": 0,
                        "gamma": gamma,
                        "seed": 3,
                        "method": method,
                        "n_steps": 10,
                        "eval_every": 10,
                        "execution": {"mode": "scalar-jit-staged", "batch_size": 1},
                    },
                )
                (directory / "status.json").write_text(
                    json.dumps({"status": "completed", "completed_steps": 10})
                )
                paths.append(path)
    return tuple(paths)


def test_runtime_facets_allow_different_environments_and_discounts(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    paths = runtime_inputs()
    result = build(Config(csvs=paths, kind="runtime"))
    manifest = json.loads((result / "data/manifest.json").read_text())
    assert manifest["kind"] == "runtime" and len(manifest["runs"]) == 20
    assert {r["gamma"] for r in manifest["runs"]} == {0.95, 0.99}
    assert {r["environment"] for r in manifest["runs"]} == {
        "Garnet (6×6)",
        "Graph (6×6 padded)",
    }
    assert {r["seed"] for r in manifest["runs"]} == {3}


@pytest.mark.parametrize(
    "change",
    [
        "vmap",
        "incomplete",
        "seed",
        "budget",
        "instance",
        "missing",
        "zero",
        "replicate",
    ],
)
def test_runtime_rejects_unmatched_or_batch_timings(change, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = runtime_inputs()
    path = paths[0]
    config = path.parent / "config.json"
    metadata = json.loads(config.read_text())
    if change == "vmap":
        metadata["execution"]["mode"] = "single-flat-vmap-staged"
    elif change == "incomplete":
        (path.parent / "status.json").write_text(
            json.dumps({"status": "completed", "completed_steps": 9})
        )
    elif change == "seed":
        metadata["seed"] = 4
    elif change == "budget":
        metadata["n_steps"] = 20
    elif change == "instance":
        metadata["env_seed"] = 1
    elif change == "missing":
        paths = paths[:-1]
    elif change == "zero":
        path.write_text("step,train_seconds,error\n0,0,1\n10,0,0.5\n")
    else:
        duplicate = write_csv(Path("runs/replicate"), metadata=metadata)
        (duplicate.parent / "status.json").write_text(
            (path.parent / "status.json").read_text()
        )
        paths += (duplicate,)
    config.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        build(Config(csvs=paths, kind="runtime"))
    assert not Path("runs/figures").exists()


@pytest.mark.parametrize("mismatch", [None, "seeds", "settings"])
def test_runtime_seed_replicates_require_matched_sets_and_settings(
    mismatch, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    originals = runtime_inputs()
    paths = list(originals)
    for index, original in enumerate(originals):
        for seed in (4, 5):
            if mismatch == "seeds" and index == 0 and seed == 5:
                continue
            metadata = json.loads((original.parent / "config.json").read_text())
            metadata["seed"] = seed
            if mismatch == "settings" and index == 0:
                metadata["eta"] = 0.9
            replica = write_csv(
                Path(f"runs/replicas/{index}/{seed}"), metadata=metadata
            )
            (replica.parent / "status.json").write_text(
                (original.parent / "status.json").read_text()
            )
            paths.append(replica)
    if mismatch:
        with pytest.raises(
            ValueError, match="matched seed sets|identical algorithm settings"
        ):
            build(Config(csvs=tuple(paths), kind="runtime"))
    else:
        result = build(Config(csvs=tuple(paths), kind="runtime"))
        manifest = json.loads((result / "data/manifest.json").read_text())
        assert len(manifest["runs"]) == 60
        assert {run["seed"] for run in manifest["runs"]} == {3, 4, 5}


def test_explicit_title(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_csv(Path("runs/input"))
    output = build(Config(csvs=(path,), title="Custom experiment"))
    manifest = json.loads((output / "data/manifest.json").read_text())
    assert manifest["title"] == "Custom experiment"
