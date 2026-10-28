"""Experiment workflow contracts, independent scalar timing, and cache safety."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from greq import checkpoint
from scripts import finetune, report, run


@pytest.mark.parametrize("method", ["greq", "ql", "nesa", "zap", "zap-batched"])
def test_scalar_run_and_flat_vectorization(method, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = run.Config(
        method=method,
        output=Path("runs/single"),
        n_states=2,
        n_actions=2,
        branching=2,
        gamma=0.9,
        eta=0.9,
        rho=0.6,
        k0=5.0 if method == "nesa" else None,
        zeta=0.75 if method == "nesa" else 1.0,
        n_steps=4,
        eval_every=2,
        batch_size=3 if method == "zap-batched" else None,
    )
    table = np.array(
        [[seed, *run.parameters(cfg)[1:]] for seed in (3, 4)],
        dtype=np.float64,
    )
    runner = run.build(cfg, jnp.array([0.9]))
    result = jax.jit(jax.vmap(runner.run))(jnp.asarray(table))
    assert result.finite.shape == (2, 3)
    assert np.asarray(result.finite).all()
    for index, row in enumerate(table):
        individual = run.run(replace(cfg, seed=int(row[0])))
        np.testing.assert_allclose(
            result.state.learner.q[index],
            individual.state.learner.q,
            atol=1e-7,
            rtol=1e-7,
        )
        assert int(individual.state.step) == 4
        path = tmp_path / f"{method}-{index}.npz"
        checkpoint.save(path, individual.state)
        restored = checkpoint.load(path, runner.initialize(jnp.asarray(row)))
        np.testing.assert_array_equal(restored.learner.q, individual.state.learner.q)
        assert int(restored.step) == 4


@pytest.mark.parametrize("period", [1, 3, 20])
def test_flat_run_preserves_lazy_batched_gain(period, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    class ObservedGain:
        def __init__(self, *, max_norm):
            pass

        def apply(self, jacobian, vector):
            jax.debug.callback(lambda value: calls.append(float(value)), jacobian[0, 0])
            return vector

    monkeypatch.setattr(run, "InverseGain", ObservedGain)
    cfg = run.Config(
        method="zap-batched",
        batch_size=period,
        output=Path("runs/lazy"),
        n_states=2,
        n_actions=2,
        branching=2,
        n_steps=8,
        eval_every=2,
    )
    runner = run.build(cfg, jnp.asarray([0.9]))
    rows = jnp.asarray(
        [[seed, 0.9, 1.0, 0.7] for seed in (3, 4, 2**32 - 1)], dtype=jnp.float64
    )
    result = jax.block_until_ready(jax.jit(jax.vmap(runner.run))(rows))
    jax.effects_barrier()
    assert len(calls) == 3 * (8 // period)
    np.testing.assert_array_equal(result.state.learner.step, [8] * 3)
    np.testing.assert_array_equal(result.state.learner.batch_step, [8 % period] * 3)
    assert np.asarray(result.finite).all()


def test_report_generates_missing_tuning_and_reuses_valid_csvs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = report.Config(
        n_states=2,
        n_actions=2,
        branching=2,
        n_steps=4,
        eval_every=2,
        gammas=(0.9,),
        selection_seeds=(0, 1),
        etas=(1.0,),
        rhos=(0.7,),
        ql_etas=(0.5,),
        ql_k0s=(10.0,),
        runtime=True,
    )
    assert cfg.output == Path("runs/results")
    path = report.run(cfg)
    tuning = finetune.read_rows(cfg.output / "tuning.csv")
    evaluation = finetune.read_rows(path)
    runtime = finetune.read_rows(cfg.output / "runtime.csv")
    assert len(tuning) == 16
    assert len(evaluation) == 264
    assert len(runtime) == 88
    assert {r.seed for r in evaluation} == set(range(3, 14))
    assert all(r.train_seconds is None for r in evaluation)
    assert all(r.train_seconds is not None and r.train_seconds > 0 for r in runtime)
    assert all(r.eval_every == r.n_steps for r in runtime)
    assert all(r.status == "completed" for r in tuning + evaluation + runtime)
    winners = json.loads((cfg.output / "selection.json").read_text())["winners"]
    assert len(winners) == 8
    before = {p.name: p.read_bytes() for p in cfg.output.glob("*.csv")}

    def forbidden(*args, **kwargs):
        raise AssertionError("valid cache triggered computation")

    monkeypatch.setattr(finetune, "finetune", forbidden)
    monkeypatch.setattr(report, "execute", forbidden)
    monkeypatch.setattr(report, "benchmark", forbidden)
    assert report.run(cfg) == path
    assert before == {p.name: p.read_bytes() for p in cfg.output.glob("*.csv")}
    original_read = report.read_cached
    for transform, message in (
        (lambda rows: rows[:-1], "missing"),
        (lambda rows: (replace(rows[0], eta=0.95), *rows[1:]), "frozen selection"),
    ):
        with monkeypatch.context() as patch:

            def incomplete(path, specification, transform=transform):
                rows = original_read(path, specification)
                return transform(rows) if path.name == "evaluation.csv" else rows

            patch.setattr(report, "read_cached", incomplete)
            with pytest.raises(ValueError, match=message):
                report.run(cfg)
    with pytest.raises(ValueError, match="different protocol"):
        report.run(replace(cfg, n_steps=8))
    with (cfg.output / "tuning.csv").open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        report.run(cfg)


def test_runtime_uses_scalar_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime called vmap")

    monkeypatch.setattr(jax, "vmap", forbidden)
    cfg = run.Config(
        output=Path("runs/runtime"),
        n_states=2,
        n_actions=2,
        branching=2,
        n_steps=4,
        eval_every=2,
        eta=1.0,
    )
    table = np.array([[3, 0.99, 1, 0.7], [4, 0.99, 1, 0.7]], dtype=np.float64)
    rows = report.benchmark(cfg, table)
    assert [r.seed for r in rows] == [3, 4]
    assert all(r.step == 4 and r.train_seconds > 0 for r in rows)


def test_selection_rejects_failed_replicates_and_missing_seeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = run.Config(
        output=Path("runs/tuning"),
        n_states=2,
        n_actions=2,
        branching=2,
        n_steps=2,
        eval_every=2,
        eta=1.0,
    )
    table = np.array([[0, 0.99, 1, 0.7], [1, 0.99, 1, 0.7]], dtype=np.float64)
    rows = finetune.finetune(cfg, table)
    winner = report.select(rows, (0, 1))[0]
    assert winner.score == np.median([r.value_linf for r in rows])
    tied = rows + tuple(replace(r, eta=0.9) for r in rows)
    assert report.select(tied, (0, 1))[0].runs[0].eta == 0.9
    huge = tuple(replace(r, value_linf=1e308) for r in rows)
    assert report.select(huge, (0, 1))[0].score == 1e308
    with pytest.raises(ValueError, match="missing"):
        report.select(rows[:1], (0, 1))
    failed = (replace(rows[0], status="failed", error="nonfinite"), rows[1])
    with pytest.raises(ValueError, match="no eligible"):
        report.select(failed, (0, 1))
    with pytest.raises(ValueError, match="unknown"):
        replace(rows[0], method="unsupported")
    protocol = finetune.Config(
        output=Path("runs/grid"),
        environments=("garnet",),
        methods=("greq",),
        gammas=(0.99,),
        selection_seeds=(0, 1),
        etas=(1.0,),
        rhos=(0.7,),
        n_states=2,
        n_actions=2,
        branching=2,
        n_steps=2,
        eval_every=2,
    )
    finetune.validate_tuning(protocol, rows)
    with pytest.raises(ValueError, match="complete requested grid"):
        finetune.validate_tuning(protocol, rows[:1])
    with pytest.raises(ValueError, match="scalar"):
        replace(rows[0], train_seconds=0.1)
    with pytest.raises(ValueError, match="metadata"):
        finetune.validate_tuning(protocol, tuple(replace(r, env_seed=4) for r in rows))


def test_checkpoint_failure_preserves_computed_observations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = run.Config(
        output=Path("runs/failure"),
        n_states=2,
        n_actions=2,
        branching=2,
        n_steps=2,
        eval_every=2,
    )
    table = np.array([[0, 0.99, 1, 0.7], [1, 0.99, 1, 0.7]], dtype=np.float64)

    def fail(*args, **kwargs):
        raise OSError("injected checkpoint failure")

    monkeypatch.setattr(finetune.checkpoint, "save", fail)
    with pytest.raises(OSError, match="injected"):
        finetune.finetune(cfg, table)
    assert len(finetune.read_rows(cfg.output / "tuning.csv")) == 2
    assert json.loads((cfg.output / "status.json").read_text())["status"] == "failed"
    cfg = replace(cfg, output=Path("runs/runtime-failure"))
    with pytest.raises(OSError, match="injected"):
        report.benchmark(cfg, table)
    progress = json.loads((cfg.output / "progress.json").read_text())
    assert progress["finished_rows"] == 1
    assert progress["rows"][0]["train_seconds"] > 0
    assert json.loads((cfg.output / "status.json").read_text())["status"] == "failed"


def test_nesa_report_figures_momentum_selection_and_legacy_csv(tmp_path, monkeypatch):
    from scripts.figures.__main__ import Config as FigureConfig
    from scripts.figures.__main__ import run as figures

    monkeypatch.chdir(tmp_path)
    cfg = report.Config(
        methods=("nesa",),
        n_states=2,
        n_actions=2,
        branching=2,
        n_steps=12,
        eval_every=4,
        gammas=(0.9,),
        selection_seeds=(0, 1),
        nesa_etas=(0.6, 1.0),
        nesa_k0s=(1.0, 10.0),
        nesa_zetas=(0.5, 1.0),
        runtime=True,
    )
    assert cfg.table("nesa").shape == (16, 5)
    path = report.run(cfg)
    tuning = finetune.read_rows(cfg.output / "tuning.csv")
    evaluation = finetune.read_rows(path)
    runtime = finetune.read_rows(cfg.output / "runtime.csv")
    assert len(tuning) == 32
    assert len(evaluation) == 88
    assert len(runtime) == 22
    for row in tuning + evaluation + runtime:
        assert row.method == "nesa" and row.status == "completed"
        assert row.eta in cfg.nesa_etas and row.k0 in cfg.nesa_k0s
        assert row.zeta in cfg.nesa_zetas and row.rho is None
    winners = report.select(tuning, cfg.selection_seeds)
    settings = {winner.runs[0].cell: winner.runs[0].setting for winner in winners}
    assert all(row.setting == settings[row.cell] for row in evaluation + runtime)
    assert {row.seed for row in evaluation} == set(range(3, 14))
    assert all(row.train_seconds > 0 for row in runtime)
    tied = tuple(replace(row, value_linf=1.0) for row in tuning)
    assert all(w.runs[0].zeta == 0.5 for w in report.select(tied, cfg.selection_seeds))
    for value in (None, 0, float("inf")):
        with pytest.raises(ValueError, match="zeta|nonfinite"):
            replace(tuning[0], zeta=value)
    legacy = replace(tuning[0], method="ql", zeta=None)
    record = asdict(legacy)
    record.pop("zeta")
    legacy_path = Path("legacy.csv")
    with legacy_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    assert finetune.read_rows(legacy_path) == (legacy,)
    with pytest.raises(ValueError, match="only NeSA"):
        replace(legacy, zeta=0.5)
    output = figures(FigureConfig())
    assert output.is_file() and '"label": "NeSA"' in output.read_text()

    def forbidden(*args, **kwargs):
        raise AssertionError("valid NeSA cache triggered computation")

    monkeypatch.setattr(finetune, "finetune", forbidden)
    monkeypatch.setattr(report, "execute", forbidden)
    monkeypatch.setattr(report, "benchmark", forbidden)
    assert report.run(cfg) == path
    with pytest.raises(ValueError, match="different protocol"):
        report.run(replace(cfg, nesa_zetas=(1.0,)))


@pytest.mark.parametrize("name,index", [("eta", 2), ("k0", 3), ("zeta", 4)])
@pytest.mark.parametrize("value", [0, -0.1, float("inf"), float("nan")])
def test_nesa_experiment_validation(name, index, value):
    with pytest.raises(ValueError, match="finite|positive"):
        run.build(run.Config(method="nesa", **{name: value}), jnp.array([0.9]))
    table = np.array([[0, 0.9, 1, 1, 1]], dtype=np.float64)
    table[0, index] = value
    with pytest.raises(ValueError, match="finite|positive"):
        finetune.validate_table(table, np.array([0.9]), "nesa")


def test_nesa_rows_map_all_parameters_discounts_and_full_uint32_seeds():
    cfg = run.Config(
        method="nesa", n_states=2, n_actions=2, branching=2, n_steps=32, eval_every=8
    )
    table = np.array(
        [[0, 0.9, 1, 1, 1], [2**32 - 1, 0.999, 0.6, 10, 0.25]], dtype=np.float64
    )
    finetune.validate_table(table, np.array([0.9, 0.999]), "nesa")
    runner = run.build(cfg, jnp.array([0.9, 0.999]))
    result = jax.jit(jax.vmap(runner.run))(jnp.asarray(table))
    assert np.asarray(result.finite).all()
    for index, row in enumerate(table):
        scalar = run.run(
            replace(
                cfg,
                seed=int(row[0]),
                gamma=float(row[1]),
                eta=float(row[2]),
                k0=float(row[3]),
                zeta=float(row[4]),
            )
        )
        mapped = jax.tree.map(lambda value, index=index: value[index], result)
        for actual, expected in zip(
            jax.tree.leaves(mapped.state.learner),
            jax.tree.leaves(scalar.state.learner),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(mapped.state.row, row)


def test_nesa_source_point_is_in_the_matched_budget_and_defaults_are_unchanged():
    np.testing.assert_array_equal(
        run.parameters(run.Config(method="ql")), [0, 0.99, 0.5, 10]
    )
    np.testing.assert_array_equal(
        run.parameters(run.Config(method="nesa")), [0, 0.99, 1, 1, 1]
    )
    study = finetune.Config()
    assert study.methods == ("greq", "ql", "zap", "zap-batched")
    assert study.table("nesa").shape == (4 * 24 * 3, 5)
    assert study.table("ql").shape == (4 * 24 * 3, 4)
    assert [0, 0.95, 1, 1, 1] in study.table("nesa").tolist()
    with pytest.raises(ValueError, match="H×5"):
        finetune.validate_table(
            np.array([[0, 0.9, 1, 1.0]], dtype=np.float64), np.array([0.9]), "nesa"
        )
    with pytest.raises(ValueError, match="duplicate"):
        replace(study, methods=("nesa",), nesa_zetas=(1.0, 1.0))
    with pytest.raises(ValueError, match="only to NeSA"):
        run.Config(method="ql", zeta=0.5)


def test_protocol_requires_disjoint_eleven_seed_evaluation():
    with pytest.raises(ValueError, match="11"):
        finetune.Config(evaluation_seeds=(3, 4))
    with pytest.raises(ValueError, match="disjoint"):
        finetune.Config(evaluation_seeds=tuple(range(11)))
