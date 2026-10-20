"""Run one reproducible Garnet pilot with GReQ or Q-learning."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from jaxdp.mdp import Mdp
from jaxtor.env.tabular import TabularEnv
from jaxtor.sampler import Imc, Mc

from greq import Greq, QLearning, checkpoint
from greq.environment import Garnet, require_irreducible
from greq.evaluation import Evaluation, optimal_q
from greq.online import Online, Uniform
from scripts.artifacts import (
    Record,
    Status,
    append_record,
    archive_provenance,
    write_status,
)


@dataclass(frozen=True)
class Config:
    """Configuration for one run, not a tuned paper benchmark."""

    method: Literal["greq", "ql"] = "greq"  # Learning algorithm.
    output: Path = Path("runs/pilot")  # New directory beneath runs/; never overwritten.
    n_steps: int = 10_000  # Total samples, divisible by eval_every.
    eval_every: int = 1_000  # Samples between exact evaluations and checkpoints.
    env_seed: int = 0  # MDP instance seed, independent of trajectory randomness.
    seed: int = 0  # Paired behavior/environment-step seed.
    n_states: int = 10  # Garnet state count.
    n_actions: int = 5  # Garnet action count.
    branching: int = 5  # Distinct successors per state-action pair.
    gamma: float = 0.99  # Continuing-task discount factor.
    eta: float | None = None  # Q exponent; resolved to 1.0 for GReQ or 0.5 for QL.
    rho: float = 0.7  # GReQ model exponent.
    k0: float = 10.0  # QL global-clock scale.

    def __post_init__(self) -> None:
        """Validate the sample budget, seed ranges, and artifact location."""
        if self.method not in ("greq", "ql"):
            raise ValueError("method must be greq or ql")
        if not 0 < self.n_steps < 2_147_483_647:
            raise ValueError(
                "n_steps must fit a positive int32 below the episode limit"
            )
        if self.eval_every < 1 or self.n_steps % self.eval_every:
            raise ValueError("n_steps must be divisible by positive eval_every")
        if not 0 <= self.seed < 2**32 or not 0 <= self.env_seed < 2**32:
            raise ValueError("seeds must be unsigned 32-bit integers")
        if not self.output.resolve().is_relative_to(Path("runs").resolve()):
            raise ValueError("experiment artifacts must be written under runs/")
        if self.output.resolve() == Path("runs").resolve():
            raise ValueError("output must be a new subdirectory of runs/")


class LearningState(Protocol):
    """Q-values consumed by the experiment evaluator."""

    q: jax.Array


class Learner[S: LearningState](Protocol):
    """Online update capability consumed by the experiment driver."""

    def update(self, transition: Mc.Transition, state: S, /) -> S: ...


def _record(step: int, train_seconds: float, metrics: Evaluation.Metrics) -> Record:
    """Move scalar diagnostics to the host and reject nonfinite results."""
    values = [float(value) for value in jax.tree.leaves(metrics)]
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite evaluation metric")
    return Record(
        step=step,
        train_seconds=train_seconds,
        bellman_linf=float(metrics.bellman_linf),
        value_linf=float(metrics.value_linf),
        policy_return=float(metrics.policy_return),
        policy_gap=float(metrics.policy_gap),
    )


def _execute[S: LearningState](
    config: Config,
    learner: Learner[S],
    learner_state: S,
    env: TabularEnv,
    env_state: TabularEnv.State,
) -> None:
    """Compose the run and retain its last state even if diagnostics fail."""
    directory = config.output
    mc_key, policy_key = jax.random.split(
        jax.random.key(config.seed, impl="threefry2x32")
    )
    policy = Uniform(n_actions=config.n_actions)
    mc = Mc(env=env, max_eps_len=2_147_483_647)
    sampler = Imc(mc=mc, agent=policy)
    sampler_state = sampler.init(
        mc=mc.init(mc_key, env_state), agent=policy.init(policy_key)
    )
    online = Online(sampler=sampler, learner=learner, n_steps=config.eval_every)
    state = online.init(sampler=sampler_state, learner=learner_state)
    checkpoint.save(directory / "checkpoint.npz", state)
    model = env_state.mdp
    mdp = Mdp(
        transition=model.transition,
        reward=model.reward,
        initial=model.initial,
        terminal=model.terminal,
    )
    reference = optimal_q(mdp, config.gamma)
    np.savez_compressed(directory / "reference.npz", q=np.asarray(reference))
    evaluator = Evaluation(gamma=config.gamma)
    eval_state = evaluator.init(mdp, reference)
    evaluate = jax.jit(evaluator.evaluate)
    append_record(directory, _record(0, 0.0, evaluate(state.learner.q, eval_state)))
    started = time.perf_counter()
    advance = jax.jit(online.advance).lower(state).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    jax.block_until_ready(advance(state))
    warmup_seconds = time.perf_counter() - started
    train_seconds = 0.0
    write_status(
        directory,
        Status(
            status="running",
            compile_seconds=compile_seconds,
            warmup_seconds=warmup_seconds,
        ),
    )
    for step in range(config.eval_every, config.n_steps + 1, config.eval_every):
        started = time.perf_counter()
        state = jax.block_until_ready(advance(state))
        train_seconds += time.perf_counter() - started
        checkpoint.save(directory / "checkpoint.npz", state)
        write_status(
            directory,
            Status(
                status="running",
                completed_steps=step,
                compile_seconds=compile_seconds,
                warmup_seconds=warmup_seconds,
            ),
        )
        leaves = jax.tree.leaves(state.learner)
        if not all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
            raise FloatingPointError(f"nonfinite learning state at sample {step}")
        record = _record(step, train_seconds, evaluate(state.learner.q, eval_state))
        append_record(directory, record)
        write_status(
            directory,
            Status(
                status="completed" if step == config.n_steps else "running",
                completed_steps=step,
                compile_seconds=compile_seconds,
                warmup_seconds=warmup_seconds,
            ),
        )
        print(f"step={step} bellman_linf={record.bellman_linf:.6g}")


def run(config: Config) -> Path:
    """Execute one float64 pilot without overwriting any existing run directory."""
    jax.config.update("jax_enable_x64", True)
    eta = (
        config.eta
        if config.eta is not None
        else (1.0 if config.method == "greq" else 0.5)
    )
    directory = config.output
    directory.mkdir(parents=True, exist_ok=False)
    resolved = asdict(config)
    resolved["output"] = str(config.output)
    resolved["eta"] = eta
    write_status(directory, Status(status="running"))
    try:
        (directory / "config.json").write_text(
            json.dumps(resolved, indent=2, allow_nan=False) + "\n"
        )
        archive_provenance(directory)
        env = TabularEnv(
            config=Garnet(
                n_states=config.n_states,
                n_actions=config.n_actions,
                branching=config.branching,
            )
        )
        env_state = env.init(jax.random.key(config.env_seed, impl="threefry2x32"))
        mdp = env_state.mdp
        np.savez_compressed(
            directory / "environment.npz",
            transition=np.asarray(mdp.transition),
            reward=np.asarray(mdp.reward),
            initial=np.asarray(mdp.initial),
            terminal=np.asarray(mdp.terminal),
        )
        require_irreducible(mdp)
        q = jnp.zeros((config.n_actions, config.n_states), dtype=jnp.float64)
        if config.method == "greq":
            agent = Greq(
                n_states=config.n_states,
                n_actions=config.n_actions,
                gamma=config.gamma,
                eta=eta,
                rho=config.rho,
            )
            _execute(config, agent, agent.init(q), env, env_state)
        else:
            baseline = QLearning(
                n_states=config.n_states,
                n_actions=config.n_actions,
                gamma=config.gamma,
                eta=eta,
                k0=config.k0,
            )
            _execute(config, baseline, baseline.init(q), env, env_state)
    except Exception as error:
        previous = json.loads((directory / "status.json").read_text())
        previous["status"] = "failed"
        previous["error"] = f"{type(error).__name__}: {error}"
        write_status(directory, Status(**previous))
        raise
    return directory


def main() -> None:
    """Parse the tyro CLI and execute one run."""
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
