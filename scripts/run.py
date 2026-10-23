"""Single-setting experiments with scalar and vectorized JAX execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass as host_dataclass
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from chex import dataclass
from jaxdp.mdp import Mdp
from jaxtor.env.tabular import TabularEnv, graph
from jaxtor.sampler import Imc, Mc

from greq import BatchedZap, Greq, InverseGain, QLearning, Zap
from greq.environment import Garnet, require_irreducible
from greq.evaluation import Evaluation, evaluate, optimal_q
from greq.policy import Uniform


@host_dataclass(frozen=True)
class Environment:
    """Shared host settings for generated Garnet or the upstream padded graph.

    Graph dimensions are fixed at six states and six action labels. Its
    branching field is zero because Garnet's branching parameter does not apply.
    Unavailable graph actions retain the library's self-loop aliases.
    """

    environment: Literal["garnet", "graph"] = "garnet"
    n_states: int = 50  # Garnet states; graph always uses six.
    n_actions: int = 5  # Garnet actions; graph always uses six labels.
    branching: int = 5  # Garnet successors; graph records zero (not applicable).
    env_seed: int = 0  # Garnet instance seed; the upstream graph is fixed.

    def __post_init__(self) -> None:
        """Resolve the dimensions of the upstream padded graph."""
        if self.environment not in ("garnet", "graph"):
            raise ValueError("unknown environment")
        if not 0 <= self.env_seed < 2**32:
            raise ValueError("environment seed must be an unsigned 32-bit integer")
        if self.environment == "graph":
            object.__setattr__(self, "n_states", 6)
            object.__setattr__(self, "n_actions", 6)
            object.__setattr__(self, "branching", 0)
        else:
            Garnet(
                n_states=self.n_states,
                n_actions=self.n_actions,
                branching=self.branching,
            )

    def make_environment(self) -> TabularEnv:
        """Construct a tabular environment with the maximum int32 episode limit."""
        if self.environment == "graph":
            return graph.make(graph.Config(max_eps_len=2_147_483_647))
        return TabularEnv(
            config=Garnet(
                n_states=self.n_states,
                n_actions=self.n_actions,
                branching=self.branching,
            )
        )

    @property
    def title(self) -> str:
        """Identify the graph's padded representation in every generated figure."""
        if self.environment == "graph":
            return "Graph (6×6 padded)"
        return f"Garnet ({self.n_states}×{self.n_actions})"


@host_dataclass(frozen=True)
class Config(Environment):
    """Environment, algorithm, budget, and trajectory seed for one run."""

    method: Literal["greq", "ql", "zap", "zap-batched"] = "greq"
    output: Path = Path("runs/single")  # Unused output directory under runs/.
    n_steps: int = 10_000  # Samples, divisible by eval_every.
    eval_every: int = 100  # Samples between observations; final state is checkpointed.
    seed: int = 0  # Paired trajectory seed.
    gamma: float = 0.99
    eta: float | None = None  # Default 0.5 for QL, 1 for matrix-gain methods.
    rho: float = 0.7
    k0: float = 10.0  # QL global-clock scale.
    batch_size: int | None = None  # Samples per batched-Zap update; default A*S.
    max_gain: float | None = 1e6  # Zap gain bound; None selects raw inversion.
    initial_scale: float = 1.0  # Initial Zap Jacobian scale.

    def __post_init__(self) -> None:
        """Validate budgets, seeds, and artifact boundaries."""
        super().__post_init__()
        if self.method not in ("greq", "ql", "zap", "zap-batched"):
            raise ValueError("unknown learning method")
        if self.batch_size is not None and self.method != "zap-batched":
            raise ValueError("batch_size applies only to zap-batched")
        if not 0 < self.n_steps < 2_147_483_647:
            raise ValueError(
                "n_steps must fit a positive int32 below the episode limit"
            )
        if self.eval_every < 1 or self.n_steps % self.eval_every:
            raise ValueError("n_steps must be divisible by positive eval_every")
        if (
            not 0 <= self.seed < 2**32
            or int(self.seed) != self.seed
            or not 0 <= self.env_seed < 2**32
        ):
            raise ValueError("seeds must be unsigned 32-bit integers")
        root, output = Path("runs").resolve(), self.output.resolve()
        if output == root or not output.is_relative_to(root):
            raise ValueError("output must be a new subdirectory of runs/")


class LearningState(Protocol):
    """Action values consumed by experiment evaluation."""

    q: jax.Array


class Learner[S: LearningState](Protocol):
    """Initialization and parameterized updates consumed by the experiment."""

    def init(self, q: jax.Array, /) -> S: ...
    def update_with(
        self,
        transition: Mc.Transition,
        state: S,
        gamma: jax.Array,
        eta: jax.Array,
        other: jax.Array,
        /,
    ) -> S: ...


class Sampler[S](Protocol):
    """Continuing transitions consumed by the experiment."""

    def sample(self, state: S, /) -> tuple[Mc.Transition, S]: ...


@dataclass
class Experiment[EnvS, LearnS: LearningState]:
    """Advance sampling and learning by threading their array-only states.

    Required protocols::

        sampler.sample(state) -> (transition, state)
        learner.update_with(transition, state, gamma, eta, other) -> state
        learner_state.q

    Attributes:
        sampler: Continuing behavior sampler.
        learner: Parameterized learning recurrence.
        n_steps: Samples per run.
        eval_every: Samples between observations.

    Public dataclasses:
        State: Dynamic parameters, child states, evaluation data, and clock.
        Result: Final state and observation history.

    Public methods:
        init: Receive child states and initialize the experiment clock.
        advance: Consume one observation interval.
        observe: Evaluate the current state while preserving the trajectory.
        run: Complete one run and return its observation history.
    """

    sampler: Sampler[EnvS]
    learner: Learner[LearnS]
    n_steps: int
    eval_every: int

    @dataclass
    class State[SampleData, LearnData]:
        """Array-only experiment data.

        Attributes:
            row: Seed, discount, eta, and rho/k0, ``(4,)``.
            sampler: Behavior sampler state.
            learner: Learning state.
            evaluation: MDP arrays and certified reference.
            step: Completed samples, scalar.
        """

        row: jax.Array
        sampler: SampleData
        learner: LearnData
        evaluation: Evaluation.State
        step: jax.Array

    @dataclass
    class Result[S]:
        """Completed trajectory and sample-indexed diagnostics.

        Attributes:
            state: Final experiment state.
            metrics: Metrics with leading observation axis ``K``.
            finite: Validity of each observation, ``(K,)``.
        """

        state: S
        metrics: Evaluation.Metrics
        finite: jax.Array

    def __post_init__(self) -> None:
        """Reject invalid sample budgets."""
        if self.n_steps < 1 or self.eval_every < 1 or self.n_steps % self.eval_every:
            raise ValueError("sample budget must be divisible by evaluation interval")

    def init(
        self,
        row: jax.Array,
        sampler: EnvS,
        learner: LearnS,
        evaluation: Evaluation.State,
    ) -> Experiment.State[EnvS, LearnS]:
        """Initialize owned data while receiving explicitly initialized children."""
        chex.assert_shape(row, (4,))
        return self.State(
            row=row,
            sampler=sampler,
            learner=learner,
            evaluation=evaluation,
            step=jnp.array(0, dtype=jnp.int32),
        )

    def advance(
        self, state: Experiment.State[EnvS, LearnS]
    ) -> Experiment.State[EnvS, LearnS]:
        """Advance sampling and learning for one observation interval."""

        def step(_, state):
            transition, sampled = self.sampler.sample(state.sampler)
            return replace(
                state,
                sampler=sampled,
                learner=self.learner.update_with(
                    transition, state.learner, state.row[1], state.row[2], state.row[3]
                ),
                step=state.step + 1,
            )

        return jax.lax.fori_loop(0, self.eval_every, step, state)

    def observe(
        self, state: Experiment.State[EnvS, LearnS]
    ) -> tuple[Evaluation.Metrics, jax.Array]:
        """Evaluate the current Q-table and check all learner arrays."""
        metrics = evaluate(state.learner.q, state.evaluation, state.row[1])
        finite = jnp.all(jnp.isfinite(jnp.stack(jax.tree.leaves(metrics))))
        for leaf in jax.tree.leaves(state.learner):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        return metrics, finite

    def run(
        self, state: Experiment.State[EnvS, LearnS]
    ) -> Experiment.Result[Experiment.State[EnvS, LearnS]]:
        """Run one initialized trajectory; vectorization belongs to the caller."""
        first = self.observe(state)

        def interval(state, _):
            state = self.advance(state)
            return state, self.observe(state)

        state, history = jax.lax.scan(
            interval, state, None, length=self.n_steps // self.eval_every
        )
        metrics, finite = jax.tree.map(
            lambda a, b: jnp.concatenate((a[None], b)), first, history
        )
        return self.Result(state=state, metrics=metrics, finite=finite)


@host_dataclass(frozen=True)
class Runner[S]:
    """Host bindings to one configured component and its explicit initialization."""

    initialize: Callable[[jax.Array], S]
    run: Callable[[jax.Array], Experiment.Result[S]]
    advance: Callable[[S], S]
    observe: Callable[[S], tuple[Evaluation.Metrics, jax.Array]]


def _bind[S: LearningState](
    config: Config, gammas: jax.Array, learner: Learner[S]
) -> Runner:
    """Construct child states once and bind the single-configuration function."""
    env = config.make_environment()
    env_state = env.init(jax.random.key(config.env_seed, impl="threefry2x32"))
    require_irreducible(env_state.mdp)
    model = env_state.mdp
    mdp = Mdp(
        transition=model.transition,
        reward=model.reward,
        initial=model.initial,
        terminal=model.terminal,
    )
    references = jnp.stack([optimal_q(mdp, float(gamma)) for gamma in gammas])
    policy = Uniform(n_actions=config.n_actions)
    mc = Mc(env=env, max_eps_len=2_147_483_647)
    sampler = Imc(mc=mc, agent=policy)
    experiment = Experiment(
        sampler=sampler,
        learner=learner,
        n_steps=config.n_steps,
        eval_every=config.eval_every,
    )
    initial = learner.init(
        jnp.zeros((config.n_actions, config.n_states), dtype=jnp.float64)
    )

    def initialize(row):
        mc_key, policy_key = jax.random.split(
            jax.random.key(row[0].astype(jnp.uint32), impl="threefry2x32")
        )
        sampled = sampler.init(
            mc=mc.init(mc_key, env_state), agent=policy.init(policy_key)
        )
        evaluation = Evaluation.State(
            mdp=mdp, reference=references[jnp.argmax(gammas == row[1])]
        )
        return experiment.init(row, sampled, initial, evaluation)

    def run(row):
        return experiment.run(initialize(row))

    return Runner(
        initialize=initialize,
        run=run,
        advance=experiment.advance,
        observe=experiment.observe,
    )


def build(config: Config, gammas: jax.Array) -> Runner:
    """Configure a learner and its environment for the requested discounts."""
    eta = (
        config.eta
        if config.eta is not None
        else (0.5 if config.method == "ql" else 1.0)
    )
    if config.method == "ql":
        return _bind(
            config,
            gammas,
            QLearning(
                n_states=config.n_states,
                n_actions=config.n_actions,
                gamma=config.gamma,
                eta=eta,
                k0=config.k0,
            ),
        )
    if config.method == "greq":
        return _bind(
            config,
            gammas,
            Greq(
                n_states=config.n_states,
                n_actions=config.n_actions,
                gamma=config.gamma,
                eta=eta,
                rho=config.rho,
            ),
        )
    gain = InverseGain(max_norm=config.max_gain)
    if config.method == "zap":
        return _bind(
            config,
            gammas,
            Zap(
                n_states=config.n_states,
                n_actions=config.n_actions,
                gamma=config.gamma,
                eta=eta,
                rho=config.rho,
                gain=gain,
                initial_scale=config.initial_scale,
            ),
        )
    return _bind(
        config,
        gammas,
        BatchedZap(
            n_states=config.n_states,
            n_actions=config.n_actions,
            gamma=config.gamma,
            eta=eta,
            rho=config.rho,
            gain=gain,
            initial_scale=config.initial_scale,
            batch_size=config.batch_size
            if config.batch_size is not None
            else config.n_states * config.n_actions,
        ),
    )


def parameters(config: Config) -> np.ndarray:
    """Encode one setting as a float64 row, preserving the full uint32 seed."""
    eta = (
        config.eta
        if config.eta is not None
        else (0.5 if config.method == "ql" else 1.0)
    )
    return np.asarray(
        [
            config.seed,
            config.gamma,
            eta,
            config.k0 if config.method == "ql" else config.rho,
        ],
        dtype=np.float64,
    )


def run(config: Config) -> Experiment.Result:
    """Return one setting's result in memory using the vmappable runner."""
    jax.config.update("jax_enable_x64", True)
    runner = build(config, jnp.asarray([config.gamma]))
    return jax.jit(runner.run)(jnp.asarray(parameters(config)))


def main() -> None:
    """Write one trajectory's evaluation CSV and reproducibility evidence."""
    from scripts.finetune import execute

    config = tyro.cli(Config)
    execute(config, parameters(config)[None], mapped=False, history=True)
    print(config.output / "evaluation.csv")


if __name__ == "__main__":
    main()
