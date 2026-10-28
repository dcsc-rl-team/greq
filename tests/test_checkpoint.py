"""Sampling, chunking, and checkpoint continuation preserve the trajectory."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxdp.mdp import Mdp
from jaxtor.env.tabular import TabularEnv
from jaxtor.sampler import Imc, Mc

from greq import BatchedZap, Greq, InverseGain, NeSA, QLearning, Zap, checkpoint
from greq.environment import Garnet
from greq.evaluation import Evaluation, optimal_q
from greq.policy import Uniform
from scripts.run import Experiment


def composition(method=Greq, block=5, gamma=0.9):
    """Build independent behavior and transition streams on one fixed MDP."""
    env = TabularEnv(config=Garnet(n_states=3, n_actions=2, branching=3))
    env_state = env.init(jax.random.key(0))
    mc = Mc(env=env, max_eps_len=2_147_483_647)
    policy = Uniform(n_actions=2)
    sampler = Imc(mc=mc, agent=policy)
    sampler_state = sampler.init(
        mc=mc.init(jax.random.key(1), env_state), agent=policy.init(jax.random.key(2))
    )
    agent = method(n_states=3, n_actions=2, gamma=gamma)
    experiment = Experiment(
        sampler=sampler, learner=agent, n_steps=block, eval_every=block
    )
    model = env_state.mdp
    mdp = Mdp(
        transition=model.transition,
        reward=model.reward,
        initial=model.initial,
        terminal=model.terminal,
    )
    row = jnp.asarray(
        [
            0,
            gamma,
            agent.eta,
            agent.k0 if isinstance(agent, (QLearning, NeSA)) else agent.rho,
            *((agent.zeta,) if isinstance(agent, NeSA) else ()),
            *((agent.t0,) if isinstance(agent, NeSA) and agent.t0 is not None else ()),
        ]
    )
    initial = experiment.init(
        row,
        sampler_state,
        agent.init(jnp.zeros((2, 3))),
        Evaluation.State(mdp=mdp, reference=optimal_q(mdp, gamma)),
    )
    return experiment, initial


def assert_same(left, right):
    """Compare data and typed PRNG keys exactly."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "method",
    [
        Greq,
        QLearning,
        NeSA,
        partial(NeSA, eta=0.6, k0=10, zeta=0.25),
        partial(NeSA, t0=100, zeta=0.1),
        partial(Zap, gain=InverseGain()),
        partial(BatchedZap, gain=InverseGain(), batch_size=7),
    ],
)
def test_checkpoint_continuation(method, tmp_path):
    """Restored sampler and learning state reproduce the next block."""
    experiment, initial = composition(method)
    advance = jax.jit(experiment.advance)
    middle = advance(initial)
    path = tmp_path / "state.npz"
    checkpoint.save(path, middle)
    restored = checkpoint.load(path, initial)
    assert_same(middle, restored)
    assert_same(advance(middle), advance(restored))
    assert not path.with_name("state.npz.tmp").exists()


def test_checkpoint_schema_mismatch(tmp_path):
    _, initial = composition()
    _, other = composition(QLearning)
    path = tmp_path / "state.npz"
    checkpoint.save(path, initial)
    with pytest.raises(ValueError, match="match"):
        checkpoint.load(path, other)


def test_checkpoint_rejects_corrupt_leaf(tmp_path):
    _, initial = composition()
    path = tmp_path / "state.npz"
    checkpoint.save(path, initial)
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays["leaf_0"] = np.zeros(99)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="schema"):
        checkpoint.load(path, initial)


@pytest.mark.parametrize(
    "method", [Greq, NeSA, partial(NeSA, zeta=0.5), partial(NeSA, t0=100, zeta=0.1)]
)
def test_block_boundaries_do_not_reset_chain(method):
    small, first = composition(method, block=5)
    large, second = composition(method, block=10)
    advance = jax.jit(small.advance)
    after_small = advance(advance(first))
    after_large = jax.jit(large.advance)(second)
    assert_same(after_small, after_large)
    assert int(after_small.sampler.mc.eps_idx) == 10
    assert int(after_small.sampler.mc.env.step) == 10


@pytest.mark.parametrize(
    "method",
    [
        Greq,
        NeSA,
        partial(NeSA, eta=0.6, k0=10, zeta=0.25),
        partial(NeSA, t0=100, zeta=0.1),
        partial(Zap, gain=InverseGain()),
        partial(BatchedZap, gain=InverseGain(), batch_size=7),
    ],
)
def test_methods_have_identical_behavior_streams(method):
    full, a = composition(method)
    baseline, b = composition(QLearning)
    a = jax.jit(full.advance)(a)
    b = jax.jit(baseline.advance)(b)
    assert_same(a.sampler, b.sampler)


def test_long_horizon_gain_drift():
    """After 10,000 steps, float64 gain agrees with a fresh linear solve."""
    experiment, initial = composition(block=10_000, gamma=0.999)
    state = jax.jit(experiment.advance)(initial)
    model = np.asarray(state.learner.model)
    gain = np.asarray(state.learner.gain)
    matrix = np.eye(6) - 0.999 * model
    reference = np.linalg.solve(matrix, np.eye(6))
    assert np.isfinite(state.learner.q).all()
    np.testing.assert_allclose(gain, reference, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(matrix @ gain, np.eye(6), rtol=0, atol=1e-9)


def test_checkpoint_disallows_nonarray_state(tmp_path):
    with pytest.raises(ValueError, match="array"):
        checkpoint.save(tmp_path / "bad.npz", (1, "bad"))
