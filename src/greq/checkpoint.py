"""Atomic, pickle-free checkpoints for array-only state pytrees."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class _Payload:
    """Host-side checkpoint arrays and their structural schema."""

    arrays: dict[str, np.ndarray]
    schema: str


def _payload[S](state: S) -> _Payload:
    """Encode typed PRNG keys as portable uint32 key data."""
    leaves, structure = jax.tree.flatten(state)
    arrays = {}
    descriptions = []
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, jax.Array):
            raise ValueError("checkpoint state must contain only JAX array leaves")
        is_key = jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
        impl = str(jax.random.key_impl(leaf)) if is_key else None
        array = np.asarray(jax.random.key_data(leaf) if is_key else leaf)
        arrays[f"leaf_{index}"] = array
        descriptions.append((array.shape, str(array.dtype), impl))
    schema = json.dumps(
        {"version": 1, "structure": str(structure), "leaves": descriptions},
        sort_keys=True,
    )
    return _Payload(arrays=arrays, schema=schema)


def save[S](path: Path, state: S) -> None:
    """Atomically save every state leaf, including sampler RNG and MDP arrays."""
    payload = _payload(state)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(
            file, allow_pickle=False, schema=payload.schema, **payload.arrays
        )
    temporary.replace(path)


def load[S](path: Path, template: S) -> S:
    """Restore a state using an independently constructed, schema-matching template.

    The caller must match the component settings to the saved run config.
    These settings are supplied separately from the dynamic state.
    """
    expected = _payload(template)
    template_leaves, structure = jax.tree.flatten(template)
    restored = []
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"schema", *expected.arrays}:
            raise ValueError("checkpoint leaf names do not match the template")
        if str(data["schema"].item()) != expected.schema:
            raise ValueError("checkpoint schema does not match the template")
        for index, template_leaf in enumerate(template_leaves):
            stored = data[f"leaf_{index}"]
            expected_array = expected.arrays[f"leaf_{index}"]
            if (
                stored.shape != expected_array.shape
                or stored.dtype != expected_array.dtype
            ):
                raise ValueError("checkpoint leaf does not match its declared schema")
            leaf = jnp.asarray(stored)
            if jax.dtypes.issubdtype(template_leaf.dtype, jax.dtypes.prng_key):
                leaf = jax.random.wrap_key_data(
                    leaf, impl=str(jax.random.key_impl(template_leaf))
                )
            restored.append(leaf)
    return jax.tree.unflatten(structure, restored)
