# Reproducibility

## Install and run

From the repository root, install the locked Python 3.13 environment with [uv](https://docs.astral.sh/uv/).
The lock pins both third-party packages and the Git revisions of `jaxtor` and `jaxdp`; local checkouts are not substituted.

```bash
uv sync --locked
uv run --locked python -m scripts.run --help
uv run --locked python -m scripts.run --method greq --output runs/pilot-greq
uv run --locked python -m scripts.run --method ql --output runs/pilot-ql
```

These commands run 10,000-sample **pilots**, not tuned paper benchmarks.
A `completed` run status records finite execution, not convergence or competitive performance.
Use a new output directory for each run. Existing directories are never overwritten.
The sample budget must be divisible by `--eval-every`.

## Organization

| Path | Contents |
| --- | --- |
| `src/greq/` | Learning, environment, sampling, evaluation, and checkpoint components |
| `scripts/` | Experiment entry points and artifact-writing utilities |
| `tests/` | Numerical oracles, transformation tests, and experiment checks |
| `runs/` | Generated CSVs, figures, checkpoints, and other artifacts; ignored by Git |

Each run writes:

| Artifact | Contents |
| --- | --- |
| `config.json` | Resolved method settings, sample budget, seeds, and output path |
| `provenance.json` | Python, backend, device, package versions, dependency revisions, and source hashes |
| `source/` | Project source snapshot, `pyproject.toml`, `uv.lock`, and Python version |
| `environment.npz` | Exact transition, reward, initial-distribution, and terminal arrays |
| `reference.npz` | Reference Q-table certified by its Bellman residual |
| `metrics.csv` | Step-zero and periodic Bellman error, value error, selected-policy return, policy gap, and training time |
| `checkpoint.npz` | Latest sampler and learner state, including PRNG keys and local clocks |
| `status.json` | Running, completed, or failed status; completed samples and any error |

## Numerical and sampling conventions

GReQ follows Algorithm 1 of *Greedy-resolvent matrix-gain Q-learning* in the ECC manuscript.
The Q-update uses the pre-update gain $R_k$; model and gain updates then incorporate the current sample.
Local counts are incremented before computing $\alpha=c^{-\eta}$ and $\beta=c^{-\rho}$, with $1/2<\rho<\eta\leq1$.
Q-tables have shape `(A, S)` and matrices use action-major index `a * S + s`.
Greedy ties select the lowest action index. There is no projection, denominator clipping, or inverse refresh.

The runner uses float64 and explicit Threefry keys.
`--env-seed` selects the Garnet MDP; `--seed` selects independent behavior-action and environment-step streams.
Matching both seeds across methods pairs their trajectories because behavior is uniform and independent of learning.
The initial state is uniform, and evaluation boundaries do not reset or truncate the chain.
Reducible MDP instances fail explicitly rather than being silently regenerated.

Garnet transitions use `jaxdp`; rewards are separate uniform draws on `[0, 1)` for each state-action pair, constant over successor states.
This implements the manuscript's deterministic $r(s,a)$ convention rather than the dependency's transition-dependent reward default.
Reference policy iteration must certify $\|Tq-q\|_\infty/(1-\gamma)\leq10^{-8}$.
Greedy-policy return evaluates the policy selected by the current Q-table, without taking a second maximum over its evaluated action values.

Training time includes sampling and learning, with explicit device synchronization.
It excludes compilation, a discarded warmup block, evaluation, checkpointing, and other file I/O.
Compilation and warmup times are recorded separately. These timings are not end-to-end runtime measurements.
Numerical repeatability is tested on a fixed software/backend setup; bitwise agreement across hardware is not guaranteed.
The tests also check continuation through `greq.checkpoint.load(path, template)`; automatic CLI resume is not yet implemented.

## Checks

```bash
uv run --locked ruff format --check src scripts tests
uv run --locked ruff check src scripts tests
uv run --locked ty check
uv run --locked pytest -q
uv build
```

CI runs these checks and a small CLI smoke test on CPU.
The numerical tests compare GReQ against an independent NumPy direct-solve recurrence, including discounts up to `0.999`.

## Experiment scope

The current implementation includes GReQ, asynchronous Q-learning, and Garnet pilots.
It does not yet reproduce the manuscript's complete comparisons, tuning, runtime study, or figures.
The Graph benchmark needs an explicit feasible-action mapping.
The manuscript specifies 18 state-action pairs, while `jaxdp` supplies a padded 6-by-6 table.
SQL, NeSA, and the Zap variants require source-verified implementations before comparison.

Before tuning, freeze the selection score, environments, search ranges, seed partitions, and budgets in experiment configuration.
Give methods and retuned ablations equal search budgets and shared ranges, retain failed trials, and report on new paired test seeds with uncertainty intervals.
Neither the pilot defaults nor the existing manuscript settings constitute a completed fair-search protocol.
The paper's convergence result remains conditional on its stated coverage, boundedness, and convex-resolvent assumptions.
