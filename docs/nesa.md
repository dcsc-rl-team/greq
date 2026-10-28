# NeSA Q-learning

`greq.NeSA` follows the tabular Q-learning formulation in Section 4 of Devraj, Bušić, and Meyn, [*Optimal Matrix Momentum Stochastic Approximation and Applications to Q-learning*, arXiv:1809.06277v2](https://arxiv.org/abs/1809.06277v2). It consumes one continuing transition per update. There is no coverage round, frozen source table, matrix inverse, clipping, or terminal masking.

## Recurrence

Flatten the action-first Q-table into $q_t$, and let $v_t=q_t-q_{t-1}$ denote the full momentum vector. Initialize $q_{-1}=q_0$, hence $v_0=0$. For the observed transition $(s,a,r,s')$, let $i=(s,a)$ and $j=(s',\arg\max_b Q_t(s',b))$. With coordinate vectors $e_i$ and $e_j$, define

$$
\delta_t=r+\gamma Q_t(j)-Q_t(i),\qquad
A_t=e_i(\gamma e_j-e_i)^\top,\qquad
\alpha_t=(1+t/k_0)^{-\eta}.
$$

The update is

$$
v_{t+1}=(I+\zeta A_t)v_t+\zeta\alpha_t\delta_t e_i,
\qquad q_{t+1}=q_t+v_{t+1}.
$$

Only the correction to $v_t$ is sparse. The identity term retains momentum at every coordinate, and the entire Q-table advances by $v_{t+1}$. Resetting momentum outside the observed pair gives a different algorithm. The implementation uses one sparse correction and a full-vector addition, requiring $O(SA)$ storage rather than a dense gain matrix.

The TD residual is the reward-maximization counterpart of Section 4's Bellman residual. The sampled Jacobian uses the current greedy successor, as in the paper's Q-learning table. At a greedy-action switch, this need not equal the general nonlinear function-difference recursion in Equation (5). Tests compare every update against an independently constructed dense $(I+\zeta A_t)$ matrix.

## Experiments and checkpoints

A single run defaults to $\eta=k_0=\zeta=1$, giving $\alpha_t=1/(t+1)$ and the unit momentum scale in the paper's Q-learning formulation. The clock counts completed transitions, not visits to a particular pair. `--eta`, `--k0`, and `--zeta` configure a single run. The power schedules are our finite-budget choices, not a reproduction of the paper's numerical protocol.

```bash
uv run -m scripts.run --method nesa --zeta 1 --output runs/nesa
uv run -m scripts.finetune --methods nesa --output runs/nesa-study
uv run -m scripts.report --methods nesa --runtime --output runs/nesa-study
uv run -m scripts.figures --data runs/nesa-study
```

- NeSA is opt-in; default method lists and other algorithms' grids are unchanged. Include `nesa` alongside other methods with `--methods`, using a fresh output directory.
- The default grid has 24 candidates: $\eta\in\{1,0.8,0.6\}$, $k_0\in\{1,10\}$, and $\zeta\in\{1,0.75,0.5,0.25\}$. It includes the reciprocal/unit-momentum reference setting. The flags are `--nesa-etas`, `--nesa-k0s`, and `--nesa-zetas`.
- Power-schedule NeSA parameter rows are `(seed, gamma, eta, k0, zeta)`. Other methods retain four-column rows. CSVs have dedicated optional `zeta` and `t0` columns; readers also accept both older schemas. Momentum and the shifted offset are never mislabeled as `rho` or `k0`.
- Selection minimizes median final `value_linf` on seeds 0–2 and requires every selection seed to finish finitely. Ties break by ascending parameter values. Winners are frozen before evaluation on seeds 3–13. Those evaluation seeds were inspected in previous comparisons and are not a fresh test set for algorithm design.
- Learning curves use vmapped trajectories; scalar whole-run training times are measured independently. Initialization, compilation, warmup, evaluation, and I/O are excluded from scalar timing.
- Checkpoints retain the Q-table, full momentum, transition count, parameter row, and sampler state. Evaluation does not reset momentum or consume behavior samples.

## Shifted reciprocal schedule with two tuned parameters

The normalized power schedule always starts at $\alpha_0=1$, even when $k_0$ changes. To control that initial impulse without changing the momentum recurrence, explicitly supply a positive `t0`:

$$
\alpha_t=\frac{1}{t+t_0}.
$$

A single run uses `--t0`; this fixes $\eta=k_0=1$ and rejects conflicting values. Larger $t_0$ softens the initial step while retaining the reciprocal tail. Omitting `t0` preserves the original power schedule and its checkpoint format.

```bash
uv run -m scripts.run --method nesa --t0 100 --zeta .1 --output runs/nesa-shifted-single
uv run -m scripts.report --methods nesa --runtime --output runs/nesa-shifted \
  --nesa-t0s 1 10 100 1000 3000 10000 --nesa-zetas .03 .1 .3 1
uv run -m scripts.figures --data runs/nesa-shifted
```

A nonempty `--nesa-t0s` selects a grid over **only $t_0$ and $\zeta$**. The normalized-power exponent and scale grids are inactive in this mode. Six offsets and four momentum multipliers give 24 candidates per environment and discount, including $(t_0,\zeta)=(1,1)$. With three selection seeds, two environments, and four discounts, there are 576 selection trials. NeSA's matrix-free update, zero initialization, continuing trajectory, and selection criterion are unchanged.

Shifted rows are `(seed, gamma, 1, 1, zeta, t0)`. The two ones are fixed compatibility fields, not search dimensions. CSVs record `eta=1`, the actual `t0` and `zeta`, and blank `k0`/`rho`. Requests explicitly record the two tuned parameters. The offset is not a reinterpretation of `k0`.

Keep this study separate from the original power-schedule study. The new study has the same candidate count, but it adds to cumulative exploration; it does not erase the first search. Freeze winners using the original 10,000-sample criterion on seeds 0–2. Longer-horizon diagnostics can continue those winning checkpoints without adding configurations or reselecting. Evaluation seeds 3–13 have already been inspected, and cannot be treated as fresh confirmation of the revised schedule design.

## Scope

This is not the archived coordinate-only adaptation. It is matrix-free NeSA, not PolSA or an ordinary Nesterov optimizer applied to squared TD error. The paper's covariance analysis imposes linear-model and noise assumptions that do not establish a finite-budget guarantee for our nonlinear Markovian Q-learning runs. NeSA does not generally have Zap's optimal asymptotic covariance. Performance under the project's sample budget must be measured rather than inferred from the method's name.
