# Greedy-resolvent matrix-gain Q-learning

Official implementation of GreQ that fully reproduce the results and the figures.

## Installation

We recommend installation via [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked
```

Source distributions include the library, experiment scripts, tests, documentation, and dependency lock. Local `runs/` and `research/` contents and generated documentation HTML are excluded, including when rebuilding outside a Git checkout.

## Experiments

```bash
uv run -m scripts.run       # Run one algorithm
uv run -m scripts.finetune  # Tune hyperparameters
uv run -m scripts.report    # Evaluate selected settings
```

Finetuning creates `runs/results/` with results in `tuning.csv`. The report saves the selected settings' evaluation results in `evaluation.csv`.

[NeSA](docs/nesa.md) is available as an optional, per-transition momentum baseline:

```bash
uv run -m scripts.run --method nesa --output runs/nesa
uv run -m scripts.report --methods nesa --runtime --output runs/nesa-study
uv run -m scripts.figures --data runs/nesa-study
```

NeSA retains the full momentum vector and does not wait for state-action coverage. Its 24-setting search includes the reciprocal-step, unit-momentum reference setting and tunes decay, scale, and momentum with `--nesa-etas`, `--nesa-k0s`, and `--nesa-zetas`. Default comparisons remain GReQ, QL, Zap, and batched Zap. Include `nesa` alongside those methods in `--methods` to compare them, using a fresh output directory.

**Render**: To render the results run the `figures` script:

```bash
uv run -m scripts.figures   # Generate figures
```

The script will produce `runs/results/figures.html`, which you can open in a browser.
This file contains interactive figures and a control panel for exploring the results.

![Learning curves for GReQ, QL, Zap, and batched Zap](docs/example.svg)

Above is an example of Garnet MDP with $\gamma = 0.99$.
Curves show median value error over eleven trajectory seeds and shaded bands span the 25th–75th percentiles.

## Citation


```bibtex
@misc{sunil_greq,
  title  = {Greedy-resolvent matrix-gain {Q}-learning},
  author = {Sunil, Mukund and Ok, Tolga and Sharifi Kolarijani, Mohamad Amin},
  year   = {2026},
  url    = {PAPER_URL}
}
```
