# Dzanga-Sangha PPO microphone deployment optimization

Uses the supplied `dzanga_sim(2).py` unchanged, renamed to `dzanga_sim.py`.

## Installation

Python 3.10+ recommended. In this directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick check

```bash
python dzanga_optimize.py smoke --output results/smoke
```

## Full baseline matrix

```bash
python dzanga_optimize.py baseline --sim-seconds 86400 --dt 1 --seeds 20 --output results
python dzanga_optimize.py plot --output results
```

There are **31 microphone counts**: 10, 15, ..., 160. The baseline compares
**8 layout families**, one canonical spacing parameter per family: 248 cases
per seed, or 4,960 episodes with 20 seeds. This can take a very long time.
Use `--max-cases 8 --seeds 1 --sim-seconds 120` for a brief exploratory test.

## PPO training and held-out evaluation

```bash
python dzanga_optimize.py train --timesteps 4096 --sim-seconds 86400 --dt 1 --output results
python dzanga_optimize.py evaluate --model results/ppo_policy.zip --seeds 50 --sim-seconds 86400 --dt 1 --output results
```

PPO selects `(microphone_count, layout_family, spacing_level)` with a
`MultiDiscrete([31,8,5])` policy. Candidate counts are 10 to 160 inclusive in
steps of 5. The layout families are four original strategies (projected onto
valid unique sites), random uniform, regular rectangular grid, jittered grid,
and grid/random mixture. The five spacing levels are 0, 0.2, 0.4, 0.65, 0.9.
Regular grid snaps ideal lattice positions onto allowable park sites.
Grid/random mix uses spacing level to control the fraction of random sites.

Outputs: `baseline_runs.csv`, `baseline_summary.csv`, baseline plot,
`ppo_policy.zip`, `chosen_configuration.json`, `ppo_evaluation.csv`, and
`ppo_deployment.csv` (coordinates for one evaluation seed).

## Critical scientific notes

* The policy is a **one-step RL deployment selector**, not adaptive sensor
  repositioning during a mission. Evolutionary search or Bayesian optimization
  may be more sample-efficient; PPO optimality is not guaranteed.
* The original simulator has strong modeling assumptions and small poacher
  detection radius (200 m), so many deployments may fail to neutralize every
  poacher before the time limit. An incomplete mission incurs a large penalty.
* Baselines and PPO evaluations must use the **same horizon and timestep**;
  report completion rate, poaching, and the number caught separately.
* Stochastic layouts are regenerated per seed. Use sufficiently many independent
  held-out seeds to estimate uncertainty.
* The underlying `step()` implementation scans once per step if the simulated
  time since last scan exceeds one second; for scientific comparisons use
  `--dt 1` (not 5 or 10) so scan frequency is consistent.
* PPO compares 1,240 discrete combinations (31 counts x 8 families x 5 levels).
  The baseline matrix only covers 248 count/family pairs at the middle level;
  training explores other levels. A one-parameter-family grid search would
  be needed to exhaustively enumerate all 1,240 cases.
* The `uniform_original` etc. are slightly modified for fair comparison:
  their proposed points are snapped to a common in-park candidate-site grid.

This package contains working source but **does not include a trained PPO policy**.
