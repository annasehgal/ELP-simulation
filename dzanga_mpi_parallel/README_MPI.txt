Dzanga MPI Parallel Version
===========================

Files
-----
dzanga_sim_mpi.py        Simulation model (same simulation behavior; renamed for clean import)
dzanga_optimize_mpi.py   MPI-enabled Monte Carlo baseline, PPO training, and evaluation
requirements_mpi.txt     Python dependencies including mpi4py

Install
-------
You need an MPI implementation plus Python packages.

Ubuntu/Debian example:
  sudo apt install openmpi-bin libopenmpi-dev
  python3 -m pip install -r requirements_mpi.txt

Windows: install Microsoft MPI (MS-MPI) or another mpi4py-compatible MPI implementation, then:
  py -m pip install -r requirements_mpi.txt

Examples
--------
Serial compatibility:
  python3 dzanga_optimize_mpi.py baseline --sim-seconds 86400 --dt 1 --seeds 20 --output results

8-rank Monte Carlo baseline:
  mpiexec -n 8 python3 dzanga_optimize_mpi.py baseline --sim-seconds 86400 --dt 1 --seeds 20 --output results

8-rank PPO training:
  mpiexec -n 8 python3 dzanga_optimize_mpi.py train --sim-seconds 86400 --dt 1 --timesteps 4096 --output results

8-rank held-out PPO evaluation:
  mpiexec -n 8 python3 dzanga_optimize_mpi.py evaluate --sim-seconds 86400 --dt 1 --seeds 100 --model results/ppo_policy.zip --output results

Plot (one process is sufficient):
  python3 dzanga_optimize_mpi.py plot --output results

How parallelism works
---------------------
Baseline/evaluate: independent (configuration, seed) Monte Carlo jobs are divided round-robin across MPI ranks. Rank 0 gathers rows, sorts them, and writes the same consolidated CSV outputs.

PPO: rank 0 owns the Stable-Baselines3 PPO model and optimizer. Every MPI rank supplies one independent Gymnasium environment. Rank 0 broadcasts actions to worker ranks and gathers rewards/transitions through an SB3 VecEnv. This parallelizes the expensive simulation rollouts without running conflicting PPO optimizers.

Timesteps note
--------------
Stable-Baselines3 counts total transitions across all vector environments. With N MPI ranks, each PPO vector step produces N transitions. Thus --timesteps remains an aggregate training-sample target, rather than a per-core target.

Reproducibility
---------------
Monte Carlo baseline seeds remain cfg.seed + repetition, preserving the original comparison semantics. PPO worker environments receive separate reset streams. Exact PPO learning curves can differ from a one-environment run because vectorized rollout ordering/batching changes.
