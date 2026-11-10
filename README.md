# RL Coverage Path Planning

Code implementation of the conference paper [__Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning, ICML, 2024__](https://proceedings.mlr.press/v235/jonnarth24a.html), and its journal extension [__Sim-to-Real Transfer of Deep Reinforcement Learning Agents for Online Coverage Path Planning, IEEE Access, 2025__](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11039829), extended to multiple robots (MAPPO/IPPO in JAX).

Lawn mowing | Exploration
:-------------------------:|:-------------------------:
<img src="figures/mowing_path.png" alt="mowing path" height="300" /> | <img src="figures/exploration_path.png" alt="exploration path" height="300" />

WITHOUT total variation reward | WITH total variation reward
:-------------------------:|:-------------------------:
<img src="figures/mowing_no_tv.png" alt="without total variation reward" width="300" /> | <img src="figures/mowing_tv.png" alt="with total variation reward" width="300" />

## Repository layout

```
rlm/                    single-agent library (original paper: gym env, SB3 networks)
marl/                   multi-agent library (JAX env, MAPPO/IPPO, statistics)
scripts/
  single_agent/         train, eval, plot, show_path, make_giant_maps, corridor_test
  multi_agent/          train, eval, play, collect, benchmark, init_from_sb3,
                        make_test_maps, corridor_test
  analysis/             dashboard, dashboard_compare, paired_stats
pipelines/              end-to-end shell workflows (training baselines, comparisons)
tests/                  pytest suite (single- and multi-agent)
maps/                   training, evaluation, held-out test and corridor maps
weights/                pre-trained single-agent models
results/                comparison results (PDFs + the CSV data they show)
experiments/            training runs (not tracked)
```

Every script runs from the repository root as a module, e.g. `python -m scripts.multi_agent.train --help`.

## Install

The single-agent and multi-agent code have separate environments.

Single-agent (Python 3.9, gym, torch, stable-baselines3):
* [Install miniconda](https://docs.conda.io/en/latest/miniconda.html), then `conda create -n rlm python=3.9 && conda activate rlm`
* `pip install setuptools==65.5.0 pip==21 && pip install wheel==0.38.0`
* (CPU) `pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`
* (GPU) `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116`

Multi-agent (Python 3.11-3.13, JAX; does not use `rlm/`):
* `python3 -m venv .venv-marl && source .venv-marl/bin/activate`
* `pip install -U pip && pip install -r requirements-marl.txt`
* Check that JAX sees the GPU: `python -c "import jax; print(jax.devices())"` should list a `CudaDevice`. The `jax[cuda12]` wheels include CUDA and cuDNN, only the NVIDIA driver is needed (>= 525; with >= 580 `jax[cuda13]` can be used instead).

Tests: `pytest` in either environment (tests whose dependencies are not installed are skipped).

## Single agent

Train (reduce `--buffer_size` if memory is limited):
* Mowing: `python -m scripts.single_agent.train --logdir my-experiment`
* Exploration: `python -m scripts.single_agent.train --logdir my-experiment --exploration --local_tv_reward_scale 0.2 --no-overlap_observation --no-steering_limits_lin_vel`

Inspect:
* Plot logs: `python -m scripts.single_agent.plot --load my-experiment`
* Render: `python -m scripts.single_agent.eval --load my-experiment` (`--render_mode full` also shows the multi-scale observations). Keys: `c` multi-scale crops, `m` composite / separate scales, `s` simulation speed (x1, x2, x4, x8, max), `q`/`Esc` quit.
* Full evaluation: `python -m scripts.single_agent.eval --load my-experiment --verbose --no-render --metrics_dir metrics`, then `python -m scripts.single_agent.show_path --metrics_dir my-experiment/metrics --type eval --episode 1`

Maps larger than the largest scale of the multi-scale maps (76.8 m by default):
* `python -m scripts.single_agent.make_giant_maps --task exploration --method tile --size 108 --num 5` (`tile`: mosaic of the evaluation maps, `random`: the training map generator)
* `python -m scripts.single_agent.eval --load my-experiment --maps "maps/giant_exploration_tile_*.png" --render_mode full`

Corridor test (a corridor longer than the largest scale, results in `results/corridor/`): `python -m scripts.single_agent.corridor_test --load weights/mowing_tv1`

### Pre-trained weights

In `weights/` (also at [this link](https://drive.google.com/file/d/1hJXxRQTaFd0MEuB2UuQT9D6I-p_QXiAM/view?usp=sharing)):
* __exploration__: exploration CPP model configured for Explore-Bench.
* __mowing_tv1__: lawn mowing, incremental TV reward of 1.
* __mowing_tv2__: same with TV reward 2: slightly nicer patterns, slower coverage on average.

Show the recorded paths: `python -m scripts.single_agent.show_path --metrics_dir weights/mowing_tv1/metrics --type eval --episode 1`

## Multi agent (MAPPO / IPPO, JAX)

The `marl` package extends the mowing task to N robots, trained with MAPPO (centralized training, decentralized execution). Environment, networks and training all run in JAX, so thousands of environments are simulated and trained in parallel on a GPU.

* All robots share one global map (perfect communication). Each step all robots move simultaneously, the shared coverage and known obstacle maps are updated with the contributions of all robots, and each robot extracts its own egocentric multi-scale observation from the shared maps.
* The observation of each robot is the single-agent one (coverage, known obstacles, frontier, lidar) plus a multi-scale map of the other robots' positions. Robots are dynamic obstacles: seen by the lidar, collisions between robots are penalized, not added to the obstacle map (unless `--robot_lidar_obstacles`).
* The actor only uses the robot's own observation and its parameters are shared between robots, so a policy runs with any number of robots. The centralized critic also sees an agent-specific global state: global maps of the coverage, the true and known obstacles and all robots, the robot's own position, the team coverage and elapsed time. Critic options: `--critic_state agent_specific` (also the robot's own local observation), `--global_crop` (64x64 global maps of the map itself instead of the whole canvas: a 4.8 m map gets 7.5 cm instead of 19 cm cells, positions relative to the map, map side as an input), `--critic_gather` (features of the global maps sampled at each robot's position, so that V(s, i) sees the map around robot i). The explained variance of the critic is logged as `explained_variance`.
* IPPO (`--algo ippo`): local critic V(o_i) with the same inputs as the actor; everything else unchanged, so the comparison with MAPPO isolates the centralized critic.
* Limited communication (`--comm_range 5`, meters; 0 = unlimited): each robot keeps its own belief of the coverage and obstacle maps; connected robots (within range, directly or through others) merge beliefs every step. Rewards, metrics and the critic still use the true global maps. `--no_communication` disables it entirely.
* Reward: the single-agent reward per robot, where the area term mixes the robot's own and the team's newly covered area divided by N (`--team_reward_weight`, 0 = individual, 1 = team). Area covered by k robots in the same step is split 1/k. The incremental total variation term uses each robot's own contribution.
* Cooperative terms (off by default), computed from the true state also with limited communication (centralized training):
  * `--workload_size 64`: every step the cells of a 64x64 grid over the map are assigned to the robot with the shortest path to them around the true obstacles (geodesic Voronoi partition); the workload of a robot is the free area not yet covered in its cells. Costs about 12% of the training speed.
  * `--workload_reward_scale` (needs `--workload_size`): team reward for balancing the workloads, potential-based shaping with phi = -(maximum workload), `--shaping_gamma` must equal `--gamma`.
  * `--critic_workload` (needs `--workload_size`): privileged inputs of the centralized critic, from the true state: each robot's workload and its share of the team's, the maximum workload (the potential of the workload reward), and with limited communication the team coverage the robot does not know and the fraction of robots it is connected to. Recipes `mappo_workload` (workload 0.02 and critic_workload), `mappo_workload_nc` (workload only) and `ippo_workload` in `pipelines/train_from_mappo.sh`.
  * `--min_agents 2 --num_agents 8`: variable number of robots, at every reset 2, ..., 8 robots (uniform); `num_agents` is the number of slots of the arrays, the parked robots interact with nothing and the losses ignore them. Simulates 8 robots in every episode: about half the training speed of 4 robots. Recipe `mappo_crash_var` (the recipe of `mappo_crash_long`: workload 0.02, collisions end the episode). The tests always run exactly `--num_agents` robots.
  * Every term of the reward is logged per episode as `ep_r_<term>` in `progress.csv`. `python -m scripts.multi_agent.reward_terms --load <checkpoint>` measures all terms for a trained policy (new terms at weight 1, no training), to choose their weights.
* Maps of different sizes are placed on a fixed canvas (`--canvas_size`, 320 pixels for the training maps). The training levels follow `MowerEnv._set_level`; random maps come from a bank regenerated periodically.

Every field of `marl/config.py` is a command line argument (booleans as `--flag` / `--no-flag`). Logs go to `<logdir>/progress.csv`, checkpoints to `<logdir>/checkpoint` and `<logdir>/checkpoint_best`.

### Train

* From scratch: `python -m scripts.multi_agent.train --logdir experiments/mappo_4 --num_agents 4` (`--algo ippo`, `--comm_range 5`, ...)
* From the trained single-agent policy (how all runs in `results/` were trained):
  1. `python -m scripts.multi_agent.init_from_sb3 --load weights/mowing_tv1 --out experiments/sb3_init/checkpoint --lidar_fov 180` converts its actor (read without torch) into a MARL checkpoint. The pose branch of the other robots starts at zero, so the initial policy is the single-agent one; the critic is new.
  2. `python -m scripts.multi_agent.train --logdir experiments/mappo_sb3_v2 --checkpoint experiments/sb3_init/checkpoint --lidar_fov 180 --critic_warmup 20` (`--critic_warmup`: first iterations only train the critic).
* Baselines trained exactly as `experiments/mappo_sb3_v2` (same settings via `--config_from`, initialization and budget), changing only one thing: `pipelines/train_from_mappo.sh <recipe>` with `ippo_sb3_v2` (local critic), `sa_indep` (independent robots, no communication), `mappo_obst` / `ippo_obst` (other robots in the obstacle map), and the critic variants `mappo_rerun` (unchanged, logs the explained variance), `mappo_agent`, `mappo_crop`, `mappo_gather`, `mappo_agent_gather`.
* Pick `--num_envs` with `python -m scripts.multi_agent.benchmark --num_envs 256 1024 4096`. The rollout stores `num_envs * num_steps * num_agents` uint8 observations of 32x32x16 bytes plus one 64x64x4 global map per environment and step: with 4 robots and `--num_steps 128`, about 2.5 GB + 10.6 MB per environment. Gradients use minibatches of `--minibatch_size` agent samples (default 4096).
* `--total_steps` counts environment steps (each one is `num_agents` agent steps).

Long jobs on the lab machine: `systemd-run --user --unit=<name> --working-directory=$PWD <command>` (survives logout), with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (set by default by the train script and the pipelines).

### Evaluate and watch

* Test rules, always on in every test (`eval`, `collect`, `play`, `corridor_test`, via `EnvConfig.for_testing`): the first collision of any robot (wall, obstacle or robot) ends the episode, so every episode ends in exactly one of success, collision or timeout and the three rates add up to 1. Training keeps its own rules (`--collision_ends_episode`, off by default), `--min_agents`). With a checkpoint trained with a variable number of robots, pass `--num_agents` (default: the maximum).
* Quick evaluation on the mowing evaluation maps: `python -m scripts.multi_agent.eval --load experiments/mappo_sb3_v2/checkpoint_best` (`--num_agents 8`, `--maps "maps/giant_mowing_*.png"`). Writes `metrics.csv` and path images to `<checkpoint>/eval`.
* Live pygame viewer: `python -m scripts.multi_agent.play --load experiments/mappo_sb3_v2/checkpoint_best --num_agents 8` (`JAX_PLATFORMS=cpu` runs it without the GPU). Keys: space pause, s/a speed, n/p map, tab robot, b belief of the selected robot, c multi-scale crops, m separate scales, l lidar, t paths, q quit.
* Corridor test: `python -m scripts.multi_agent.corridor_test --load experiments/mappo_sb3_v2/checkpoint_best`
* Held-out test maps (already in `maps/`): `python -m scripts.multi_agent.make_test_maps` generates 100 small (2.4-7.5 m, training sizes) and 100 large (12-24 m) random maps with a seed not used in training.

### Compare methods

`scripts.multi_agent.collect` runs the same episodes (same maps, start poses, seed, deterministic actions) for every policy and writes one row per episode to `episodes.csv` and coverage curves to `curves.csv`. A run is `NAME=CHECKPOINT_GLOB[,env_option=value,...]`, e.g. `--run "MAPPO + flip=experiments/mappo_sb3_v2/checkpoint_best,flip_when_stuck=true"`; methods collected again replace their old rows. The analysis scripts then draw from that data:
* `scripts.analysis.dashboard`: two pages per map set (overview, statistical tests of every pair, across seeds and maps, see `marl/stats.py`).
* `scripts.analysis.dashboard_compare`: one page, one column per map set, success / collision / timeout rates, time to the goal, team efficiency, overlap, collisions, imbalance and coverage over time against the number of robots, plus a summary table.
* `scripts.analysis.paired_stats`: paired tests over maps (bootstrap CI, Wilcoxon, Holm) between two policies.

Every PDF writes the values it shows to `<pdf name>_data/` next to it.

The pipelines run complete comparisons on the three map sets (`test_small`, `test_large`, `paper`; pass map set names to run only some):

| Pipeline | Compares | Results |
| --- | --- | --- |
| `pipelines/compare_sa.sh` | single-agent policy replicated on N robots vs MAPPO vs IPPO, with and without `flip_when_stuck`, plus the single-agent policy fine-tuned alone | `results/compare_sa/` |
| `pipelines/compare_obst.sh` | the same under strict rules (robots in the obstacle map, the first collision ends the episode), plus independent robots and retrained MAPPO/IPPO | `results/compare_obst/` |
| `pipelines/compare_scale.sh` | IPPO evaluated with scale factor 2, 3 and 4 of the multi-scale maps | `results/compare_scale/` |

### Notes

* Not implemented in the multi-agent version: stacked observations, action observations, acceleration limits, action delay, the frontier and global total variation rewards (all off by default for mowing), and the exploration task.
* Truncated episodes (time limit or no new coverage) are treated as terminal in the advantage estimation.
* PPO uses KL early stopping: the actor stops training within an update once the approximate KL divergence exceeds 1.5 x `--target_kl` (default 0.02, 0 disables it); the fraction of minibatches with an actor step is logged as `actor_updates`.
* Per-scale (grouped) convolutions use a block-diagonal kernel, since grouped convolutions are very slow to differentiate on some XLA backends.
* On CPU, XLA runs convolutions inside the training loop slowly: CPU training is only useful for debugging and tests.

## Cite

Conference paper:
```
@InProceedings{jonnarth2024icml,
  title = {Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning},
  author = {Jonnarth, Arvi and Zhao, Jie and Felsberg, Michael},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series = {Proceedings of Machine Learning Research},
  year = {2024},
  volume = {235},
  pages = {22491--22508},
}
```

Journal extension:
```
@article{jonnarth2025access,
  title = {Sim-to-Real Transfer of Deep Reinforcement Learning Agents for Online Coverage Path Planning},
  author = {Jonnarth, Arvi and Johansson, Ola and Zhao, Jie and Felsberg, Michael},
  journal = {IEEE Access},
  year = {2025},
  volume = {13},
  pages = {106883-106905},
}
```
