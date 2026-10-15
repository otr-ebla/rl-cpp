# RL Coverage Path Planning

Code implementation of the conference paper [__Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning, ICML, 2024__](https://proceedings.mlr.press/v235/jonnarth24a.html), and its journal extension [__Sim-to-Real Transfer of Deep Reinforcement Learning Agents for Online Coverage Path Planning, IEEE Access, 2025__](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11039829).

Lawn mowing | Exploration
:-------------------------:|:-------------------------:
<img src="figures/mowing_path.png" alt="mowing path" height="300" /> | <img src="figures/exploration_path.png" alt="exploration path" height="300" />

WITHOUT total variation reward | WITH total variation reward
:-------------------------:|:-------------------------:
<img src="figures/mowing_no_tv.png" alt="without total variation reward" width="300" /> | <img src="figures/mowing_tv.png" alt="with total variation reward" width="300" />

## Install

* [Install miniconda](https://docs.conda.io/en/latest/miniconda.html)
* `conda create -n rlm python=3.9`
* `conda activate rlm`
* `pip install setuptools==65.5.0 pip==21`
* `pip install wheel==0.38.0`
* (CPU) `pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`
* (GPU) `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116`

## Run

Train an RL agent for CPP in a simulated 2D environment (reduce `--buffer_size` if memory is limited):
* Mowing: `python train.py --logdir my-experiment`
* Exploration: `python train.py --logdir my-experiment --exploration --local_tv_reward_scale 0.2 --no-overlap_observation --no-steering_limits_lin_vel`

Check how the trained agent performs:
* Plot logs: `python plot.py --load my-experiment`
* Render: `python eval.py --load my-experiment` (add `--render_mode full` to also show the multi-scale observations)

Rendering uses pygame. Keyboard controls in the render window:
* `c`: show/hide the multi-scale map crops, centered on and rotated with the agent (the thick edge is the forward direction), also outlined in the observation panels
* `m`: switch the observation panels between the composite view and each scale shown separately, as received by the network (with `--render_mode full`)
* `s`: cycle the simulation speed: x1 (real time), x2, x4, x8, max
* `q`/`Esc`: quit

Evaluate on maps larger than the region covered by the largest scale of the multi-scale maps (76.8 m with the default parameters):
* Generate maps: `python make_giant_maps.py --task exploration --method tile --size 108 --num 5` (`--method tile` is a mosaic of the evaluation maps, `--method random` uses the generator of the training maps)
* Evaluate: `python eval.py --load my-experiment --maps "maps/giant_exploration_tile_*.png" --render_mode full`

Run full quantitative evaluation and show the resulting paths and metrics:
* Run eval: `python eval.py --load my-experiment --verbose --no-render --metrics_dir metrics`
* Show path: `python show_path.py --metrics_dir my-experiment/metrics --type eval --episode 1`

## Pre-trained weights

Pre-trained weights can be found at [this link](https://drive.google.com/file/d/1hJXxRQTaFd0MEuB2UuQT9D6I-p_QXiAM/view?usp=sharing). The zip file contains three models:
* __exploration__: An exploration CPP model configured for Explore-Bench.
* __mowing_tv1__: A CPP model for the lawn mowing task, with an incremental TV reward of 1.
* __mowing_tv2__: Same as _mowing_tv1_, but with TV reward = 2. Makes slightly nicer looking patterns, but has slower coverage times on average.

Each folder contains a metrics folder. Visualize the paths like this:
* `python show_path.py --metrics_dir weights/mowing_tv1/metrics --type eval --episode 1`

## Tests

Run tests:
* Run `pytest` from the root folder.

## Multi-agent lawn mowing (MAPPO, JAX)

The `marl` package extends the mowing task to N robots, trained with MAPPO (centralized training, decentralized execution). The environment, the networks and the training all run in JAX, so thousands of environments can be simulated and trained in parallel on a GPU.

* All robots share one global map (perfect communication). Each step all robots move simultaneously, the shared coverage and known obstacle maps are updated with the contributions of all robots, and each robot extracts its own egocentric multi-scale observation from the shared maps.
* The observation of each robot is the same as in the single-agent case (coverage, known obstacles, frontier, lidar), plus a multi-scale map of the other robots' positions. Robots are dynamic obstacles: they are seen by the lidar and collisions between robots are penalized, but they are not added to the obstacle map.
* The actor only uses the robot's own observation, and its parameters are shared between robots, so a trained policy can be run with any number of robots. The centralized critic also sees an agent-specific global state: global maps of the coverage, the true and the known obstacles and all robots, the robot's own position, and the team coverage and elapsed time.
* Reward: the single-agent reward per robot, where the area term mixes the robot's own newly covered area and the team's newly covered area divided by N (`--team_reward_weight`, 0 = individual, 1 = team). Area covered by k robots in the same step is split 1/k between them. The incremental total variation term is computed on each robot's own contribution.
* Maps of different sizes are placed on a fixed canvas (`--canvas_size`, 320 pixels for the training maps). The training levels follow `MowerEnv._set_level` (fixed training maps, then random maps); random maps are drawn from a bank that is regenerated periodically.

Install: the multi-agent code does not use the single-agent code (`rlm/`) or its old dependencies (Python 3.9, gym, torch), so it has its own environment with a modern Python (3.11-3.13 tested) and `requirements-marl.txt`. On Ubuntu with an NVIDIA GPU:
* `python3 -m venv .venv-marl && source .venv-marl/bin/activate`
* `pip install -U pip && pip install -r requirements-marl.txt`
* Check that JAX sees the GPU: `python -c "import jax; print(jax.devices())"` should list a `CudaDevice`. The `jax[cuda12]` wheels include CUDA and cuDNN, only the NVIDIA driver is needed (driver >= 525 for CUDA 12; with a driver >= 580 `jax[cuda13]` can be used instead).
* Run the tests: `pytest` (tests of the single-agent code are skipped when their dependencies are not installed)
* Measure the speed on the GPU and choose `--num_envs`: `python benchmark_marl.py --num_envs 256 1024 4096`

Train and evaluate:
* Train with 4 robots: `python train_marl.py --logdir experiments/marl_4 --num_agents 4`
* Use as many parallel environments as the GPU memory allows (see `benchmark_marl.py`). The rollout stores `num_envs * num_steps * num_agents` uint8 observations of 32x32x16 bytes, plus one 64x64x4 global map per environment and step: with 4 robots and `--num_steps 128`, an update needs about 2.5 GB + 10.6 MB per environment (about 5 GB for 256 environments, 13 GB for 1024). Gradients are computed on minibatches of `--minibatch_size` agent samples (default 4096), independent of `--num_envs`.
* `--total_steps` counts environment steps (each one is `num_agents` agent steps). Every field of `marl/config.py` is a command line argument.
* Logs are written to `<logdir>/progress.csv`, checkpoints to `<logdir>/checkpoint`.
* Evaluate on the mowing evaluation maps: `python eval_marl.py --load experiments/marl_4/checkpoint` (add `--num_agents 8` to change the number of robots, `--maps "maps/giant_mowing_*.png"` for other maps). Writes `metrics.csv` (coverage, time to 90/95/99% coverage, overlap, collisions) and an image of the robot paths per episode to `<checkpoint>/eval`.
* Watch the robots live in a pygame window: `python play_marl.py --load experiments/marl_4/checkpoint` (same `--maps` and `--num_agents` options; keys: space pause, s/a speed, n/p map, tab robot, l lidar, t paths, q quit).

Notes:
* Not implemented in the multi-agent version: stacked observations, action observations, acceleration limits, action delay, the frontier and global total variation rewards (all off by default for mowing), and the exploration task.
* Truncated episodes (time limit or no new coverage) are treated as terminal in the advantage estimation.
* PPO uses KL early stopping: within an update, the actor stops training once the approximate KL divergence to the data-collecting policy exceeds 1.5 x `--target_kl` (default 0.02, 0 disables it); the fraction of minibatches with an actor step is logged as `actor_updates`.
* The per-scale (grouped) convolutions are implemented with a block-diagonal kernel, since grouped convolutions are very slow to differentiate on some XLA backends.
* On CPU, XLA runs convolutions inside the training loop slowly, so CPU training is only useful for debugging and tests.

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
