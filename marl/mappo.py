"""
MAPPO (multi-agent PPO with a centralized critic) in JAX.

One call of the jitted update function collects num_steps steps in num_envs
parallel environments and then runs update_epochs epochs of PPO on the
collected data, entirely on the device. The Python loop around it (train())
only handles logging, the curriculum, regeneration of the random maps and
checkpoints.

Actor and critic parameters are shared between the robots. Each robot gets
its own reward, advantage and value V(s, i) (agent-specific global state).
"""
import csv
import json
import os
import time
from functools import partial
from typing import NamedTuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from marl.config import EnvConfig, TrainConfig
from marl.curriculum import Curriculum
from marl.env import EnvParams, MultiMowerEnv
from marl.maps import RANDOM_GROUP, build_bank, concat_banks, load_map, random_map, train_map_files
from marl.networks import Actor, Critic


class Transition(NamedTuple):
    maps: jnp.ndarray       # T x E x N x C x H x W uint8 (channels first, see _rollout)
    lidar: jnp.ndarray      # T x E x N x R
    global_maps: jnp.ndarray  # T x E x G x G x 4 uint8
    agent: jnp.ndarray      # T x E x N x F
    action: jnp.ndarray     # T x E x N x A
    logp: jnp.ndarray       # T x E x N
    value: jnp.ndarray      # T x E x N (normalized if value normalization is used)
    reward: jnp.ndarray     # T x E x N
    done: jnp.ndarray       # T x E


@flax.struct.dataclass
class ValueNorm:
    """Running mean/std of the value targets with debiasing, as in MAPPO"""
    mean: jnp.ndarray
    mean_sq: jnp.ndarray
    debias: jnp.ndarray
    beta: float = flax.struct.field(pytree_node=False, default=0.995)

    @classmethod
    def create(cls):
        return cls(jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))

    def stats(self):
        d = jnp.maximum(self.debias, 1e-5)
        mean = self.mean / d
        var = jnp.maximum(self.mean_sq / d - mean ** 2, 1e-2)
        return mean, jnp.sqrt(var)

    def update(self, x):
        b = self.beta
        return self.replace(
            mean=b * self.mean + (1 - b) * jnp.mean(x),
            mean_sq=b * self.mean_sq + (1 - b) * jnp.mean(x ** 2),
            debias=b * self.debias + (1 - b))

    def normalize(self, x):
        mean, std = self.stats()
        return (x - mean) / std

    def denormalize(self, x):
        mean, std = self.stats()
        return x * std + mean


class RunnerState(NamedTuple):
    actor: TrainState
    critic: TrainState
    value_norm: ValueNorm
    env_state: object
    obs: dict
    key: jnp.ndarray


def gaussian_log_prob(x, mean, log_std):
    return jnp.sum(-0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * (1 + jnp.log(2 * jnp.pi)), axis=-1)


def to_channels_first(maps):
    """(..., H, W, C) -> (..., C, H, W)"""
    return jnp.moveaxis(maps, -1, -3)


def to_channels_last(maps):
    """(..., C, H, W) -> (..., H, W, C)"""
    return jnp.moveaxis(maps, -3, -1)


def huber(x, delta):
    a = jnp.abs(x)
    return jnp.where(a <= delta, 0.5 * x ** 2, delta * (a - 0.5 * delta))


def make_networks(env_cfg: EnvConfig, train_cfg: TrainConfig):
    actor = Actor(env_cfg.action_dim, env_cfg.num_maps, train_cfg.hidden_dims,
                  train_cfg.grouped_convs, train_cfg.init_log_std, train_cfg.squash_mean, train_cfg.min_log_std)
    critic = Critic(env_cfg.num_maps, train_cfg.hidden_dims, train_cfg.grouped_convs)
    return actor, critic


def init_params(env_cfg: EnvConfig, actor, critic, key):
    """Initial actor and critic parameters"""
    H = env_cfg.input_size
    G = env_cfg.global_size
    maps = jnp.zeros((1, 1, H, H, env_cfg.obs_channels), dtype=jnp.uint8)
    lidar = jnp.zeros((1, 1, env_cfg.lidar_rays))
    global_maps = jnp.zeros((1, G, G, env_cfg.global_channels), dtype=jnp.uint8)
    agent = jnp.zeros((1, 1, env_cfg.agent_features))
    k_actor, k_critic = jax.random.split(key)
    return actor.init(k_actor, maps[0], lidar[0]), critic.init(k_critic, maps, lidar, global_maps, agent)


class MAPPO:

    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig):
        self.env_cfg = env_cfg
        self.cfg = train_cfg
        self.env = MultiMowerEnv(env_cfg)
        self.actor, self.critic = make_networks(env_cfg, train_cfg)
        self.num_updates = max(1, train_cfg.total_steps // (train_cfg.num_envs * train_cfg.num_steps))
        # Minibatches hold all robots of minibatch_size / num_agents environment steps
        env_steps = train_cfg.num_envs * train_cfg.num_steps
        self.minibatch_env_steps = max(1, min(env_steps, train_cfg.minibatch_size // env_cfg.num_agents))
        self.num_minibatches = env_steps // self.minibatch_env_steps
        self.update = jax.jit(self._update, donate_argnums=0)

    # ---------------------------------------------------------------- setup

    def _optimizer(self, lr):
        cfg = self.cfg
        if cfg.anneal_lr:
            steps = self.num_updates * cfg.update_epochs * self.num_minibatches
            lr = optax.linear_schedule(lr, 0.0, steps)
        return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))

    def init(self, key, env_params: EnvParams, actor_params=None, critic_params=None):
        cfg = self.cfg
        k_init, k_env, key = jax.random.split(key, 3)
        actor_init, critic_init = init_params(self.env_cfg, self.actor, self.critic, k_init)
        actor_params = actor_init if actor_params is None else actor_params
        critic_params = critic_init if critic_params is None else critic_params
        actor = TrainState.create(apply_fn=self.actor.apply, params=actor_params,
                                  tx=self._optimizer(cfg.actor_lr))
        critic = TrainState.create(apply_fn=self.critic.apply, params=critic_params,
                                   tx=self._optimizer(cfg.critic_lr))
        reset = jax.jit(jax.vmap(self.env.reset, in_axes=(0, None)))
        obs, env_state = reset(jax.random.split(k_env, cfg.num_envs), env_params)
        return RunnerState(actor, critic, ValueNorm.create(), env_state, obs, key)

    # --------------------------------------------------------------- update

    def _rollout(self, runner: RunnerState, env_params):
        cfg = self.cfg
        E = cfg.num_envs
        N = self.env_cfg.num_agents
        A = self.env_cfg.action_dim
        env_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

        def step(carry, _):
            env_state, obs, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            maps = obs['maps'].reshape((E * N,) + obs['maps'].shape[2:])
            lidar = obs['lidar'].reshape(E * N, -1)
            mean, log_std = self.actor.apply(runner.actor.params, maps, lidar)
            action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            logp = gaussian_log_prob(action, mean, log_std)
            value = self.critic.apply(runner.critic.params, obs['maps'], obs['lidar'], obs['global'], obs['agent'])
            next_obs, env_state, reward, done, info = env_step(
                jax.random.split(k_env, E), env_state, action.reshape(E, N, A), env_params)
            # The maps are stored channels first, the layout XLA picks for the
            # convolutions. Stored channels last, XLA transposes the whole
            # trajectory (2 GB for 256 environments) before the minibatch loop.
            transition = Transition(
                to_channels_first(obs['maps']), obs['lidar'], obs['global'], obs['agent'], action.reshape(E, N, A),
                logp.reshape(E, N), value, reward, done)
            info = dict(info, done=done)
            return (env_state, next_obs, key), (transition, info)

        (env_state, obs, key), (traj, infos) = jax.lax.scan(
            step, (runner.env_state, runner.obs, runner.key), None, cfg.num_steps)
        return runner._replace(env_state=env_state, obs=obs, key=key), traj, infos

    def _gae(self, traj: Transition, last_value, value_norm):
        cfg = self.cfg
        if cfg.use_value_norm:
            values = value_norm.denormalize(traj.value)
            last_value = value_norm.denormalize(last_value)
        else:
            values = traj.value
        dones = traj.done[..., None].astype(jnp.float32)

        def step(carry, x):
            gae, next_value = carry
            value, reward, done = x
            delta = reward + cfg.gamma * next_value * (1 - done) - value
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            step, (jnp.zeros_like(last_value), last_value), (values, traj.reward, dones), reverse=True)
        return advantages, advantages + values

    def _actor_loss(self, params, batch):
        cfg = self.cfg
        # flatten environment steps and robots
        maps = to_channels_last(batch['maps'])
        maps = maps.reshape((-1,) + maps.shape[2:])
        lidar = batch['lidar'].reshape((-1,) + batch['lidar'].shape[2:])
        action = batch['action'].reshape((-1,) + batch['action'].shape[2:])
        old_logp = batch['logp'].reshape(-1)
        adv = batch['adv'].reshape(-1)
        mean, log_std = self.actor.apply(params, maps, lidar)
        logp = gaussian_log_prob(action, mean, log_std)
        log_ratio = logp - old_logp
        ratio = jnp.exp(log_ratio)
        pg_loss = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        entropy = gaussian_entropy(log_std).mean()
        metrics = dict(
            actor_loss=pg_loss, entropy=entropy, log_std=log_std.mean(),
            approx_kl=((ratio - 1) - log_ratio).mean(),
            clip_frac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean())
        return pg_loss - cfg.ent_coef * entropy, metrics

    def _critic_loss(self, params, value_norm, batch):
        cfg = self.cfg
        value = self.critic.apply(params, to_channels_last(batch['maps']), batch['lidar'], batch['global_maps'], batch['agent'])
        target = batch['target']
        if cfg.use_value_norm:
            target = value_norm.normalize(target)
        old_value = batch['value']
        value_clipped = old_value + jnp.clip(value - old_value, -cfg.value_clip_eps, cfg.value_clip_eps)
        loss = jnp.maximum(huber(target - value, cfg.huber_delta),
                           huber(target - value_clipped, cfg.huber_delta)).mean()
        return loss, dict(critic_loss=loss)

    def _update(self, runner: RunnerState, env_params: EnvParams):
        cfg = self.cfg

        # Collect experience
        runner, traj, infos = self._rollout(runner, env_params)
        obs = runner.obs
        last_value = self.critic.apply(runner.critic.params, obs['maps'], obs['lidar'], obs['global'], obs['agent'])

        # Advantages and value targets
        advantages, targets = self._gae(traj, last_value, runner.value_norm)
        value_norm = runner.value_norm.update(targets) if cfg.use_value_norm else runner.value_norm
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten time and environments. Minibatches are sampled over
        # environment steps, with all robots of each step.
        M = cfg.num_steps * cfg.num_envs
        flat = lambda x: x.reshape((M,) + x.shape[2:])
        data = dict(maps=flat(traj.maps), lidar=flat(traj.lidar), global_maps=flat(traj.global_maps),
                    agent=flat(traj.agent), action=flat(traj.action), logp=flat(traj.logp),
                    value=flat(traj.value), adv=flat(advantages), target=flat(targets))
        num_mb, mb_size = self.num_minibatches, self.minibatch_env_steps

        def minibatch_step(carry, idx):
            actor, critic, stop, i = carry
            batch = {k: jnp.take(x, idx, axis=0) for k, x in data.items()}
            actor_grads, actor_metrics = jax.grad(self._actor_loss, has_aux=True)(actor.params, batch)
            critic_grads, critic_metrics = jax.grad(self._critic_loss, has_aux=True)(critic.params, value_norm, batch)
            # KL early stopping: once the policy moved too far from the one
            # that collected the data, skip the remaining actor updates of
            # this iteration (the critic keeps training). The KL of the first
            # minibatch is measured before any update, i.e. it is only
            # numerical noise, and must not freeze the actor.
            if cfg.target_kl > 0:
                stop = stop | ((i > 0) & (actor_metrics['approx_kl'] > 1.5 * cfg.target_kl))
            new_actor = actor.apply_gradients(grads=actor_grads)
            actor = jax.tree_util.tree_map(lambda old, new: jnp.where(stop, old, new), actor, new_actor)
            critic = critic.apply_gradients(grads=critic_grads)
            metrics = {**actor_metrics, **critic_metrics, 'actor_updates': 1.0 - stop.astype(jnp.float32)}
            return (actor, critic, stop, i + 1), metrics

        def epoch_step(carry, key):
            perm = jax.random.permutation(key, M)[:num_mb * mb_size].reshape(num_mb, mb_size)
            return jax.lax.scan(minibatch_step, carry, perm)

        key, k_epochs = jax.random.split(runner.key)
        (actor, critic, _, _), metrics = jax.lax.scan(
            epoch_step, (runner.actor, runner.critic, jnp.zeros((), dtype=bool), jnp.zeros((), dtype=jnp.int32)),
            jax.random.split(k_epochs, cfg.update_epochs))
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics['reward'] = traj.reward.mean()
        metrics['value_mean'], metrics['value_std'] = value_norm.stats()

        runner = runner._replace(actor=actor, critic=critic, value_norm=value_norm, key=key)
        return runner, metrics, infos


# ------------------------------------------------------------------ training


def make_training_maps(env_cfg: EnvConfig, train_cfg: TrainConfig, rng, maps_dir='maps', groups=(0, 1, 2, 3)):
    """
    Fixed training maps of the mowing levels, followed by num_random_maps
    random maps. Returns (fixed_bank, random_bank, groups, floor_plan, obstacles).
    """
    maps, map_groups = [], []
    for group in groups:
        for f in train_map_files(group, maps_dir):
            maps.append(load_map(f))
            map_groups.append(group)
    fixed_bank = build_bank(maps, env_cfg) if maps else None
    random_bank, floor_plan, obstacles = make_random_maps(env_cfg, train_cfg.num_random_maps, rng)
    groups = np.array(map_groups + [RANDOM_GROUP] * train_cfg.num_random_maps)
    floor_plan = np.concatenate([np.zeros(len(maps), dtype=bool), floor_plan])
    obstacles = np.concatenate([np.zeros(len(maps), dtype=bool), obstacles])
    return fixed_bank, random_bank, groups, floor_plan, obstacles


def make_random_maps(env_cfg: EnvConfig, num, rng):
    maps, floor_plan, obstacles = [], [], []
    for _ in range(num):
        known, unknown, fp, ob = random_map(rng, env_cfg)
        maps.append((known, unknown))
        floor_plan.append(fp)
        obstacles.append(ob)
    bank = build_bank(maps, env_cfg) if num > 0 else None
    return bank, np.array(floor_plan, dtype=bool), np.array(obstacles, dtype=bool)


def save_checkpoint(path, runner: RunnerState, env_cfg: EnvConfig, train_cfg: TrainConfig, extra=None):
    os.makedirs(path, exist_ok=True)
    state = {'actor': runner.actor.params, 'critic': runner.critic.params,
             'value_norm': {'mean': runner.value_norm.mean, 'mean_sq': runner.value_norm.mean_sq,
                            'debias': runner.value_norm.debias}}
    with open(os.path.join(path, 'params.msgpack'), 'wb') as f:
        f.write(flax.serialization.to_bytes(jax.device_get(state)))
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump({'env': env_cfg.to_dict(), 'train': train_cfg.to_dict(), **(extra or {})}, f, indent=2)


def load_checkpoint(path, env_cfg: EnvConfig = None, train_cfg: TrainConfig = None):
    """Returns (params dict, env_cfg, train_cfg), configs are read from the checkpoint if not given"""
    with open(os.path.join(path, 'config.json')) as f:
        config = json.load(f)
    env_cfg = env_cfg or EnvConfig.from_dict(config['env'])
    if train_cfg is None:
        # checkpoints from before squash_mean / min_log_std: unbounded actor
        legacy = {'squash_mean': False, 'min_log_std': float('-inf')}
        train_cfg = TrainConfig.from_dict({**legacy, **config['train']})
    actor, critic = make_networks(env_cfg, train_cfg)
    actor_params, critic_params = init_params(env_cfg, actor, critic, jax.random.PRNGKey(0))
    target = {'actor': actor_params, 'critic': critic_params,
              'value_norm': {'mean': jnp.zeros(()), 'mean_sq': jnp.zeros(()), 'debias': jnp.zeros(())}}
    with open(os.path.join(path, 'params.msgpack'), 'rb') as f:
        params = flax.serialization.from_bytes(target, f.read())
    return params, env_cfg, train_cfg


def train(env_cfg: EnvConfig, train_cfg: TrainConfig, logdir, maps_dir='maps', checkpoint=None, verbose=True):
    os.makedirs(logdir, exist_ok=True)
    rng = np.random.default_rng(train_cfg.seed)
    key = jax.random.PRNGKey(train_cfg.seed)

    # Maps and curriculum
    fixed_bank, random_bank, groups, floor_plan, obstacles = make_training_maps(env_cfg, train_cfg, rng, maps_dir)
    banks = [b for b in [fixed_bank, random_bank] if b is not None]
    bank = concat_banks(banks) if len(banks) > 1 else banks[0]
    num_fixed = 0 if fixed_bank is None else fixed_bank.num_maps
    curriculum = Curriculum(groups, floor_plan, obstacles, train_cfg.start_level, train_cfg.curriculum)

    def env_params():
        return EnvParams(bank=bank, map_logits=jnp.asarray(curriculum.map_logits()),
                         goal_coverage=jnp.float32(curriculum.goal_coverage))

    # Agent
    algo = MAPPO(env_cfg, train_cfg)
    actor_params = critic_params = None
    if checkpoint is not None:
        params, _, _ = load_checkpoint(checkpoint, env_cfg, train_cfg)
        actor_params, critic_params = params['actor'], params['critic']
    runner = algo.init(key, env_params(), actor_params, critic_params)
    params = env_params()

    log_file = open(os.path.join(logdir, 'progress.csv'), 'w', newline='')
    writer = None
    steps_per_update = train_cfg.num_envs * train_cfg.num_steps
    t_start = time.time()
    t_last = t_start
    if verbose:
        print(f'{algo.num_updates} updates of {steps_per_update} environment steps '
              f'({steps_per_update * env_cfg.num_agents} agent steps)', flush=True)

    for update in range(1, algo.num_updates + 1):
        runner, metrics, infos = algo.update(runner, params)

        # Episode statistics and curriculum (on the host)
        infos = jax.device_get(infos)
        metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        done = infos['done'].astype(bool)
        level_changed = curriculum.update(infos['map_idx'][done], infos['success'][done])

        # Regenerate the random maps
        regenerate = train_cfg.random_maps_every > 0 and update % train_cfg.random_maps_every == 0 and \
            random_bank is not None and curriculum.spec['p_random'] > 0
        if regenerate:
            random_bank, fp, ob = make_random_maps(env_cfg, train_cfg.num_random_maps, rng)
            bank = concat_banks([fixed_bank, random_bank]) if fixed_bank is not None else random_bank
            curriculum.set_random_maps(fp, ob)
        if level_changed or regenerate:
            params = env_params()

        if update % train_cfg.log_every == 0 or update == algo.num_updates:
            now = time.time()
            n_done = int(done.sum())
            ep = lambda k: float(infos[k][done].mean()) if n_done > 0 else float('nan')
            row = dict(
                update=update,
                env_steps=update * steps_per_update,
                agent_steps=update * steps_per_update * env_cfg.num_agents,
                sps=train_cfg.log_every * steps_per_update / (now - t_last) if update > 1 else 0.0,
                time=now - t_start,
                level=curriculum.level,
                goal_coverage=curriculum.goal_coverage,
                episodes=n_done,
                success=ep('success'),
                coverage=ep('coverage'),
                overlap=ep('overlap'),
                ep_length=ep('length'),
                ep_return=ep('ep_return'),
                collisions=ep('collisions'),
                **metrics)
            t_last = now
            if writer is None:
                writer = csv.DictWriter(log_file, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            log_file.flush()
            if verbose:
                print(f"update {update}/{algo.num_updates}  steps {row['env_steps']:,}  "
                      f"sps {row['sps']:,.0f}  level {row['level']}  episodes {n_done}  "
                      f"success {row['success']:.2f}  coverage {row['coverage']:.3f}  "
                      f"ep_len {row['ep_length']:.0f}  reward {metrics['reward']:.3f}  "
                      f"kl {metrics['approx_kl']:.4f}  actor_updates {metrics['actor_updates']:.2f}  "
                      f"entropy {metrics['entropy']:.3f}", flush=True)

        if update % train_cfg.save_every == 0 or update == algo.num_updates:
            save_checkpoint(os.path.join(logdir, 'checkpoint'), runner, env_cfg, train_cfg,
                            extra={'update': update, 'level': curriculum.level})

    log_file.close()
    return runner
