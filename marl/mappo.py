"""
MAPPO (multi-agent PPO with a centralized critic) and IPPO (independent PPO
with a local critic) in JAX.

One call of the jitted update function collects num_steps steps in num_envs
parallel environments and then runs update_epochs epochs of PPO on the
collected data, entirely on the device. The Python loop around it (train())
only handles logging, the curriculum, regeneration of the random maps and
checkpoints.

Actor and critic parameters are shared between the robots. Each robot gets
its own reward, advantage and value: V(s, i) on the global state or on the
agent-specific global state with MAPPO (TrainConfig.critic_state), V(o_i) on
its own observation with IPPO (TrainConfig.algo).
Everything else is the same, so the two can be compared directly.
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
from marl.networks import Actor, Critic, GlobalCritic, LocalCritic


class Transition(NamedTuple):
    maps: jnp.ndarray       # T x E x N x C x H x W uint8 (channels first, see _rollout)
    lidar: jnp.ndarray      # T x E x N x R
    global_maps: jnp.ndarray  # T x E x G x G x C uint8 (None with IPPO)
    agent: jnp.ndarray      # T x E x N x F (None with IPPO)
    others: jnp.ndarray     # T x E x N x N x F (None without neighbor_observation)
    others_mask: jnp.ndarray  # T x E x N x N bool (None without neighbor_observation)
    team: jnp.ndarray       # T x E x N x N x F' (only with the agent-specific critic and neighbor_observation)
    progress: jnp.ndarray   # T x E x P (None with IPPO or without neighbor_observation)
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
    pose = dict(neighbor_dims=train_cfg.neighbor_dims, neighbor_layers=train_cfg.neighbor_layers,
                neighbor_heads=train_cfg.neighbor_heads)
    actor = Actor(env_cfg.action_dim, env_cfg.num_maps, train_cfg.hidden_dims,
                  train_cfg.grouped_convs, train_cfg.init_log_std, train_cfg.squash_mean, train_cfg.min_log_std,
                  **pose)
    if not train_cfg.centralized_critic:
        critic = LocalCritic(env_cfg.num_maps, train_cfg.hidden_dims, train_cfg.grouped_convs, **pose)
    elif train_cfg.critic_state == 'global':
        assert env_cfg.neighbor_observation, 'the global critic needs neighbor_observation'
        critic = GlobalCritic(train_cfg.hidden_dims, **pose)
    else:
        critic = Critic(env_cfg.num_maps, train_cfg.hidden_dims, train_cfg.grouped_convs, **pose)
    return actor, critic


def actor_apply(actor, params, obs):
    """
    Mean and log std of the actions for observations (dict, see
    MultiMowerEnv) with any leading axes (..., N), e.g. (N,) for one
    environment or (E, N). Returns (..., N, action_dim) arrays.
    """
    lead = obs['lidar'].shape[:-1]
    flat = lambda x: None if x is None else x.reshape((-1,) + x.shape[len(lead):])
    mean, log_std = actor.apply(params, flat(obs['maps']), flat(obs['lidar']),
                                flat(obs.get('others')), flat(obs.get('others_mask')))
    return mean.reshape(lead + mean.shape[1:]), log_std.reshape(lead + log_std.shape[1:])


def critic_inputs(critic, obs):
    """Arguments of the critic from observations (dict, see MultiMowerEnv) of B environments"""
    if isinstance(critic, LocalCritic):
        return obs['maps'], obs['lidar'], obs.get('others'), obs.get('others_mask')
    if isinstance(critic, GlobalCritic):
        return obs['global'], obs['agent'], obs['progress']
    return (obs['maps'], obs['lidar'], obs['global'], obs['agent'],
            obs.get('others'), obs.get('others_mask'), obs.get('team'), obs.get('progress'))


def critic_value(critic, params, obs):
    """(B, N) values for observations (dict, see MultiMowerEnv) of B environments"""
    return critic.apply(params, *critic_inputs(critic, obs))


def init_params(env_cfg: EnvConfig, actor, critic, key):
    """Initial actor and critic parameters"""
    H = env_cfg.input_size
    G = env_cfg.global_size
    obs = {'maps': jnp.zeros((1, 1, H, H, env_cfg.obs_channels), dtype=jnp.uint8),
           'lidar': jnp.zeros((1, 1, env_cfg.lidar_rays)),
           'global': jnp.zeros((1, G, G, env_cfg.global_channels), dtype=jnp.uint8),
           'agent': jnp.zeros((1, 1, env_cfg.agent_features))}
    if env_cfg.neighbor_observation:
        obs.update(others=jnp.zeros((1, 1, 1, env_cfg.neighbor_features)),
                   others_mask=jnp.zeros((1, 1, 1), dtype=bool),
                   team=jnp.zeros((1, 1, 1, env_cfg.team_features)),
                   progress=jnp.zeros((1, env_cfg.progress_features)))
    k_actor, k_critic = jax.random.split(key)
    first = lambda x: None if x is None else x[0]
    local = (obs['maps'], obs['lidar'], obs.get('others'), obs.get('others_mask'))
    return actor.init(k_actor, *map(first, local)), critic.init(k_critic, *critic_inputs(critic, obs))


class MAPPO:
    """MAPPO, or IPPO with train_cfg.algo == 'ippo'"""

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
        self.update = jax.jit(self._update, donate_argnums=0, static_argnames='train_actor')

    # ---------------------------------------------------------------- setup

    def _optimizer(self, lr, adaptive=False):
        """
        Adam with gradient clipping. With adaptive, the learning rate is a
        hyperparameter in the optimizer state (opt_state[1].hyperparams), set
        after each iteration from the KL (see _adapt_actor_lr), otherwise it
        is constant or annealed linearly to 0 (anneal_lr).
        """
        cfg = self.cfg
        if adaptive:
            return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                               optax.inject_hyperparams(optax.adam)(learning_rate=lr, eps=1e-5))
        if cfg.anneal_lr:
            steps = self.num_updates * cfg.update_epochs * self.num_minibatches
            lr = optax.linear_schedule(lr, 0.0, steps)
        return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))

    @staticmethod
    def actor_lr(actor: TrainState):
        """Current learning rate of an actor with the adaptive learning rate"""
        return actor.opt_state[1].hyperparams['learning_rate']

    def _adapt_actor_lr(self, actor: TrainState, kl):
        """
        Adaptive learning rate of the actor (as in rsl_rl): divided by 1.5 when
        the KL of the iteration exceeds 1.5 * target_kl (where early stopping
        stops it), multiplied by 1.5 when it is below target_kl / 2, within
        [min_actor_lr, max_actor_lr]
        """
        cfg = self.cfg
        lr = self.actor_lr(actor)
        lr = jnp.where(kl > 1.5 * cfg.target_kl, lr / 1.5, jnp.where(kl < cfg.target_kl / 2, lr * 1.5, lr))
        lr = jnp.clip(lr, cfg.min_actor_lr, cfg.max_actor_lr)
        inject = actor.opt_state[1]
        inject = inject._replace(hyperparams={**inject.hyperparams, 'learning_rate': lr})
        return actor.replace(opt_state=(actor.opt_state[0], inject) + tuple(actor.opt_state[2:]))

    def init(self, key, env_params: EnvParams, actor_params=None, critic_params=None):
        cfg = self.cfg
        k_init, k_env, key = jax.random.split(key, 3)
        actor_init, critic_init = init_params(self.env_cfg, self.actor, self.critic, k_init)
        actor_params = actor_init if actor_params is None else actor_params
        critic_params = critic_init if critic_params is None else critic_params
        actor = TrainState.create(apply_fn=self.actor.apply, params=actor_params,
                                  tx=self._optimizer(cfg.actor_lr, cfg.adaptive_lr))
        critic = TrainState.create(apply_fn=self.critic.apply, params=critic_params,
                                   tx=self._optimizer(cfg.critic_lr))
        reset = jax.jit(jax.vmap(self.env.reset, in_axes=(0, None)))
        obs, env_state = reset(jax.random.split(k_env, cfg.num_envs), env_params)
        return RunnerState(actor, critic, ValueNorm.create(), env_state, obs, key)

    # --------------------------------------------------------------- update

    def _value(self, params, obs):
        return critic_value(self.critic, params, obs)

    def _rollout(self, runner: RunnerState, env_params):
        cfg = self.cfg
        E = cfg.num_envs
        env_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

        def step(carry, _):
            env_state, obs, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            mean, log_std = actor_apply(self.actor, runner.actor.params, obs)
            action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            logp = gaussian_log_prob(action, mean, log_std)
            value = self._value(runner.critic.params, obs)
            next_obs, env_state, reward, done, info = env_step(
                jax.random.split(k_env, E), env_state, action, env_params)
            # The maps are stored channels first, the layout XLA picks for the
            # convolutions. Stored channels last, XLA transposes the whole
            # trajectory (2 GB for 256 environments) before the minibatch loop.
            # The local critic of IPPO does not use the global state, it is
            # not stored (None).
            central = self.cfg.centralized_critic
            agent_specific = central and self.cfg.critic_state == 'agent_specific'
            transition = Transition(
                to_channels_first(obs['maps']), obs['lidar'],
                obs['global'] if central else None, obs['agent'] if central else None,
                obs.get('others'), obs.get('others_mask'), obs.get('team') if agent_specific else None,
                obs.get('progress') if central else None,
                action, logp, value, reward, done)
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

    @staticmethod
    def _batch_obs(batch):
        """Observation dict (see MultiMowerEnv) of a minibatch"""
        obs = {k: batch[k] for k in ('lidar', 'agent', 'others', 'others_mask', 'team', 'progress') if k in batch}
        obs['maps'] = to_channels_last(batch['maps'])
        if 'global_maps' in batch:
            obs['global'] = batch['global_maps']
        return obs

    def _actor_loss(self, params, batch):
        cfg = self.cfg
        action = batch['action']
        old_logp = batch['logp']
        adv = batch['adv']
        mean, log_std = actor_apply(self.actor, params, self._batch_obs(batch))
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
        value = self._value(params, self._batch_obs(batch))
        target = batch['target']
        if cfg.use_value_norm:
            target = value_norm.normalize(target)
        old_value = batch['value']
        value_clipped = old_value + jnp.clip(value - old_value, -cfg.value_clip_eps, cfg.value_clip_eps)
        loss = jnp.maximum(huber(target - value, cfg.huber_delta),
                           huber(target - value_clipped, cfg.huber_delta)).mean()
        return loss, dict(critic_loss=loss)

    def _update(self, runner: RunnerState, env_params: EnvParams, train_actor=True):
        """One iteration, train_actor=False only trains the critic (warm-up)"""
        cfg = self.cfg

        # Collect experience
        runner, traj, infos = self._rollout(runner, env_params)
        last_value = self._value(runner.critic.params, runner.obs)

        # Advantages and value targets
        advantages, targets = self._gae(traj, last_value, runner.value_norm)
        value_norm = runner.value_norm.update(targets) if cfg.use_value_norm else runner.value_norm
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten time and environments. Minibatches are sampled over
        # environment steps, with all robots of each step.
        M = cfg.num_steps * cfg.num_envs
        flat = lambda x: x.reshape((M,) + x.shape[2:])
        data = dict(traj._asdict(), adv=advantages, target=targets)
        data = {k: flat(x) for k, x in data.items() if x is not None and k not in ('reward', 'done')}
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
            epoch_step, (runner.actor, runner.critic, jnp.full((), not train_actor), jnp.zeros((), dtype=jnp.int32)),
            jax.random.split(k_epochs, cfg.update_epochs))
        # KL of the last minibatch: after early stopping the actor is frozen,
        # so it is the KL of the whole iteration
        final_kl = metrics['approx_kl'][-1, -1]
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics['final_kl'] = final_kl
        if cfg.adaptive_lr:
            if train_actor:
                actor = self._adapt_actor_lr(actor, final_kl)
            metrics['actor_lr'] = self.actor_lr(actor)
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
    # checkpoints from before neighbor_observation: robot map channels
    env_cfg = env_cfg or EnvConfig.from_dict({'neighbor_observation': False, **config['env']})
    if train_cfg is None:
        # checkpoints from before squash_mean / min_log_std: unbounded actor
        # and before critic_state: agent-specific critic
        legacy = {'squash_mean': False, 'min_log_std': float('-inf'), 'critic_state': 'agent_specific'}
        train_cfg = TrainConfig.from_dict({**legacy, **config['train']})
    actor, critic = make_networks(env_cfg, train_cfg)
    actor_params, critic_params = init_params(env_cfg, actor, critic, jax.random.PRNGKey(0))
    target = {'actor': actor_params, 'critic': critic_params,
              'value_norm': {'mean': jnp.zeros(()), 'mean_sq': jnp.zeros(()), 'debias': jnp.zeros(())}}
    with open(os.path.join(path, 'params.msgpack'), 'rb') as f:
        params = flax.serialization.from_bytes(target, f.read())
    return params, env_cfg, train_cfg


def load_init_params(path, env_cfg: EnvConfig, train_cfg: TrainConfig):
    """
    Initial (actor, critic) parameters for training from a checkpoint. The
    actor must match the networks of env_cfg / train_cfg. The critic is only
    loaded when it has the same structure, otherwise it is None (newly
    initialized), e.g. IPPO (local critic) started from a MAPPO checkpoint or
    from init_from_sb3.py, whose critic is new anyway.
    """
    actor, critic = make_networks(env_cfg, train_cfg)
    actor_target, critic_target = init_params(env_cfg, actor, critic, jax.random.PRNGKey(0))
    with open(os.path.join(path, 'params.msgpack'), 'rb') as f:
        raw = flax.serialization.msgpack_restore(f.read())

    def restore(target, state):
        try:
            params = flax.serialization.from_state_dict(target, state)
        except (KeyError, ValueError):
            return None
        same = jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(target) and all(
            np.shape(a) == np.shape(b) for a, b in zip(jax.tree_util.tree_leaves(params),
                                                       jax.tree_util.tree_leaves(target)))
        return params if same else None

    actor_params = restore(actor_target, raw['actor'])
    assert actor_params is not None, f'{path}: the actor does not match the configuration'
    critic_params = restore(critic_target, raw['critic'])
    if critic_params is None:
        print(f'{path}: the critic does not match the configuration ({type(critic).__name__}), '
              f'it is newly initialized', flush=True)
    return actor_params, critic_params


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
        actor_params, critic_params = load_init_params(checkpoint, env_cfg, train_cfg)
    runner = algo.init(key, env_params(), actor_params, critic_params)
    params = env_params()

    log_file = open(os.path.join(logdir, 'progress.csv'), 'w', newline='')
    writer = None
    steps_per_update = train_cfg.num_envs * train_cfg.num_steps
    t_start = time.time()
    t_last = t_start
    recent, best_success = [], -1.0
    if verbose:
        print(f'{train_cfg.algo.upper()}: {algo.num_updates} updates of {steps_per_update} environment steps '
              f'({steps_per_update * env_cfg.num_agents} agent steps)', flush=True)

    for update in range(1, algo.num_updates + 1):
        runner, metrics, infos = algo.update(runner, params, train_actor=update > train_cfg.critic_warmup)

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
                      f"entropy {metrics['entropy']:.3f}" +
                      (f"  actor_lr {metrics['actor_lr']:.2e}" if 'actor_lr' in metrics else ''), flush=True)

        if update % train_cfg.save_every == 0 or update == algo.num_updates:
            save_checkpoint(os.path.join(logdir, 'checkpoint'), runner, env_cfg, train_cfg,
                            extra={'update': update, 'level': curriculum.level})

        # Best policy at the last level (success rate over the episodes of the
        # last save_every updates), so that a late collapse does not lose it
        if curriculum.level == len(curriculum.levels):
            recent.append((int(done.sum()), float(infos['success'][done].sum())))
            recent = recent[-train_cfg.save_every:]
            episodes = sum(n for n, _ in recent)
            if len(recent) == train_cfg.save_every and update % 10 == 0 and episodes > 0:
                success = sum(s for _, s in recent) / episodes
                if success > best_success:
                    best_success = success
                    save_checkpoint(os.path.join(logdir, 'checkpoint_best'), runner, env_cfg, train_cfg,
                                    extra={'update': update, 'level': curriculum.level, 'success': success,
                                           'episodes': episodes})
        else:
            recent = []

    log_file.close()
    return runner
