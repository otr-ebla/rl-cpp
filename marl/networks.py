"""
Actor, centralized critic (MAPPO) and local critic (IPPO) in Flax
(channels last, NHWC).

The actor only sees the robot's own egocentric observation (decentralized
execution). The critic additionally sees the global state of the environment
and the robot's own state, i.e. an agent-specific global state as in MAPPO.
The local critic of IPPO only sees the robot's own observation, like the actor.
The critic takes all robots of an environment at once, so that the global
maps are only encoded once per environment.
Parameters are shared between robots.

With neighbor_observation, the other robots are given as a set of poses,
encoded with self-attention by PoseEncoder (a branch next to the maps and
lidar branch, concatenated with it), so that no parameter depends on the
number of robots: the same networks run with any number of robots (also 1).
The centralized critic uses either the agent-specific global state (Critic)
or the global state only (GlobalCritic), see TrainConfig.critic_state.
"""
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def _dense(features, scale=np.sqrt(2)):
    return nn.Dense(features, kernel_init=nn.initializers.orthogonal(scale),
                    bias_init=nn.initializers.zeros)


class GroupedConv(nn.Module):
    """
    Grouped convolution (VALID padding) implemented as a regular convolution
    with a block diagonal kernel. Equivalent to feature_group_count=groups,
    but uses the standard convolution kernels, since grouped convolutions
    are very slow to differentiate on some XLA backends.
    """
    features: int
    kernel_size: Sequence[int]
    strides: int = 1
    groups: int = 1

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        assert in_features % self.groups == 0 and self.features % self.groups == 0
        kernel = self.param('kernel', nn.initializers.orthogonal(np.sqrt(2)),
                            tuple(self.kernel_size) + (in_features, self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        if self.groups > 1:
            g_in = np.arange(in_features) // (in_features // self.groups)
            g_out = np.arange(self.features) // (self.features // self.groups)
            kernel = kernel * jnp.asarray(g_in[:, None] == g_out[None, :], dtype=kernel.dtype)
        y = jax.lax.conv_general_dilated(
            x, kernel, window_strides=(self.strides,) * 2, padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        return y + bias


def _conv(features, kernel, strides=1, groups=1):
    return GroupedConv(features, kernel, strides, groups)


class MultiScaleEncoder(nn.Module):
    """
    Encoder of the egocentric multi-scale maps and lidar, same architecture
    as rlm.architectures.StackedMapFeaturesExtractor. With grouped
    convolutions, each scale is processed separately until the dense layer.
    """
    num_maps: int
    features: int = 256
    grouped: bool = True

    @nn.compact
    def __call__(self, maps, lidar):
        x = maps.astype(jnp.float32) / 255
        channels = 2 * x.shape[-1]
        groups = self.num_maps if self.grouped else 1
        x = nn.relu(_conv(channels, (2, 2), strides=2, groups=groups)(x))
        for _ in range(3):
            x = nn.relu(_conv(channels, (3, 3), groups=groups)(x))
        x = x.reshape(x.shape[:-3] + (-1,))
        x = nn.relu(_dense(self.features)(x))
        lidar_features = nn.relu(_dense(lidar.shape[-1])(lidar))
        x = jnp.concatenate([x, lidar_features], axis=-1)
        return nn.relu(_dense(self.features)(x))


class PoseEncoder(nn.Module):
    """
    Self-attention among the poses of a set of robots of any size. Each pose
    is embedded by the same MLP, then pre-LayerNorm transformer blocks
    (multi-head self-attention and MLP, with residuals) relate the robots to
    each other. There is no positional encoding, so the result does not
    depend on the order of the robots, and no parameter depends on their
    number. The softmax of the attention is normalized, so the inputs of the
    later layers keep their range with more or fewer robots.

    With ego, a learned ego token (the observing robot, at the origin of its
    own frame) is prepended to the set and its output is returned, i.e. one
    vector that summarizes the other robots as seen by the observer. With a
    single robot (empty set) it is a constant. Without ego, one output per
    robot is returned.

    Inputs: poses (..., K, F), mask (..., K) bool, robots that are present.
    Returns (..., features) with ego, (..., K, features) without.
    """
    features: int = 64
    layers: int = 2
    heads: int = 4
    ego: bool = True

    @nn.compact
    def __call__(self, poses, mask):
        F = self.features
        x = _dense(F)(nn.relu(_dense(F)(poses)))
        if self.ego:
            token = self.param('ego_token', nn.initializers.normal(1.0), (F,))
            x = jnp.concatenate([jnp.broadcast_to(token, x.shape[:-2] + (1, F)), x], axis=-2)
            mask = jnp.concatenate([jnp.ones(mask.shape[:-1] + (1,), dtype=bool), mask], axis=-1)
        attn_mask = mask[..., None, None, :]                                # ... x heads x queries x keys
        for _ in range(self.layers):
            y = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=F)(y, y, mask=attn_mask)
            y = nn.LayerNorm()(x)
            x = x + _dense(F)(nn.relu(_dense(2 * F)(y)))
        x = nn.LayerNorm()(x)
        return x[..., 0, :] if self.ego else x


def local_encoding(m, maps, lidar, others, others_mask):
    """
    Features of a robot's own observation, inputs with one leading batch axis:
    the multi-scale maps and lidar branch, and (if others is given) the pose
    branch of the observed robots, concatenated.
    Called from the compact method of module m (a function, not a module, so
    that the parameter names of checkpoints without the pose branch are unchanged).
    """
    x = MultiScaleEncoder(m.num_maps, m.hidden_dims, m.grouped)(maps, lidar)
    if others is None:
        return x
    n = PoseEncoder(m.neighbor_dims, m.neighbor_layers, m.neighbor_heads)(others, others_mask)
    return jnp.concatenate([x, n], axis=-1)


def global_maps_encoding(m, global_maps):
    """(B, hidden_dims) features of the global maps (B, G, G, C), see local_encoding"""
    g = global_maps.astype(jnp.float32) / 255
    for features, kernel in zip(m.global_convs, [(4, 4), (4, 4), (3, 3)]):
        g = nn.relu(nn.Conv(features, kernel, strides=2, padding='SAME',
                            kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(g))
    g = g.reshape(g.shape[0], -1)
    return nn.relu(_dense(m.hidden_dims)(g))


class Actor(nn.Module):
    """
    Gaussian policy, returns mean and log std of the (unclipped) action.

    With squash_mean, the mean is squashed into the action range [-1, 1] with
    tanh. Otherwise the mean can drift far outside of it, where all sampled
    actions are clipped to the same value and the policy gets no gradient
    back. log_std is bounded from below by min_log_std with a soft floor
    (min_log_std + softplus), so that it never gets stuck there (-inf: off).
    """
    action_dim: int
    num_maps: int
    hidden_dims: int = 256
    grouped: bool = True
    init_log_std: float = -0.5
    squash_mean: bool = True
    min_log_std: float = -2.0
    neighbor_dims: int = 64
    neighbor_layers: int = 2
    neighbor_heads: int = 4

    @nn.compact
    def __call__(self, maps, lidar, others=None, others_mask=None):
        """maps, lidar: (B, ...), others: (B, K, F) and others_mask: (B, K) or None"""
        x = local_encoding(self, maps, lidar, others, others_mask)
        x = nn.relu(_dense(self.hidden_dims)(x))
        x = nn.relu(_dense(self.hidden_dims)(x))
        mean = _dense(self.action_dim, scale=0.01)(x)
        if self.squash_mean:
            mean = jnp.tanh(mean)
        if np.isfinite(self.min_log_std):
            # parameter initialized such that log_std == init_log_std
            init = self.min_log_std + np.log(np.expm1(self.init_log_std - self.min_log_std))
            log_std = self.param('log_std', nn.initializers.constant(init), (self.action_dim,))
            log_std = self.min_log_std + nn.softplus(log_std - self.min_log_std)
        else:
            log_std = self.param('log_std', nn.initializers.constant(self.init_log_std), (self.action_dim,))
        return mean, jnp.broadcast_to(log_std, mean.shape)


class Critic(nn.Module):
    """
    Centralized value function V(s, i) for each robot i of an environment, on
    the agent-specific global state (TrainConfig.critic_state 'agent_specific'):
    the global state together with the robot's own observation.

    Inputs (one row per environment, all N robots together):
      maps, lidar: (B, N, ...) egocentric observation of each robot
      global_maps: (B, G, G, C) global maps of the environment
      agent:       (B, N, F) state of each robot (position, heading, ...)
      others, others_mask: (B, N, N, ...) robots observed by each robot
      team:        (B, N, N, F') true poses of the other robots
      progress:    (B, P) episode progress (see MultiMowerEnv)
    Returns (B, N) values.

    The global maps are encoded once per environment and the features are
    shared by its robots, the robot specific part comes from the robot's own
    observation and state. With team, all other robots are encoded with
    self-attention from the point of view of each robot.
    """
    num_maps: int
    hidden_dims: int = 256
    grouped: bool = True
    global_convs: Sequence[int] = (32, 64, 64)
    neighbor_dims: int = 64
    neighbor_layers: int = 2
    neighbor_heads: int = 4

    @nn.compact
    def __call__(self, maps, lidar, global_maps, agent, others=None, others_mask=None, team=None, progress=None):
        B, N = maps.shape[:2]
        flat = lambda x: None if x is None else x.reshape((B * N,) + x.shape[2:])
        ego = local_encoding(self, flat(maps), flat(lidar), flat(others), flat(others_mask))
        parts = [ego.reshape(B, N, -1)]
        if team is not None:
            team_mask = jnp.broadcast_to(~jnp.eye(N, dtype=bool), (B, N, N))
            parts.append(PoseEncoder(self.neighbor_dims, self.neighbor_layers, self.neighbor_heads)(team, team_mask))
        g = global_maps_encoding(self, global_maps)
        parts += [jnp.broadcast_to(g[:, None, :], (B, N, g.shape[-1])), agent]
        if progress is not None:
            parts.append(jnp.broadcast_to(progress[:, None, :], (B, N, progress.shape[-1])))
        x = jnp.concatenate(parts, axis=-1)
        x = nn.relu(_dense(self.hidden_dims)(x))
        x = nn.relu(_dense(self.hidden_dims)(x))
        return _dense(1, scale=1.0)(x)[..., 0]


class GlobalCritic(nn.Module):
    """
    Centralized value function V(s, i) for each robot i of an environment, on
    the global state only (TrainConfig.critic_state 'global'), without the
    robots' own observations. The robot i is identified by its pose in the
    global state: the poses of all robots (canvas frame) are related by
    self-attention, one output per robot, which is combined with the global
    maps and the episode progress. Nothing depends on the number of robots.

    Inputs (one row per environment, all N robots together):
      global_maps: (B, G, G, C) global maps of the environment
      agent:       (B, N, 6) position / canvas size, cos/sin heading, team
                   coverage, elapsed time (see MultiMowerEnv)
      progress:    (B, P) episode progress (see MultiMowerEnv)
    Returns (B, N) values.
    """
    hidden_dims: int = 256
    global_convs: Sequence[int] = (32, 64, 64)
    neighbor_dims: int = 64
    neighbor_layers: int = 2
    neighbor_heads: int = 4

    @nn.compact
    def __call__(self, global_maps, agent, progress):
        B, N = agent.shape[:2]
        poses, scalars = agent[..., :4], agent[:, 0, 4:]                   # coverage and time: same for all robots
        robots = PoseEncoder(self.neighbor_dims, self.neighbor_layers, self.neighbor_heads, ego=False)(
            poses, jnp.ones((B, N), dtype=bool))
        g = global_maps_encoding(self, global_maps)
        context = jnp.concatenate([g, scalars, progress], axis=-1)
        x = jnp.concatenate([robots, jnp.broadcast_to(context[:, None, :], (B, N, context.shape[-1]))], axis=-1)
        x = nn.relu(_dense(self.hidden_dims)(x))
        x = nn.relu(_dense(self.hidden_dims)(x))
        return _dense(1, scale=1.0)(x)[..., 0]


class LocalCritic(nn.Module):
    """
    Independent (IPPO) value function V(o_i) for each robot i, computed from
    the robot's own egocentric observation only, the same inputs as the actor.
    The observation still comes from the global maps shared by all robots.

    Inputs: maps, lidar, others, others_mask: (B, N, ...). Returns (B, N) values.
    """
    num_maps: int
    hidden_dims: int = 256
    grouped: bool = True
    neighbor_dims: int = 64
    neighbor_layers: int = 2
    neighbor_heads: int = 4

    @nn.compact
    def __call__(self, maps, lidar, others=None, others_mask=None):
        B, N = maps.shape[:2]
        flat = lambda x: None if x is None else x.reshape((B * N,) + x.shape[2:])
        x = local_encoding(self, flat(maps), flat(lidar), flat(others), flat(others_mask))
        x = nn.relu(_dense(self.hidden_dims)(x))
        x = nn.relu(_dense(self.hidden_dims)(x))
        return _dense(1, scale=1.0)(x)[..., 0].reshape(B, N)
