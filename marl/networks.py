"""
Actor and centralized critic in Flax (channels last, NHWC).

The actor only sees the robot's own egocentric observation (decentralized
execution). The critic additionally sees the global state of the environment
and the robot's own state, i.e. an agent-specific global state as in MAPPO.
The critic takes all robots of an environment at once, so that the global
maps are only encoded once per environment.
Parameters are shared between robots.
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


class Actor(nn.Module):
    """Gaussian policy, returns mean and log std of the (unclipped) action."""
    action_dim: int
    num_maps: int
    hidden_dims: int = 256
    grouped: bool = True
    init_log_std: float = -0.5

    @nn.compact
    def __call__(self, maps, lidar):
        x = MultiScaleEncoder(self.num_maps, self.hidden_dims, self.grouped)(maps, lidar)
        x = nn.relu(_dense(self.hidden_dims)(x))
        x = nn.relu(_dense(self.hidden_dims)(x))
        mean = _dense(self.action_dim, scale=0.01)(x)
        log_std = self.param('log_std', nn.initializers.constant(self.init_log_std), (self.action_dim,))
        return mean, jnp.broadcast_to(log_std, mean.shape)


class Critic(nn.Module):
    """
    Centralized value function V(s, i) for each robot i of an environment.

    Inputs (one row per environment, all N robots together):
      maps, lidar: (B, N, ...) egocentric observation of each robot
      global_maps: (B, G, G, C) global maps of the environment
      agent:       (B, N, F) state of each robot (position, heading, ...)
    Returns (B, N) values.

    The global maps are encoded once per environment and the features are
    shared by its robots, the robot specific part comes from the robot's own
    observation and state (agent-specific global state).
    """
    num_maps: int
    hidden_dims: int = 256
    grouped: bool = True
    global_convs: Sequence[int] = (32, 64, 64)

    @nn.compact
    def __call__(self, maps, lidar, global_maps, agent):
        B, N = maps.shape[:2]
        ego = MultiScaleEncoder(self.num_maps, self.hidden_dims, self.grouped)(
            maps.reshape((B * N,) + maps.shape[2:]), lidar.reshape(B * N, -1))
        ego = ego.reshape(B, N, -1)
        g = global_maps.astype(jnp.float32) / 255
        for features, kernel in zip(self.global_convs, [(4, 4), (4, 4), (3, 3)]):
            g = nn.relu(nn.Conv(features, kernel, strides=2, padding='SAME',
                                kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(g))
        g = g.reshape(B, -1)
        g = nn.relu(_dense(self.hidden_dims)(g))
        g = jnp.broadcast_to(g[:, None, :], (B, N, g.shape[-1]))
        x = jnp.concatenate([ego, g, agent], axis=-1)
        x = nn.relu(_dense(self.hidden_dims)(x))
        x = nn.relu(_dense(self.hidden_dims)(x))
        return _dense(1, scale=1.0)(x)[..., 0]
