"""
Tests for the multi-agent JAX environment and MAPPO (marl package).
They only need jax, flax, optax, numpy, scipy and pillow, and run on CPU.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl.config import EnvConfig, TrainConfig
from marl.curriculum import Curriculum
from marl.env import EnvParams, MultiMowerEnv, total_variation
from marl.maps import RANDOM_GROUP, build_bank, canvas_size_for, random_map
from marl.networks import GroupedConv

# Small canvas for fast tests: 64 = 4 ** 3 pixels = 2.4 m
SMALL = dict(canvas_size=64, global_size=16, position_noise=0, heading_noise=0, lidar_noise=0)


def empty_map(size=64):
    return np.zeros((size, size), dtype=bool), np.zeros((size, size), dtype=bool)


def make_env(maps, **kwargs):
    cfg = EnvConfig(**{**SMALL, **kwargs})
    bank = build_bank(maps, cfg)
    params = EnvParams(bank=bank, map_logits=jnp.zeros(bank.num_maps), goal_coverage=jnp.float32(0.99))
    return MultiMowerEnv(cfg), params


def set_poses(env, params, pos, heading, key=0):
    _, state = env.reset(jax.random.PRNGKey(key), params)
    pos = jnp.asarray(pos, dtype=jnp.float32)
    heading = jnp.asarray(heading, dtype=jnp.float32)
    state = state.replace(pos=pos, prev_pos=pos, heading=heading,
                          overlap=jnp.zeros_like(state.overlap), coverage_px=jnp.zeros(()))
    dist, _, _ = env._lidar(pos, heading, state.obstacles, state.size)
    return state.replace(lidar_dist=dist)


def reference_total_variation(img, img2):
    """Copy of rlm.utils.total_variation (mode sym-iso)"""
    diff1 = np.abs(img[1:, :] - img[:-1, :])
    diff2 = np.abs(img[:, 1:] - img[:, :-1])
    img2 = np.maximum(img, img2)
    diff1 = np.minimum(diff1, np.abs(img2[1:, :] - img2[:-1, :]))
    diff2 = np.minimum(diff2, np.abs(img2[:, 1:] - img2[:, :-1]))
    tv = np.sum(np.sqrt(diff1[:, 1:] ** 2 + diff2[1:, :] ** 2)) + \
         np.sum(np.sqrt(diff1[:, 1:] ** 2 + diff2[:-1, :] ** 2)) + \
         np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2)) + \
         np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[1:, :] ** 2))
    return tv / 4


def test_total_variation():
    rng = np.random.default_rng(0)
    img = (rng.uniform(size=(20, 20)) > 0.5).astype(np.float32)
    img2 = (rng.uniform(size=(20, 20)) > 0.8).astype(np.float32)
    assert np.isclose(float(total_variation(img, img2)), reference_total_variation(img, img2), rtol=1e-5)


def test_grouped_conv_equals_feature_group_count():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 9, 9, 8))
    conv = GroupedConv(16, (3, 3), strides=1, groups=4)
    params = conv.init(jax.random.PRNGKey(1), x)
    y = conv.apply(params, x)
    kernel = params['params']['kernel']                     # 3 x 3 x 8 x 16
    # kernel of the equivalent grouped convolution: 3 x 3 x 2 x 16
    grouped = jnp.concatenate([kernel[:, :, 2*g:2*g+2, 4*g:4*g+4] for g in range(4)], axis=-1)
    y_ref = jax.lax.conv_general_dilated(
        x, grouped, (1, 1), 'VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=4) + params['params']['bias']
    assert np.allclose(y, y_ref, atol=1e-5)


def test_reset():
    rng = np.random.default_rng(0)
    maps = [random_map(rng, EnvConfig(**SMALL), min_size_p=64, max_size_p=64)[:2] for _ in range(3)]
    env, params = make_env(maps, num_agents=3)
    for k in range(5):
        obs, state = env.reset(jax.random.PRNGKey(k), params)
        cfg = env.cfg
        assert obs['maps'].shape == (3, 32, 32, cfg.obs_channels) and obs['maps'].dtype == jnp.uint8
        assert obs['lidar'].shape == (3, cfg.lidar_rays)
        assert obs['global'].shape == (16, 16, 4)
        assert obs['agent'].shape == (3, 6)
        pix = np.floor(np.array(state.pos) * cfg.pixels_per_meter).astype(int)
        assert not np.array(state.blocked)[pix[:, 0], pix[:, 1]].any()
        d = np.linalg.norm(np.array(state.pos)[:, None] - np.array(state.pos)[None], axis=-1)
        assert (d[~np.eye(3, dtype=bool)] >= cfg.robot_separation - 1e-5).all()
        assert float(state.coverage_px) > 0


def test_egocentric_orientation():
    """Up in the observation is forward, columns increase to the right of the robot"""
    known, unknown = empty_map()
    known[40:44, 28:36] = True      # ahead of the robot (+x)
    known[30:34, 20:24] = True      # to the right of the robot (-y)
    env, params = make_env([(known, unknown)], num_agents=1, all_unknown=False)
    state = set_poses(env, params, [[32 / 26.6667, 32 / 26.6667]], [0.0])
    obs = env._observation(jax.random.PRNGKey(0), state)
    obstacles = np.array(obs['maps'][0, :, :, 1]) > 127      # scale 0, obstacle channel
    rows, cols = np.nonzero(obstacles)
    ahead = cols < 22
    assert (rows[ahead] < 10).all() and (rows[ahead] > 0).all() and ahead.sum() > 0
    right = ~ahead
    assert right.sum() > 0 and (cols[right] > 22).all() and (abs(rows[right] - 15.5) < 4).all()


def test_coverage_and_metrics_invariants():
    rng = np.random.default_rng(1)
    cfg = EnvConfig(**SMALL)
    maps = [random_map(rng, cfg, min_size_p=64, max_size_p=64)[:2] for _ in range(2)]
    env, params = make_env(maps, num_agents=3)
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    key = jax.random.PRNGKey(1)
    for _ in range(30):
        key, k1, k2 = jax.random.split(key, 3)
        actions = jax.random.uniform(k1, (3, 1), minval=-1, maxval=1)
        obs, state, rewards, done, info = env.step(k2, state, actions, params)
        assert not bool(done)
        overlap = np.array(state.overlap)
        free = ~np.array(state.dilated)
        assert np.isclose(float(state.coverage_px), ((overlap > 0) & free).sum(), atol=1e-3)
        assert np.isclose(float(state.overlap_px), overlap[free].sum() - float(state.coverage_px), atol=1e-3)
        assert not (np.array(state.known) & ~np.array(state.obstacles)).any()
        assert not ((overlap > 0) & np.array(state.obstacles)).any()
        assert np.isfinite(np.array(rewards)).all()


def test_straight_driving_covers_max_area():
    env, params = make_env([empty_map()], num_agents=1)
    state = set_poses(env, params, [[0.5, 1.2]], [0.0])
    covered = []
    for k in range(6):
        _, state, _, _, _ = env.step(jax.random.PRNGKey(k), state, jnp.zeros((1, 1)), params)
        covered.append(float(state.coverage_px))
    per_step = np.diff(covered[1:]) / env.cfg.max_newly_visited
    assert np.allclose(per_step, 1, atol=0.2)
    assert np.isclose(float(state.pos[0, 0]), 0.5 + 6 * env.cfg.max_lin_vel * env.cfg.step_size, atol=1e-4)


def test_shared_coverage_is_split():
    """Two robots on top of the same path: newly covered area is split, team reward unchanged"""
    env, params = make_env([empty_map()], num_agents=2, robot_collision_reward=0, team_reward_weight=0)
    state = set_poses(env, params, [[0.5, 1.2], [0.5, 1.2]], [0.0, 0.0])
    cov = env._coverage_update(state, state.pos + jnp.array([[0.13, 0.0], [0.13, 0.0]]))
    assert np.isclose(float(cov['new_share'][0]), float(cov['new_share'][1]))
    assert np.isclose(float(cov['team_new']), float(cov['new_share'].sum()))
    single, _ = make_env([empty_map()], num_agents=1)
    s1 = set_poses(single, params, [[0.5, 1.2]], [0.0])
    cov1 = single._coverage_update(s1, s1.pos + jnp.array([[0.13, 0.0]]))
    assert np.isclose(float(cov['team_new']), float(cov1['team_new']))


def test_robot_collision():
    env, params = make_env([empty_map()], num_agents=2)
    r = env.cfg.mower_radius
    pos = [[1.0, 1.2], [1.0 + 2 * r + 0.1, 1.2]]
    state = set_poses(env, params, pos, [0.0, math.pi])
    _, state, rewards, _, _ = env.step(jax.random.PRNGKey(0), state, jnp.zeros((2, 1)), params)
    assert np.allclose(np.array(state.pos), pos)
    assert (np.array(state.collisions) == 1).all()
    assert (np.array(rewards) < env.cfg.robot_collision_reward + 1).all()


def test_lidar_detects_robots():
    env, params = make_env([empty_map()], num_agents=2)
    state = set_poses(env, params, [[0.9, 1.2], [1.9, 1.2]], [0.0, 0.0])
    dist = np.array(state.lidar_dist[0]) / env.cfg.pixels_per_meter
    # rays at -7.5 and 7.5 degrees hit the other robot
    expected = 1.0 * math.cos(math.radians(7.5)) - math.sqrt(0.15 ** 2 - (1.0 * math.sin(math.radians(7.5))) ** 2)
    assert np.allclose(dist[11:13], expected, atol=0.01)
    # robots are not added to the known obstacles (only the map border is known)
    assert np.array(state.known).any()
    assert not (np.array(state.known) & ~np.array(state.obstacles)).any()


def test_auto_reset():
    env, params = make_env([empty_map()], num_agents=2, max_episode_steps=3)
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    for k in range(3):
        obs, state, _, done, info = env.step(jax.random.PRNGKey(k), state, jnp.zeros((2, 1)), params)
    assert bool(done) and int(info['length']) == 3
    assert int(state.t) == 0


def test_vmap_step():
    env, params = make_env([empty_map(), empty_map(48)], num_agents=2)
    E = 4
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(0), E), params)
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    obs, state, rewards, done, info = step(jax.random.split(jax.random.PRNGKey(1), E), state, jnp.zeros((E, 2, 1)), params)
    assert rewards.shape == (E, 2) and done.shape == (E,)
    assert obs['maps'].shape == (E, 2, 32, 32, env.cfg.obs_channels)


def test_canvas_size_for():
    cfg = EnvConfig()
    assert canvas_size_for([128, 320], cfg) == 320
    assert canvas_size_for([800], cfg) == 832


def test_curriculum():
    groups = [0, 0, 1, RANDOM_GROUP, RANDOM_GROUP]
    cur = Curriculum(groups, [False, False, False, True, False], [False, False, False, False, True])
    p = np.exp(cur.map_logits())
    assert np.allclose(p, [0.5, 0.5, 0, 0, 0])
    assert not cur.update([0, 1], [True, False])
    assert cur.update([1], [True])
    assert cur.level == 2
    cur = Curriculum(groups, [False, False, False, True, False], [False, False, False, False, True], start_level=8)
    assert np.isclose(np.exp(cur.map_logits())[3:].sum(), 0.5)


@pytest.mark.slow
def test_mappo_update():
    from marl.mappo import MAPPO
    env_cfg = EnvConfig(**SMALL, num_agents=2, max_episode_steps=8)
    train_cfg = TrainConfig(num_envs=2, num_steps=8, total_steps=64, minibatch_size=16, update_epochs=2, hidden_dims=32)
    bank = build_bank([empty_map(), empty_map(48)], env_cfg)
    params = EnvParams(bank=bank, map_logits=jnp.zeros(2), goal_coverage=jnp.float32(0.9))
    algo = MAPPO(env_cfg, train_cfg)
    runner = algo.init(jax.random.PRNGKey(0), params)
    before = jax.tree_util.tree_map(lambda x: np.array(x), runner.actor.params)
    runner, metrics, infos = algo.update(runner, params)
    for k, v in metrics.items():
        assert np.isfinite(float(v)), k
    assert infos['done'].shape == (8, 2) and np.array(infos['done']).sum() == 2
    changed = jax.tree_util.tree_map(lambda a, b: not np.allclose(a, np.array(b)), before, runner.actor.params)
    assert any(jax.tree_util.tree_leaves(changed))
