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
                          overlap=jnp.zeros_like(state.overlap), coverage_px=jnp.zeros(()),
                          belief_overlap=jnp.zeros_like(state.belief_overlap), connected=env._connectivity(pos),
                          work=params.bank.cell_work[state.map_idx])
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
        assert obs['global'].shape == (16, 16, cfg.global_channels) == (16, 16, 3)
        assert obs['agent'].shape == (3, 6)
        assert obs['others'].shape == (3, 3, cfg.neighbor_features)
        assert obs['team'].shape == (3, 3, cfg.team_features)
        assert np.array_equal(np.array(obs['others_mask']), ~np.eye(3, dtype=bool))
        pix = np.floor(np.array(state.pos) * cfg.pixels_per_meter).astype(int)
        assert not np.array(state.blocked)[pix[:, 0], pix[:, 1]].any()
        d = np.linalg.norm(np.array(state.pos)[:, None] - np.array(state.pos)[None], axis=-1)
        assert (d[~np.eye(3, dtype=bool)] >= cfg.robot_separation - 1e-5).all()
        assert float(state.coverage_px) > 0


@pytest.mark.parametrize('size,G', [(64, 16), (48, 16), (37, 8), (20, 16)])
def test_area_resample(size, G):
    """Exact area average of the map on G x G cells, also for cells that are not whole pixels"""
    from marl.env import area_resample, avg_pool_hwc
    rng = np.random.default_rng(size)
    x = rng.uniform(size=(64, 64, 2)).astype(np.float32)
    out = np.array(area_resample(jnp.asarray(x), jnp.int32(size), G))
    # reference: upsample by G (each pixel G x G), then blocks of size x size
    up = np.repeat(np.repeat(x[:size, :size], G, axis=0), G, axis=1)
    ref = up.reshape(G, size, G, size, 2).mean(axis=(1, 3))
    assert np.allclose(out, ref, atol=1e-4)
    if size == 64:
        assert np.allclose(out, np.array(avg_pool_hwc(jnp.asarray(x), 64 // G)), atol=1e-5)


def test_global_crop():
    """The global maps cover only the map, positions relative to the map size, map side in the progress"""
    rng = np.random.default_rng(0)
    cfg = EnvConfig(**SMALL)
    maps = [random_map(rng, cfg, min_size_p=40, max_size_p=40)[:2]]
    canvas, params = make_env(maps, num_agents=2)
    crop, _ = make_env(maps, num_agents=2, global_crop=True)
    assert crop.cfg.progress_features == 3
    obs_c, state = canvas.reset(jax.random.PRNGKey(0), params)
    obs, _ = crop.reset(jax.random.PRNGKey(0), params)
    for k in ('maps', 'lidar', 'others', 'others_mask'):
        assert np.array_equal(np.array(obs[k]), np.array(obs_c[k])), k
    pos_p = np.array(state.pos) * cfg.pixels_per_meter
    assert np.allclose(np.array(obs['agent'])[:, :2], pos_p / 40, atol=1e-5)
    assert np.allclose(np.array(obs_c['agent'])[:, :2], pos_p / 64, atol=1e-5)
    assert np.allclose(np.array(obs['team'])[0, :, -2:], pos_p / 40, atol=1e-5)
    assert np.allclose(np.array(obs['progress'])[-1], 40 * cfg.meters_per_pixel / 10)
    # obstacle channel: the true obstacles of the map only, no cells outside of it
    obstacles = np.array(state.obstacles)[:40, :40].astype(np.float32)[..., None]
    ref = np.repeat(np.repeat(obstacles, 16, axis=0), 16, axis=1).reshape(16, 40, 16, 40).mean(axis=(1, 3))
    assert np.abs(np.array(obs['global'])[..., 1] / 255 - ref).max() <= 1 / 255
    assert np.array(obs_c['global'])[12:, 12:, 1].min() == 255                  # canvas: outside is blocked


def test_geodesic_voronoi_distances():
    """One seed on a free grid: octile distances (8-connected shortest paths)"""
    from marl.env import geodesic_voronoi
    free = jnp.ones((12, 12), dtype=bool)
    dist, owner = geodesic_voronoi(jnp.array([[3, 5]], dtype=jnp.int32), free)
    di, dj = np.abs(np.mgrid[:12, :12] - np.array([3, 5])[:, None, None])
    ref = np.maximum(di, dj) + (np.sqrt(2) - 1) * np.minimum(di, dj)
    assert np.allclose(np.array(dist), ref, atol=1e-4) and (np.array(owner) == 0).all()


def test_geodesic_voronoi_wall():
    """A cell just behind a wall belongs to the robot that walks there first, not to the closest in a straight line"""
    from marl.env import geodesic_voronoi
    free = np.ones((16, 16), dtype=bool)
    free[:12, 8] = False                                     # wall, door in rows 12-15
    seeds = jnp.array([[2, 6], [10, 14]], dtype=jnp.int32)
    dist, owner = geodesic_voronoi(seeds, jnp.asarray(free))
    owner = np.array(owner)
    assert owner[2, 10] == 1 and owner[2, 4] == 0            # behind the wall: robot 1, straight line: robot 0
    # far from the door each side of the wall has its robot (next to the door, robot 1 walks through it first)
    assert (owner[:12, 9:] == 1).all() and (owner[:9, :8] == 0).all() and owner[11, 7] == 1
    assert (owner >= 0).all() and np.isfinite(np.array(dist)[free]).all()
    # no corner cutting: a diagonal wall of single cells is not crossed
    free = np.ones((8, 8), dtype=bool)
    free[np.arange(8), 7 - np.arange(8)] = False
    dist, owner = geodesic_voronoi(jnp.array([[0, 0]], dtype=jnp.int32), jnp.asarray(free))
    beyond = np.add.outer(np.arange(8), np.arange(8)) > 7
    assert np.isinf(np.array(dist)[beyond & free]).all()
    assert (np.array(owner) == 0).all()                      # not reached: the closest seed in a straight line
    # paths longer than max_sweeps: the cells not reached get the closest seed in a straight line
    free = np.ones((16, 16), dtype=bool)
    dist, owner = geodesic_voronoi(jnp.array([[0, 0], [0, 3]], dtype=jnp.int32), jnp.asarray(free), max_sweeps=4)
    dist, owner = np.array(dist), np.array(owner)
    assert np.isinf(dist[15, 15]) and owner[15, 15] == 1 and owner[15, 0] == 0


def wall_map():
    """64 pixel map with a wall (columns 30-33) from the top to row 47, door in rows 48-63"""
    known = np.zeros((64, 64), dtype=bool)
    known[:48, 30:34] = True
    return known, np.zeros_like(known)


def test_workload():
    env, params = make_env([wall_map()], num_agents=2, workload_size=16)
    ppm = env.cfg.pixels_per_meter
    state = env._update_workload(set_poses(env, params, np.array([[10, 20], [56, 50]]) / ppm, [0.0, 0.0]))
    owner = np.array(state.owner)
    assert owner.shape == (16, 16) and owner[2, 10] == 1 and owner[2, 2] == 0   # behind the wall: robot 1
    # the workloads add up to the free area not yet covered
    workload = np.array(state.workload)
    assert np.isclose(workload.sum(), float(state.free_space) / ppm ** 2, rtol=1e-4)
    assert workload[1] > workload[0] > 0
    # robot order: the workloads are permuted
    swapped = env._update_workload(set_poses(env, params, np.array([[56, 50], [10, 20]]) / ppm, [0.0, 0.0]))
    assert np.allclose(np.array(swapped.workload), workload[::-1], rtol=1e-5)
    # after steps: the workloads add up to the free area not yet covered, which decreases
    for k in range(5):
        _, state, _, _, _ = env.step(jax.random.PRNGKey(k), state, jnp.zeros((2, 1)), params)
        remaining = (float(state.free_space) - float(state.coverage_px)) / ppm ** 2
        assert np.isclose(np.array(state.workload).sum(), remaining, rtol=1e-4)
    assert np.array(state.workload).sum() < workload.sum()
    # off by default
    env, params = make_env([wall_map()], num_agents=2)
    _, state = env.reset(jax.random.PRNGKey(0), params)
    assert state.owner.shape == (0, 0) and not np.array(state.workload).any()


@pytest.mark.parametrize('comm_range', [0.0, 1.0])
def test_critic_workload(comm_range):
    """Privileged features of the critic: workloads, max workload, unknown coverage, connected robots"""
    env, params = make_env([wall_map()], num_agents=3, workload_size=16, critic_workload=True, comm_range=comm_range)
    cfg = env.cfg
    assert cfg.agent_features == 6 + (4 if comm_range else 2) and cfg.progress_features == 3
    unit = cfg.max_newly_visited * cfg.meters_per_pixel ** 2
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    for k in range(30):
        obs, state, _, _, _ = env.step(jax.random.PRNGKey(k), state, jnp.ones((3, 1)), params)
        agent = np.array(obs['agent'])
        assert agent.shape == (3, cfg.agent_features)
        workload = np.array(state.workload)
        assert np.allclose(agent[:, 6] * 1000 * unit, workload, rtol=1e-4)
        assert np.isclose(agent[:, 7].sum(), 1, atol=1e-5)
        assert np.isclose(float(obs['progress'][-1]) * 1000 * unit, workload.max(), rtol=1e-4)
        if comm_range:
            assert ((agent[:, 8] >= 0) & (agent[:, 8] <= 1)).all()
            connected = (np.array(state.connected) & ~np.eye(3, dtype=bool)).sum(1) / 2
            assert np.allclose(agent[:, 9], connected)
    if comm_range:
        # a robot does not know what the robots it is not connected to have covered
        conn = np.array(state.connected)
        unknown = np.array([((np.array(state.overlap) > 0) & (np.array(state.belief_overlap[i]) == 0)).sum()
                            for i in range(3)]) / float(state.free_space)
        assert np.allclose(agent[:, 8], unknown, atol=1e-6)
        if not conn.all():
            assert agent[:, 8].max() > 0
    # off: the agent features are unchanged
    env, params = make_env([wall_map()], num_agents=3, workload_size=16)
    assert env.cfg.agent_features == 6 and env.reset(jax.random.PRNGKey(0), params)[0]['agent'].shape == (3, 6)


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


def test_workload_shaping_telescopes():
    """With gamma 1 the shaping of an episode adds up to -phi(s0): the initial maximum workload (phi = 0 at the end)"""
    from marl.env import REWARD_TERMS
    k = REWARD_TERMS.index('workload')
    env, params = make_env([wall_map()], num_agents=2, workload_size=16, workload_reward_scale=0.3,
                           shaping_gamma=1.0, max_episode_steps=12)
    _, state = env.reset(jax.random.PRNGKey(0), params)
    initial = 0.3 * float(env._max_workload(state))
    assert initial > 0
    total = 0.0
    key = jax.random.PRNGKey(1)
    for i in range(12):
        key, k1, k2 = jax.random.split(key, 3)
        actions = jax.random.uniform(k1, (2, 1), minval=-1, maxval=1)
        _, state, rewards, done, info = env.step(k2, state, actions, params)
        total += float(np.mean(np.array(rewards)))
    assert bool(done)
    assert np.isclose(float(info['terms'][k]), initial, rtol=1e-4)
    # the terms add up to the return of the episode
    assert np.isclose(float(np.sum(info['terms'])), float(info['ep_return']), rtol=1e-4)
    assert np.isclose(float(info['ep_return']), total, rtol=1e-4)


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


def test_robot_collision_only_when_touching():
    """
    Robot-robot collisions only when the circles touch during the step:
    following a robot into the place it leaves is not a collision, crossing
    paths at the same time is one even if the end positions are apart
    """
    env, params = make_env([empty_map()], num_agents=2)
    r = env.cfg.mower_radius
    step = env.cfg.max_lin_vel * env.cfg.step_size
    # robot 0 follows robot 1 at 2 r + 0.5 step: its new position is near the old one of robot 1
    state = set_poses(env, params, [[1.0, 1.2], [1.0 + 2 * r + step / 2, 1.2]], [0.0, 0.0])
    state, _, _, _ = env._step_state(jax.random.PRNGKey(0), state, jnp.zeros((2, 1)))
    assert (np.array(state.collisions) == 0).all()
    # crossing paths (robot 0 along +x, robot 1 along -y): start and end more
    # than 2 r apart, but closer than 2 r in between
    s = step / math.sqrt(2)                               # half the relative motion, along (1, 1)
    a = np.array([-s + 0.29, -s - 0.29]) / math.sqrt(2)   # p0 - p1, 0.29 m off the relative path
    p0 = np.array([1.0, 1.2])
    state = set_poses(env, params, [p0, p0 - a], [0.0, -math.pi / 2])
    pos0 = np.array(state.pos)
    d = lambda q: float(np.linalg.norm(q[0] - q[1]))
    assert d(pos0) > 2 * r and d(pos0 + np.array([[step, 0], [0, -step]])) > 2 * r
    state, _, _, _ = env._step_state(jax.random.PRNGKey(0), state, jnp.zeros((2, 1)))
    assert (np.array(state.collisions) == 1).all() and np.allclose(np.array(state.pos), pos0)


def test_robot_collision_cascade():
    """A robot that collides stays still: a robot running into it collides too"""
    env, params = make_env([empty_map()], num_agents=2)
    r = env.cfg.mower_radius
    step = env.cfg.max_lin_vel * env.cfg.step_size
    # robot 1 drives into the wall at x = 0, robot 0 follows it closely
    x1 = r + step / 2
    state = set_poses(env, params, [[x1 + 2 * r + step / 2, 1.2], [x1, 1.2]], [math.pi, math.pi])
    state, _, _, _ = env._step_state(jax.random.PRNGKey(0), state, jnp.zeros((2, 1)))
    assert (np.array(state.collisions) == 1).all()
    assert int(state.robot_collisions[0]) == 1 and int(state.robot_collisions[1]) == 0


def test_flip_when_stuck():
    """Two robots driving into each other: deadlock, unless they are flipped after max_stuck_steps"""
    for flip in (False, True):
        env, params = make_env([empty_map()], num_agents=2, flip_when_stuck=flip, max_stuck_steps=3)
        state = set_poses(env, params, [[1.0, 1.0], [1.0, 1.4]], [jnp.pi / 2, -jnp.pi / 2])
        for k in range(4):
            state, _, _, _ = env._step_state(jax.random.PRNGKey(k), state, jnp.zeros((2, 1)))
        d = float(jnp.linalg.norm(state.pos[0] - state.pos[1]))
        assert (d > 0.4) == flip and int(state.robot_collisions[0]) == (3 if flip else 4)


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


@pytest.mark.parametrize('comm_range,workload_size', [(0.0, 0), (1.0, 0), (0.0, 16)])
def test_step_batch_equals_vmap_step(comm_range, workload_size):
    """step_batch (reset under lax.cond) gives exactly the results of vmap(step), with and without resets"""
    rewards = dict(workload_reward_scale=0.5) if workload_size else {}
    env, params = make_env([empty_map(), wall_map()], num_agents=2, max_episode_steps=3, comm_range=comm_range,
                           workload_size=workload_size, **rewards)
    E = 4
    _, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(0), E), params)
    # different episode lengths, so that only some environments are reset in a step
    state = state.replace(t=jnp.arange(E, dtype=jnp.int32) % 3)
    vstep = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    bstep = jax.jit(env.step_batch)
    for k in range(4):
        keys = jax.random.split(jax.random.PRNGKey(k), E)
        actions = jax.random.uniform(jax.random.PRNGKey(10 + k), (E, 2, 1), minval=-1, maxval=1)
        a, b = vstep(keys, state, actions, params), bstep(keys, state, actions, params)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        state = a[1]


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
@pytest.mark.parametrize('algo,critic_state,comm_range,gather,workload', [
    ('mappo', 'global', 0, False, False), ('mappo', 'agent_specific', 0, False, False),
    ('ippo', 'global', 0, False, False), ('mappo', 'global', 1.0, False, False), ('mappo', 'global', 0, True, False),
    ('mappo', 'agent_specific', 0, True, False), ('mappo', 'global', 1.0, False, True),
    ('mappo', 'agent_specific', 1.0, True, True)])
def test_mappo_update(algo, critic_state, comm_range, gather, workload):
    from marl.mappo import MAPPO
    coop = dict(workload_size=16, critic_workload=True, workload_reward_scale=0.02,
                shaping_gamma=TrainConfig().gamma) if workload else {}
    env_cfg = EnvConfig(**SMALL, num_agents=2, max_episode_steps=8, comm_range=comm_range, global_crop=gather, **coop)
    train_cfg = TrainConfig(algo=algo, critic_state=critic_state, num_envs=2, num_steps=8, total_steps=64, minibatch_size=16, update_epochs=2,
                            hidden_dims=32, critic_gather=gather)
    bank = build_bank([empty_map(), empty_map(48)], env_cfg)
    params = EnvParams(bank=bank, map_logits=jnp.zeros(2), goal_coverage=jnp.float32(0.9))
    algo = MAPPO(env_cfg, train_cfg)
    runner = algo.init(jax.random.PRNGKey(0), params)
    before = jax.tree_util.tree_map(lambda x: np.array(x), runner.actor.params)
    runner, metrics, infos = algo.update(runner, params)
    for k, v in metrics.items():
        assert np.isfinite(float(v)), k
    assert float(metrics['explained_variance']) <= 1
    assert infos['done'].shape == (8, 2) and np.array(infos['done']).sum() == 2
    changed = jax.tree_util.tree_map(lambda a, b: not np.allclose(a, np.array(b)), before, runner.actor.params)
    assert any(jax.tree_util.tree_leaves(changed))


def test_comm_range_covering_the_map_equals_shared_map():
    """With a range larger than the map, the beliefs equal the shared map (binary coverage observation)"""
    rng = np.random.default_rng(2)
    cfg = EnvConfig(**SMALL)
    maps = [random_map(rng, cfg, min_size_p=64, max_size_p=64)[:2] for _ in range(2)]
    shared, params = make_env(maps, num_agents=3, overlap_observation=False)
    limited, _ = make_env(maps, num_agents=3, overlap_observation=False, comm_range=100)
    obs_s, state_s = shared.reset(jax.random.PRNGKey(0), params)
    obs_l, state_l = limited.reset(jax.random.PRNGKey(0), params)
    assert state_s.belief_overlap.shape == (3, 0, 0) and state_l.belief_overlap.shape == (3, 64, 64)
    key = jax.random.PRNGKey(1)
    for _ in range(20):
        for k in obs_s:
            assert np.array_equal(np.array(obs_s[k]), np.array(obs_l[k])), k
        key, k1, k2 = jax.random.split(key, 3)
        actions = jax.random.uniform(k1, (3, 1), minval=-1, maxval=1)
        obs_s, state_s, r_s, _, _ = shared.step(k2, state_s, actions, params)
        obs_l, state_l, r_l, _, _ = limited.step(k2, state_l, actions, params)
        assert np.allclose(np.array(r_s), np.array(r_l))


def test_limited_comm_beliefs():
    env, params = make_env([empty_map()], num_agents=2, comm_range=0.5)
    state = set_poses(env, params, [[0.5, 0.6], [1.4, 1.8]], [0.0, 0.0])
    assert not bool(state.connected[0, 1])
    for k in range(4):
        obs, state, _, _, _ = env.step(jax.random.PRNGKey(k), state, jnp.zeros((2, 1)), params)
    covered = np.array(state.overlap) > 0
    belief = np.array(state.belief_overlap) > 0
    # each robot only knows its own coverage, together they know everything
    assert belief[0].any() and belief[1].any() and not (belief[0] & belief[1]).any()
    assert np.array_equal(belief[0] | belief[1], covered)
    assert not (np.array(state.belief_known) & ~np.array(state.known)).any()
    # the robots do not observe each other, the critic still sees both
    assert not np.array(obs['others_mask']).any()
    assert np.abs(np.array(obs['team'])[0, 1]).sum() > 0

    # within range, the beliefs are merged (OR) and the robots see each other
    pos = jnp.array([[1.0, 1.0], [1.0, 1.4]])
    state = state.replace(pos=pos, prev_pos=pos, connected=env._connectivity(pos))
    obs, state, _, _, _ = env.step(jax.random.PRNGKey(9), state, jnp.zeros((2, 1)), params)
    belief = np.array(state.belief_overlap) > 0
    assert np.array_equal(belief[0], belief[1])
    assert np.array_equal(belief[0], np.array(state.overlap) > 0)
    assert np.array_equal(np.array(state.belief_known[0]), np.array(state.known))
    assert np.array_equal(np.array(obs['others_mask']), ~np.eye(2, dtype=bool))


def test_legacy_robot_channel():
    """Without neighbor_observation: robot map channel and global robot channel, as in old checkpoints"""
    env, params = make_env([empty_map()], num_agents=2, robot_observation=True, neighbor_observation=False)
    obs, _ = env.reset(jax.random.PRNGKey(0), params)
    assert obs['maps'].shape[-1] == env.cfg.obs_channels == 4 * 4
    assert obs['global'].shape[-1] == 4 and 'others' not in obs


def test_relative_poses():
    """Robot 1 ahead (+x) of robot 0 (heading 0) and robot 2 to its right (-y)"""
    env, params = make_env([empty_map()], num_agents=3)
    state = set_poses(env, params, [[1.0, 1.2], [1.6, 1.2], [1.0, 0.6]], [0.0, jnp.pi / 2, 0.0])
    others = np.array(env._observation(jax.random.PRNGKey(0), state)['others'])
    assert np.allclose(others[0, 1, :2], [1, 0], atol=1e-5)          # ahead
    assert np.allclose(others[0, 2, :2], [0, 1], atol=1e-5)          # right
    assert np.allclose(others[0, 1, -2:], [0, 1], atol=1e-5)         # turned left by 90 degrees
    # closer robots are closer at every scale, closeness decreases with the scale
    assert (others[0, 1, 2:-2] > 0).all() and (np.diff(others[0, 1, 2:-2]) < 0).all()


@pytest.mark.parametrize('algo,critic_state,gather', [
    ('mappo', 'global', False), ('mappo', 'agent_specific', False), ('ippo', 'global', False),
    ('mappo', 'global', True), ('mappo', 'agent_specific', True)])
def test_networks_agnostic_to_number_of_robots(algo, critic_state, gather):
    """The same parameters work for any number of robots, invariant to the order of the robots"""
    from marl.mappo import actor_apply, critic_value, init_params, make_networks
    train_cfg = TrainConfig(algo=algo, critic_state=critic_state, hidden_dims=32, neighbor_dims=16, critic_gather=gather)
    rng = np.random.default_rng(0)
    params = None
    for N in (1, 2, 5):
        maps = [random_map(rng, EnvConfig(**SMALL), min_size_p=64, max_size_p=64)[:2]]
        env, env_params = make_env(maps, num_agents=N, robot_separation=0.3, global_crop=gather)
        actor, critic = make_networks(env.cfg, train_cfg)
        if params is None:
            params = init_params(env.cfg, actor, critic, jax.random.PRNGKey(0))
        obs, _ = env.reset(jax.random.PRNGKey(N), env_params)
        mean, _ = actor_apply(actor, params[0], obs)
        value = critic_value(critic, params[1], jax.tree_util.tree_map(lambda x: x[None], obs))
        assert mean.shape == (N, env.cfg.action_dim) and value.shape == (1, N)
        assert np.isfinite(np.array(mean)).all() and np.isfinite(np.array(value)).all()
        # permuting the other robots (columns of the sets) changes nothing
        perm = rng.permutation(N)
        permuted = dict(obs, others=obs['others'][:, perm], others_mask=obs['others_mask'][:, perm])
        assert np.allclose(actor_apply(actor, params[0], permuted)[0], mean, atol=1e-5)
        # permuting the robots permutes their values
        robots = dict(obs, **{k: obs[k][perm] for k in ('maps', 'lidar', 'agent')},
                      **{k: obs[k][perm][:, perm] for k in ('others', 'others_mask', 'team')})
        permuted_value = critic_value(critic, params[1], jax.tree_util.tree_map(lambda x: x[None], robots))
        assert np.allclose(permuted_value, value[:, perm], atol=1e-4)


def test_connectivity_is_transitive():
    env, _ = make_env([empty_map()], num_agents=4, comm_range=0.5)
    pos = jnp.array([[0.2, 1.0], [0.6, 1.0], [1.0, 1.0], [2.0, 1.0]])
    conn = np.array(env._connectivity(pos))
    assert conn[:3, :3].all()
    assert not conn[3, :3].any() and not conn[:3, 3].any() and conn[3, 3]


def test_stats_compare():
    from marl.stats import compare, holm, mean_ci
    S, M, n = 5, 100, 300
    covered = false_pos = detected = 0
    for k in range(n):
        rng = np.random.default_rng(k)
        maps = rng.uniform(100, 1000, M)                       # hard and easy maps
        run = lambda scale: scale * maps + rng.normal(0, 30, (S, 1)) + rng.normal(0, 20, (S, M))
        A, B, C = run(1.0), run(1.1), run(1.0)                 # B is 10 % slower, C as A
        r = compare(A, B)
        covered += r['diff_ci'][0] <= 0.1 * 550 <= r['diff_ci'][1]
        detected += r['p'] < 0.05 and r['p_better'] < 0.5
        false_pos += compare(A, C)['p'] < 0.05
    # with 5 seeds a bootstrap gives about 14 % false positives, see marl.stats.
    # The power to detect the 10 % difference is about 2/3, limited by the
    # variance between seeds (5 % of the mean)
    assert covered / n >= 0.92 and false_pos / n <= 0.08 and detected / n >= 0.5
    est, (lo, hi) = mean_ci(A)
    assert lo < est < hi
    # nan (not defined) values are ignored, one seed gives no p-value
    A[0, :10] = np.nan
    assert np.isfinite(compare(A, B)['p']) and np.isnan(compare(A[:1], B[:1])['p'])
    assert np.allclose(holm([0.01, 0.04, 0.03, np.nan]), [0.03, 0.06, 0.06, np.nan], equal_nan=True)


def test_adaptive_actor_lr():
    """The adaptive learning rate goes down with a large KL, up with a small one, within its bounds"""
    from marl.mappo import MAPPO
    env_cfg = EnvConfig(**SMALL, num_agents=2, max_episode_steps=8)
    train_cfg = TrainConfig(num_envs=2, num_steps=8, total_steps=64, minibatch_size=16, update_epochs=2,
                            hidden_dims=32, adaptive_lr=True, actor_lr=1e-4, target_kl=0.02, max_actor_lr=2e-4)
    bank = build_bank([empty_map(), empty_map(48)], env_cfg)
    params = EnvParams(bank=bank, map_logits=jnp.zeros(2), goal_coverage=jnp.float32(0.9))
    algo = MAPPO(env_cfg, train_cfg)
    runner = algo.init(jax.random.PRNGKey(0), params)
    actor = runner.actor
    assert np.isclose(float(algo.actor_lr(actor)), 1e-4)
    assert np.isclose(float(algo.actor_lr(algo._adapt_actor_lr(actor, 0.1))), 1e-4 / 1.5)
    assert np.isclose(float(algo.actor_lr(algo._adapt_actor_lr(actor, 0.02))), 1e-4)
    assert np.isclose(float(algo.actor_lr(algo._adapt_actor_lr(actor, 0.001))), 1.5e-4)
    high = algo._adapt_actor_lr(algo._adapt_actor_lr(actor, 0.0), 0.0)
    assert np.isclose(float(algo.actor_lr(high)), 2e-4)
    runner, metrics, _ = algo.update(runner, params)
    assert 'actor_lr' in metrics and np.isfinite(float(metrics['actor_lr']))
    runner, metrics, _ = algo.update(runner, params, train_actor=False)
    assert float(metrics['actor_updates']) == 0


def test_ippo_from_mappo_checkpoint(tmp_path):
    """IPPO started from a MAPPO checkpoint (e.g. scripts/multi_agent/init_from_sb3.py): same actor, new local critic"""
    from marl.mappo import MAPPO, load_init_params, save_checkpoint
    env_cfg = EnvConfig(**SMALL, num_agents=2, max_episode_steps=8)
    mappo_cfg = TrainConfig(num_envs=2, num_steps=8, total_steps=64, minibatch_size=16, hidden_dims=32)
    bank = build_bank([empty_map()], env_cfg)
    params = EnvParams(bank=bank, map_logits=jnp.zeros(1), goal_coverage=jnp.float32(0.9))
    runner = MAPPO(env_cfg, mappo_cfg).init(jax.random.PRNGKey(3), params)
    save_checkpoint(str(tmp_path), runner, env_cfg, mappo_cfg)

    actor, critic = load_init_params(str(tmp_path), env_cfg, mappo_cfg)
    assert critic is not None
    ippo_cfg = TrainConfig(**{**mappo_cfg.to_dict(), 'algo': 'ippo'})
    actor, critic = load_init_params(str(tmp_path), env_cfg, ippo_cfg)
    assert critic is None
    same = jax.tree_util.tree_map(lambda a, b: np.array_equal(np.array(a), np.array(b)), actor, runner.actor.params)
    assert all(jax.tree_util.tree_leaves(same))
    runner = MAPPO(env_cfg, ippo_cfg).init(jax.random.PRNGKey(0), params, actor, critic)
    assert 'MultiScaleEncoder_0' in runner.critic.params['params']


def test_stop_update(tmp_path):
    """stop_update ends the training early, the last update is logged and saved"""
    import json
    from unittest import mock
    from marl.mappo import train
    env_cfg = EnvConfig(**SMALL, num_agents=2, max_episode_steps=8)
    train_cfg = TrainConfig(num_envs=2, num_steps=8, total_steps=160, stop_update=3, minibatch_size=16,
                            hidden_dims=32, save_every=50, num_random_maps=0)
    maps = (build_bank([empty_map()], env_cfg), None, np.array([0]), np.zeros(1, bool), np.zeros(1, bool))
    with mock.patch('marl.mappo.make_training_maps', lambda *a, **k: maps):
        train(env_cfg, train_cfg, str(tmp_path), verbose=False)
    with open(tmp_path / 'progress.csv') as f:
        assert f.read().splitlines()[-1].startswith('3,')
    with open(tmp_path / 'checkpoint' / 'config.json') as f:
        assert json.load(f)['update'] == 3


def test_config_from(tmp_path):
    """scripts/multi_agent/train.py --config_from: the settings of another run are the defaults, the command line wins"""
    import json
    import sys
    from unittest import mock
    from scripts.multi_agent import train as train_marl
    env_cfg = EnvConfig(num_agents=3, lidar_fov=180)
    train_cfg = TrainConfig(critic_warmup=20, seed=7, adaptive_lr=True)
    with open(tmp_path / 'config.json', 'w') as f:
        json.dump({'env': env_cfg.to_dict(), 'train': train_cfg.to_dict()}, f)
    seen = {}
    argv = ['scripts/multi_agent/train.py', '--config_from', str(tmp_path), '--algo', 'ippo', '--logdir', str(tmp_path / 'run')]
    with mock.patch.object(sys, 'argv', argv), \
            mock.patch('marl.mappo.train', lambda e, t, *a, **k: seen.update(env=e, train=t)):
        train_marl.main()
    assert seen['env'] == env_cfg
    assert seen['train'] == TrainConfig(**{**train_cfg.to_dict(), 'algo': 'ippo'})


@pytest.mark.parametrize('comm_range', [0, 5])
def test_robot_lidar_obstacles(comm_range):
    """With robot_lidar_obstacles the lidar detections of the other robots are in the obstacle channel, only in the observation"""
    obs_channel = 1  # scale 0: coverage, obstacles, frontier
    pos, heading = [[0.95, 1.2], [1.45, 1.2]], [0.0, math.pi]
    maps = {}
    for flag in [False, True]:
        env, params = make_env([empty_map()], num_agents=2, comm_range=comm_range, robot_lidar_obstacles=flag)
        state = set_poses(env, params, pos, heading)
        state = state.replace(known=jnp.zeros_like(state.known), belief_known=jnp.zeros_like(state.belief_known))
        obs = env._observation(jax.random.PRNGKey(0), state)
        maps[flag] = np.array(obs['maps'][..., obs_channel])
    assert not np.any(maps[False])
    H = maps[True].shape[1]
    for n in range(2):
        # the other robot is ahead (up), the robot itself is not an obstacle
        rows, _ = np.nonzero(maps[True][n])
        assert len(rows) > 0 and np.all(rows < H // 2 - 4)


@pytest.mark.parametrize('flag', [False, True])
def test_collision_ends_episode(flag):
    """With collision_ends_episode the first collision ends the episode, as a failure"""
    env, params = make_env([empty_map()], num_agents=2, collision_ends_episode=flag)
    # robot 0 drives into the wall at x = 0
    state = set_poses(env, params, [[0.2, 1.2], [1.6, 1.2]], [math.pi, math.pi / 2])
    state, _, done, info = env._step_state(jax.random.PRNGKey(0), state, jnp.zeros((2, 1)))
    assert int(state.collisions[0]) == 1
    assert bool(done) == flag and bool(info['crashed']) == flag and not bool(info['success'])


def test_testing_outcomes_exclusive():
    """Test rules (for_testing): every episode ends with exactly one of success, collision, timeout"""
    assert not EnvConfig().collision_ends_episode and EnvConfig().for_testing().collision_ends_episode
    cfg = EnvConfig(**SMALL, num_agents=3, max_episode_steps=12).for_testing()
    env = MultiMowerEnv(cfg)
    bank = build_bank([empty_map(), wall_map()], cfg)
    params = EnvParams(bank=bank, map_logits=jnp.zeros(2), goal_coverage=jnp.float32(0.2))
    E = 16
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(0), E), params)

    @jax.jit
    def run(state, key):
        def body(state, k):
            k_act, k_env = jax.random.split(k)
            actions = jax.random.uniform(k_act, (E, 3, cfg.action_dim), minval=-1, maxval=1)
            _, state, _, done, info = env.step_batch(jax.random.split(k_env, E), state, actions, params)
            return state, (done, info['success'], info['crashed'], info['collisions'], info['length'])
        return jax.lax.scan(body, state, jax.random.split(key, 300))[1]

    done, success, crashed, collisions, length = map(np.array, run(state, jax.random.PRNGKey(1)))
    done = done.astype(bool)
    success, crashed = success[done].astype(bool), crashed[done].astype(bool)
    timeout = ~success & ~crashed
    assert not np.any(success & crashed)
    assert np.allclose(success.mean() + crashed.mean() + timeout.mean(), 1)
    # a collision ended the episode: exactly one step with collisions; no collision otherwise
    assert np.all(collisions[done][crashed] > 0) and np.all(collisions[done][~crashed] == 0)
    assert np.all(length[done][timeout] == cfg.max_episode_steps)
    assert success.any() and crashed.any() and timeout.any()


@pytest.mark.parametrize('flag', [False, True])
def test_no_communication(flag):
    """Without communication a robot is rewarded on its own coverage only and gets no poses of the others"""
    env, params = make_env([empty_map()], num_agents=2, no_communication=flag)
    state = set_poses(env, params, [[0.8, 1.2], [1.6, 1.2]], [0.0, math.pi / 2])
    # the whole map was already covered by the team, but not in the robots' own beliefs
    state = state.replace(overlap=jnp.ones_like(state.overlap))
    if flag:
        assert not np.any(np.array(state.connected) & ~np.eye(2, dtype=bool))
        obs = env._observation(jax.random.PRNGKey(0), state)
        assert not np.any(np.array(obs['others_mask']))
    cov = env._coverage_update(state, state.pos + jnp.array([[0.1, 0.0], [0.0, 0.1]]))
    assert np.all(np.array(cov['new_share']) == 0)
    assert np.all(np.array(cov['reward_share']) > 0) == flag


# fields of EnvState with a leading robot axis
ROBOT_FIELDS = ('pos', 'heading', 'prev_pos', 'belief_overlap', 'belief_known', 'collisions', 'robot_collisions',
                'stuck_steps', 'covered_by', 'ep_return', 'lidar_dist', 'workload', 'active')


def first_robots(state, k):
    """The state of the first k robots (as for an environment with k robots)"""
    fields = {f: getattr(state, f)[:k] for f in ROBOT_FIELDS}
    return state.replace(connected=state.connected[:k, :k], **fields)


def test_variable_agents_reset():
    """min_agents: uniform number of robots, the first ones active, the parked ones at robot 0"""
    env, params = make_env([empty_map()], num_agents=4, min_agents=2)
    counts = []
    for i in range(200):
        _, state = env.reset(jax.random.PRNGKey(i), params)
        active = np.array(state.active)
        n = int(active.sum())
        counts.append(n)
        assert np.array_equal(active, np.arange(4) < n)
        pos = np.array(state.pos)
        assert np.allclose(pos[n:], pos[0])
        d = np.linalg.norm(pos[:n, None] - pos[None, :n], axis=-1) + np.eye(n) * 9
        assert d.min() >= env.cfg.robot_separation - 1e-6
        assert np.array_equal(np.array(state.connected), (active[:, None] & active[None, :]) | np.eye(4, dtype=bool))
    assert set(counts) == {2, 3, 4}
    assert all(c > 40 for c in np.bincount(counts)[2:])
    assert EnvConfig(num_agents=8, min_agents=2).for_testing().min_agents == 0


@pytest.mark.parametrize('comm_range', [0.0, 1.0])
def test_variable_agents_equal_fixed(comm_range):
    """An episode with k of 4 robots active is the episode of an environment with k robots:
    the parked robots do not move, cover, collide, appear in any observation or get rewards"""
    rng = np.random.default_rng(3)
    cfg = EnvConfig(**SMALL)
    maps = [random_map(rng, cfg, min_size_p=64, max_size_p=64)[:2]]
    coop = dict(workload_size=16, workload_reward_scale=0.02, critic_workload=True, max_episode_steps=1000)
    var, params = make_env(maps, num_agents=4, min_agents=2, comm_range=comm_range, **coop)
    seen = set()
    for seed in range(12):
        obs_v, state_v = var.reset(jax.random.PRNGKey(seed), params)
        k = int(state_v.active.sum())
        if k == 4 or k in seen:
            continue
        seen.add(k)
        fixed, _ = make_env(maps, num_agents=k, comm_range=comm_range, **coop)
        state_f = first_robots(state_v, k)
        obs_f = fixed._observation(jax.random.PRNGKey(0), state_f)
        obs_v = var._observation(jax.random.PRNGKey(0), state_v)
        parked0 = np.array(state_v.pos[k:])
        key = jax.random.PRNGKey(seed + 100)
        for t in range(25):
            for name in obs_f:
                a, b = np.array(obs_f[name]), np.array(obs_v[name])
                if name in ('global', 'progress'):
                    assert np.allclose(a, b, atol=1e-5), (name, t)
                elif name in ('others', 'others_mask', 'team'):
                    assert np.allclose(a, b[:k, :k], atol=1e-5), (name, t)
                    if name == 'others_mask':
                        assert not b[:, k:].any() and not b[k:, :].any()
                else:
                    assert np.allclose(a, b[:k], atol=1e-5), (name, t)
            key, k_act, k_step = jax.random.split(key, 3)
            actions = jax.random.uniform(k_act, (4, 1), minval=-1, maxval=1)
            state_f, r_f, d_f, info_f = fixed._step_state(k_step, state_f, actions[:k])
            state_v, r_v, d_v, info_v = var._step_state(k_step, state_v, actions)
            state_f, r_f = fixed._add_workload_shaping(fixed._update_workload(state_f), r_f, d_f)
            state_v, r_v = var._add_workload_shaping(var._update_workload(state_v), r_v, d_v)
            assert bool(d_f) == bool(d_v)
            assert np.allclose(np.array(r_f), np.array(r_v[:k]), atol=1e-5) and np.all(np.array(r_v[k:]) == 0)
            assert np.array_equal(np.array(state_f.overlap), np.array(state_v.overlap))
            assert np.allclose(np.array(state_f.workload), np.array(state_v.workload[:k]), atol=1e-4)
            assert np.all(np.array(state_v.workload[k:]) == 0)
            assert np.array_equal(np.array(state_f.collisions), np.array(state_v.collisions[:k]))
            assert np.allclose(np.array(state_f.ep_terms), np.array(state_v.ep_terms), atol=1e-4)
            for name in ('coverage', 'ep_return', 'collisions', 'success', 'crashed'):
                assert np.allclose(np.array(info_f[name]), np.array(info_v[name]), atol=1e-4), name
            assert int(info_v['robots']) == k
            assert np.allclose(np.array(state_v.pos[k:]), parked0)
            if bool(d_v):
                break
            obs_f = fixed._observation(jax.random.PRNGKey(t), state_f)
            obs_v = var._observation(jax.random.PRNGKey(t), state_v)
    assert seen == {2, 3}


@pytest.mark.slow
@pytest.mark.parametrize('algo', ['mappo', 'ippo'])
def test_mappo_update_variable_agents(algo):
    from marl.mappo import MAPPO
    env_cfg = EnvConfig(**SMALL, num_agents=4, min_agents=2, max_episode_steps=8, workload_size=16,
                        workload_reward_scale=0.02, shaping_gamma=TrainConfig().gamma, collision_ends_episode=True)
    train_cfg = TrainConfig(algo=algo, num_envs=4, num_steps=8, total_steps=128, minibatch_size=32, update_epochs=2,
                            hidden_dims=32)
    bank = build_bank([empty_map(), empty_map(48)], env_cfg)
    params = EnvParams(bank=bank, map_logits=jnp.zeros(2), goal_coverage=jnp.float32(0.9))
    algo = MAPPO(env_cfg, train_cfg)
    runner = algo.init(jax.random.PRNGKey(0), params)
    runner, metrics, infos = algo.update(runner, params)
    for k, v in metrics.items():
        assert np.isfinite(float(v)), k
    robots = np.array(infos['robots'])[np.array(infos['done'])]
    assert robots.size > 0 and robots.min() >= 2 and robots.max() <= 4
