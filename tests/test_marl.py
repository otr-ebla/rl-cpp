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
                          belief_overlap=jnp.zeros_like(state.belief_overlap), connected=env._connectivity(pos))
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
@pytest.mark.parametrize('algo,critic_state,comm_range', [
    ('mappo', 'global', 0), ('mappo', 'agent_specific', 0), ('ippo', 'global', 0), ('mappo', 'global', 1.0)])
def test_mappo_update(algo, critic_state, comm_range):
    from marl.mappo import MAPPO
    env_cfg = EnvConfig(**SMALL, num_agents=2, max_episode_steps=8, comm_range=comm_range)
    train_cfg = TrainConfig(algo=algo, critic_state=critic_state, num_envs=2, num_steps=8, total_steps=64, minibatch_size=16, update_epochs=2,
                            hidden_dims=32)
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


@pytest.mark.parametrize('algo,critic_state', [('mappo', 'global'), ('mappo', 'agent_specific'), ('ippo', 'global')])
def test_networks_agnostic_to_number_of_robots(algo, critic_state):
    """The same parameters work for any number of robots, invariant to the order of the robots"""
    from marl.mappo import actor_apply, critic_value, init_params, make_networks
    train_cfg = TrainConfig(algo=algo, critic_state=critic_state, hidden_dims=32, neighbor_dims=16)
    rng = np.random.default_rng(0)
    params = None
    for N in (1, 2, 5):
        maps = [random_map(rng, EnvConfig(**SMALL), min_size_p=64, max_size_p=64)[:2]]
        env, env_params = make_env(maps, num_agents=N, robot_separation=0.3)
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
