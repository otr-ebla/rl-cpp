"""
Multi-agent lawn mowing environment in JAX.

N robots mow the same map. They share one global map (perfect communication):
every step all robots move simultaneously, then the shared coverage and known
obstacle maps are updated with the contributions of all robots, and then each
robot extracts its own egocentric multi-scale observation from the shared maps.

The environment is a set of pure functions on an EnvState, for one
environment. Use jax.vmap to simulate many environments in parallel, see
marl.mappo. Episodes are reset automatically in step().

Coordinates: positions are in meters, (x, y) = (first, second) array index of
the maps, pixel (i, j) covers [i, i+1) x [j, j+1) in pixel units, and the
heading is the angle from the x axis towards the y axis, as in MowerEnv.

Differences to rlm.mower_env.MowerEnv (single agent):
  * no stacked observations, action observations, acceleration limits or
    action delay
  * the covered region is computed analytically (capsule swept by the cutting
    disc) instead of being rasterized with OpenCV
  * the multi-scale maps are sampled bilinearly from an average pooled
    pyramid of the canvas, with pixel centers handled exactly
  * frontier and global total variation rewards are not implemented (their
    default scale is 0 for mowing)
"""
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from marl.config import EnvConfig
from marl.maps import MapBank


@struct.dataclass
class EnvParams:
    """
    Parameters that may change during training without recompilation.

    bank:          the maps
    map_logits:    (K,) unnormalized log probabilities to sample each map at reset
    goal_coverage: coverage at which the episode is successfully completed
    """
    bank: MapBank
    map_logits: jnp.ndarray
    goal_coverage: jnp.ndarray


@struct.dataclass
class EnvState:
    pos: jnp.ndarray            # (N, 2) robot positions [m]
    heading: jnp.ndarray        # (N,) robot headings [rad]
    prev_pos: jnp.ndarray       # (N, 2) positions one step earlier [m]
    overlap: jnp.ndarray        # (S, S) int16, number of times each pixel was covered
    known: jnp.ndarray          # (S, S) bool, known obstacles (shared by all robots)
    obstacles: jnp.ndarray      # (S, S) bool, see MapBank
    dilated: jnp.ndarray        # (S, S) bool, see MapBank
    blocked: jnp.ndarray        # (S, S) bool, see MapBank
    size: jnp.ndarray           # () int32, map size [pixels]
    free_space: jnp.ndarray     # () float32, free pixels
    map_idx: jnp.ndarray        # () int32, index in the map bank
    goal_coverage: jnp.ndarray  # () float32
    t: jnp.ndarray              # () int32, elapsed steps
    non_new_steps: jnp.ndarray  # () int32, steps since any robot covered new ground
    coverage_px: jnp.ndarray    # () float32, covered free pixels (team)
    overlap_px: jnp.ndarray     # () float32, overlapping free pixels (team)
    collisions: jnp.ndarray     # (N,) int32
    ep_return: jnp.ndarray      # (N,) float32, undiscounted return so far
    lidar_dist: jnp.ndarray     # (N, R) float32, latest lidar distances [pixels]


def total_variation(img, img2):
    """
    Symmetric isotropic total variation, see rlm.utils.total_variation.
    img, img2: (H, W) float arrays
    """
    diff1 = jnp.abs(img[1:, :] - img[:-1, :])
    diff2 = jnp.abs(img[:, 1:] - img[:, :-1])
    img2 = jnp.maximum(img, img2)
    diff1 = jnp.minimum(diff1, jnp.abs(img2[1:, :] - img2[:-1, :]))
    diff2 = jnp.minimum(diff2, jnp.abs(img2[:, 1:] - img2[:, :-1]))
    tv = jnp.sum(jnp.sqrt(diff1[:, 1:] ** 2 + diff2[1:, :] ** 2)) + \
         jnp.sum(jnp.sqrt(diff1[:, 1:] ** 2 + diff2[:-1, :] ** 2)) + \
         jnp.sum(jnp.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2)) + \
         jnp.sum(jnp.sqrt(diff1[:, :-1] ** 2 + diff2[1:, :] ** 2))
    return tv / 4


def segment_distance(pi, pj, a, b):
    """
    Distance from points (pi, pj) to the line segment a -> b.
    pi, pj: broadcastable arrays of point coordinates
    a, b: (2,) segment end points
    """
    d = b - a
    len2 = jnp.maximum(d @ d, 1e-12)
    t = jnp.clip(((pi - a[0]) * d[0] + (pj - a[1]) * d[1]) / len2, 0, 1)
    return jnp.sqrt((pi - a[0] - t * d[0]) ** 2 + (pj - a[1] - t * d[1]) ** 2)


def max_pool(x, k):
    """Separable k x k max filter (same size output, zero padding), x: (S, S) uint8"""
    pad = k // 2
    x = jax.lax.reduce_window(x, jnp.uint8(0), jax.lax.max, (k, 1), (1, 1), [(pad, pad), (0, 0)])
    return jax.lax.reduce_window(x, jnp.uint8(0), jax.lax.max, (1, k), (1, 1), [(0, 0), (pad, pad)])


def avg_pool(x, f):
    """Average pooling of (S, S) by an integer factor f"""
    S = x.shape[0]
    return x.reshape(S // f, f, S // f, f).mean(axis=(1, 3))


def bilinear(img, u, v, pad_value):
    """
    Bilinear sampling of img (H, W) at continuous pixel center coordinates
    (u, v), i.e. pixel (k, l) is at (u, v) = (k, l). Values outside img are
    pad_value.
    """
    H, W = img.shape
    u0 = jnp.floor(u)
    v0 = jnp.floor(v)
    du = u - u0
    dv = v - v0
    u0 = u0.astype(jnp.int32)
    v0 = v0.astype(jnp.int32)

    def g(a, b):
        valid = (a >= 0) & (a < H) & (b >= 0) & (b < W)
        return jnp.where(valid, img[jnp.clip(a, 0, H - 1), jnp.clip(b, 0, W - 1)], pad_value)

    return (1 - du) * (1 - dv) * g(u0, v0) + du * (1 - dv) * g(u0 + 1, v0) + \
           (1 - du) * dv * g(u0, v0 + 1) + du * dv * g(u0 + 1, v0 + 1)


def _quantize(x):
    return jnp.round(jnp.clip(x, 0, 1) * 255).astype(jnp.uint8)


class MultiMowerEnv:
    """
    Multi-agent mowing environment, see the module docstring.

    Observations (dict) of one environment:
      maps:   (N, H, W, C) uint8, egocentric multi-scale maps of each robot,
              channels are scale major: [scale 0: coverage, obstacles,
              frontier, robots], [scale 1: ...], ... (0-255 = 0-1)
      lidar:  (N, lidar_rays) float32, normalized lidar distances
      global: (G, G, 4) uint8, global maps for the critic: coverage, true
              obstacles, known obstacles, robots (canvas frame)
      agent:  (N, 6) float32, per robot state for the critic: position / canvas
              size, cos/sin heading, team coverage, elapsed time
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ API

    @partial(jax.jit, static_argnums=0)
    def reset(self, key, params: EnvParams):
        k_reset, k_obs = jax.random.split(key)
        state = self._reset_state(k_reset, params)
        return self._observation(k_obs, state), state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state: EnvState, actions, params: EnvParams):
        """
        actions: (N, action_dim) in [-1, 1]
        Returns obs, state, rewards (N,), done (), info.
        info holds the statistics of the episode that ended in this step
        (only meaningful where done is True).
        """
        k_step, k_reset, k_obs = jax.random.split(key, 3)
        state, rewards, done, info = self._step_state(k_step, state, actions)
        reset_state = self._reset_state(k_reset, params)
        state = jax.tree_util.tree_map(lambda r, s: jnp.where(done, r, s), reset_state, state)
        return self._observation(k_obs, state), state, rewards, done, info

    # --------------------------------------------------------- simulation

    def _spawn(self, key, blocked, size):
        """
        Random collision free poses, robots at least robot_separation apart.
        Rejection sampling with a fixed number of candidates per robot.
        """
        cfg = self.cfg
        N = cfg.num_agents
        num_candidates = 128
        size_m = size * cfg.meters_per_pixel
        r = cfg.mower_radius
        k_pos, k_head = jax.random.split(key)

        def place(placed, xs):
            i, k = xs
            cand = r + jax.random.uniform(k, (num_candidates, 2)) * (size_m - 2 * r)
            pix = jnp.floor(cand * cfg.pixels_per_meter).astype(jnp.int32)
            ok = ~blocked[pix[:, 0], pix[:, 1]]
            ok &= jnp.all((cand > r) & (cand < size_m - r), axis=1)
            dist = jnp.linalg.norm(cand[:, None, :] - placed[None, :, :], axis=-1)
            ok &= jnp.all((dist >= cfg.robot_separation) | (jnp.arange(N) >= i)[None, :], axis=1)
            pos = cand[jnp.argmax(ok)]
            return placed.at[i].set(pos), None

        placed, _ = jax.lax.scan(
            place, jnp.zeros((N, 2)), (jnp.arange(N), jax.random.split(k_pos, N)))
        heading = jax.random.uniform(k_head, (N,), maxval=2 * jnp.pi)
        return placed, heading

    def _reset_state(self, key, params: EnvParams):
        cfg = self.cfg
        N = cfg.num_agents
        S = cfg.canvas_size
        k_map, k_pose = jax.random.split(key)
        bank = params.bank
        idx = jax.random.categorical(k_map, params.map_logits)
        blocked = bank.blocked[idx]
        size = bank.size[idx]
        pos, heading = self._spawn(k_pose, blocked, size)
        state = EnvState(
            pos=pos,
            heading=heading,
            prev_pos=pos,
            overlap=jnp.zeros((S, S), dtype=jnp.int16),
            known=bank.known[idx],
            obstacles=bank.obstacles[idx],
            dilated=bank.dilated[idx],
            blocked=blocked,
            size=size,
            free_space=bank.free_space[idx],
            map_idx=idx,
            goal_coverage=jnp.asarray(params.goal_coverage, dtype=jnp.float32),
            t=jnp.zeros((), dtype=jnp.int32),
            non_new_steps=jnp.zeros((), dtype=jnp.int32),
            coverage_px=jnp.zeros(()),
            overlap_px=jnp.zeros(()),
            collisions=jnp.zeros((N,), dtype=jnp.int32),
            ep_return=jnp.zeros((N,)),
            lidar_dist=jnp.zeros((N, cfg.lidar_rays)))

        # Initial lidar detections and covered discs
        dist, hit_pix, write = self._lidar(pos, heading, state.obstacles, size)
        state = state.replace(known=self._update_known(state.known, hit_pix, write), lidar_dist=dist)
        cov = self._coverage_update(state, pos, initial=True)
        return state.replace(overlap=cov['overlap'], coverage_px=cov['team_new'])

    def _velocities(self, actions):
        cfg = self.cfg
        actions = jnp.clip(actions, -1, 1)
        if cfg.constant_lin_vel:
            throttle = jnp.ones(actions.shape[0])
            steering = actions[:, 0]
        else:
            throttle = actions[:, 0]
            steering = actions[:, 1]
        lin_vel = throttle
        if cfg.steering_limits_lin_vel:
            lin_vel = lin_vel * (1 - jnp.abs(steering))
        lin_vel = lin_vel * cfg.max_lin_vel
        lin_vel = jnp.where(lin_vel < 0, lin_vel / 2, lin_vel)
        ang_vel = steering * cfg.max_ang_vel
        return lin_vel, ang_vel

    def _lidar(self, pos, heading, obstacles, size):
        """
        Ray marching through the obstacle map, plus analytic ray/disc
        intersection with the other robots.

        Returns
          dist:    (N, R) distance to the first detection [pixels]
          hit_pix: (N, R, 2) pixel of the detected map obstacle
          write:   (N, R) whether the detection is a map obstacle (that should
                   be added to the known obstacles), i.e. not out of bounds,
                   not a robot and not max range
        """
        cfg = self.cfg
        N = cfg.num_agents
        samples = cfg.lidar_samples
        angles = jnp.linspace(-cfg.lidar_fov / 2, cfg.lidar_fov / 2, cfg.lidar_rays) * jnp.pi / 180
        ang = heading[:, None] + angles[None, :]                            # N x R
        dirs = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)             # N x R x 2
        pos_p = pos * cfg.pixels_per_meter
        s = jnp.arange(1, samples + 1, dtype=jnp.float32)
        pts = pos_p[:, None, None, :] + s[None, None, :, None] * dirs[:, :, None, :]  # N x R x samples x 2
        idx = jnp.floor(pts).astype(jnp.int32)
        inside = jnp.all((idx >= 0) & (idx < size), axis=-1)
        S = obstacles.shape[0]
        idx_c = jnp.clip(idx, 0, S - 1)
        hit = obstacles[idx_c[..., 0], idx_c[..., 1]] | ~inside
        first = jnp.argmax(hit, axis=-1)                                    # N x R
        any_hit = jnp.any(hit, axis=-1)
        map_dist = jnp.where(any_hit, first + 1, samples).astype(jnp.float32)
        hit_pix = jnp.take_along_axis(idx_c, first[..., None, None], axis=2)[:, :, 0, :]
        hit_inside = jnp.take_along_axis(inside, first[..., None], axis=2)[..., 0]

        # Other robots
        R = cfg.radius_p
        rel = pos_p[None, :, :] - pos_p[:, None, :]                         # N x N x 2, from i to j
        t = jnp.einsum('ird,ijd->irj', dirs, rel)                           # N x R x N
        perp2 = jnp.sum(rel ** 2, axis=-1)[:, None, :] - t ** 2
        other = ~jnp.eye(N, dtype=bool)[:, None, :]
        hit_r = other & (perp2 <= R ** 2) & (t > 0)
        t_hit = jnp.maximum(t - jnp.sqrt(jnp.maximum(R ** 2 - perp2, 0)), 0)
        robot_dist = jnp.min(jnp.where(hit_r, t_hit, jnp.inf), axis=-1)    # N x R

        dist = jnp.minimum(map_dist, robot_dist)
        write = any_hit & hit_inside & (map_dist <= robot_dist)
        return dist, hit_pix, write

    def _update_known(self, known, hit_pix, write):
        S = known.shape[0]
        i = jnp.where(write, hit_pix[..., 0], S)
        j = jnp.where(write, hit_pix[..., 1], S)
        return known.at[i.reshape(-1), j.reshape(-1)].set(True, mode='drop')

    def _coverage_update(self, state: EnvState, new_pos, initial=False):
        """
        Region covered by each robot in this step, in a local window around
        each robot, see MowerEnv._compute_coverage_diff. The covered region
        is the capsule swept from the old to the new position, minus the
        capsule swept in the previous step.

        Newly covered pixels that are covered by k robots in the same step
        are attributed 1/k to each of them.
        """
        cfg = self.cfg
        S = cfg.canvas_size
        W = cfg.window
        R = cfg.radius_p
        ppm = cfg.pixels_per_meter
        old_p = state.pos * ppm
        new_p = new_pos * ppm
        prev_p = state.prev_pos * ppm

        start = jnp.clip(jnp.floor(new_p).astype(jnp.int32) - W // 2, 0, S - W)  # N x 2
        ii = start[:, 0, None] + jnp.arange(W)[None, :]                     # N x W
        jj = start[:, 1, None] + jnp.arange(W)[None, :]
        II = jnp.broadcast_to(ii[:, :, None], (cfg.num_agents, W, W))
        JJ = jnp.broadcast_to(jj[:, None, :], (cfg.num_agents, W, W))
        ci = ii[:, :, None] + 0.5
        cj = jj[:, None, :] + 0.5

        seg = jax.vmap(segment_distance, in_axes=(0, 0, 0, 0))
        cap_now = seg(ci, cj, old_p, new_p) <= R
        if initial:
            cap_prev = jnp.zeros_like(cap_now)
        else:
            cap_prev = seg(ci, cj, prev_p, old_p) <= R
        outside = (II >= state.size) | (JJ >= state.size)
        blocked = state.obstacles[II, JJ] | outside
        diff = cap_now & ~cap_prev & ~blocked                               # N x W x W

        overlap_old = state.overlap[II, JJ]
        covered_old = overlap_old > 0
        newly = diff & ~covered_old
        overlap = state.overlap.at[II, JJ].add(diff.astype(jnp.int16))
        # for newly covered pixels, overlap_new = number of robots covering it now
        count = jnp.maximum(overlap[II, JJ], 1).astype(jnp.float32)
        dilated = state.dilated[II, JJ]
        newly_d = newly & ~dilated
        diff_d = diff & ~dilated
        new_share = jnp.sum(jnp.where(newly_d, 1 / count, 0), axis=(1, 2))  # N
        team_new = jnp.sum(new_share)
        team_diff = jnp.sum(diff_d)

        # Incremental total variation of each robot's own contribution
        cov_old_f = covered_old.astype(jnp.float32)
        cov_new_f = (covered_old | diff).astype(jnp.float32)
        dil_f = dilated.astype(jnp.float32)
        tv = jax.vmap(total_variation)
        tv_diff = tv(cov_new_f, dil_f) - tv(cov_old_f, dil_f)

        # Overlap with earlier coverage
        overlap_sum = jnp.sum(diff * overlap_old, axis=(1, 2)).astype(jnp.float32)

        return dict(overlap=overlap, new_share=new_share, team_new=team_new,
                    team_diff=team_diff, any_new=jnp.any(newly), tv_diff=tv_diff,
                    overlap_sum=overlap_sum)

    def _step_state(self, key, state: EnvState, actions):
        cfg = self.cfg
        N = cfg.num_agents
        r = cfg.mower_radius
        dt = cfg.step_size

        # New poses
        lin_vel, ang_vel = self._velocities(actions)
        dh = ang_vel * dt
        new_heading = jnp.mod(state.heading + dh, 2 * jnp.pi)
        mid = state.heading + dh / 2
        new_pos = state.pos + (lin_vel * dt)[:, None] * jnp.stack([jnp.cos(mid), jnp.sin(mid)], axis=-1)

        # Collisions with walls, obstacles and other robots
        size_m = state.size * cfg.meters_per_pixel
        wall = jnp.any((new_pos <= r) | (new_pos >= size_m - r), axis=1)
        S = cfg.canvas_size
        pix = jnp.clip(jnp.floor(new_pos * cfg.pixels_per_meter).astype(jnp.int32), 0, S - 1)
        obstacle = state.blocked[pix[:, 0], pix[:, 1]] & ~wall
        other = ~jnp.eye(N, dtype=bool)
        d_new = jnp.linalg.norm(new_pos[:, None] - new_pos[None, :], axis=-1)
        d_old = jnp.linalg.norm(new_pos[:, None] - state.pos[None, :], axis=-1)
        robot = jnp.any(other & ((d_new < 2 * r) | (d_old < 2 * r)), axis=1) & ~wall & ~obstacle
        collided = wall | obstacle | robot
        new_pos = jnp.where(collided[:, None], state.pos, new_pos)

        # Lidar and known obstacles
        lidar_dist, hit_pix, write = self._lidar(new_pos, new_heading, state.obstacles, state.size)
        known = self._update_known(state.known, hit_pix, write)

        # Coverage
        cov = self._coverage_update(state, new_pos)
        coverage_px = state.coverage_px + cov['team_new']
        overlap_px = state.overlap_px + cov['team_diff'] - cov['team_new']
        coverage = coverage_px / state.free_space

        # Area reward: individual and team part
        max_new = cfg.max_newly_visited
        indiv = cov['new_share'] / max_new
        team = cov['team_new'] / (N * max_new)
        w = cfg.team_reward_weight
        reward_area = cfg.newly_visited_reward_scale * ((1 - w) * jnp.where(collided, 0, indiv) + w * team)
        reward_area = jnp.minimum(reward_area, cfg.newly_visited_reward_max)

        # Overlap reward
        reward_ovrlp = jnp.minimum(cov['overlap_sum'] * cfg.overlap_reward_scale / max_new, cfg.overlap_reward_max)
        if not cfg.overlap_reward_always:
            reward_ovrlp = jnp.where(reward_area == 0, reward_ovrlp, 0)
        reward_ovrlp = jnp.where(collided, 0, -reward_ovrlp)

        # Local total variation reward
        reward_tv = -cov['tv_diff'] * cfg.meters_per_pixel / dt / cfg.max_lin_vel / 2.5
        reward_tv = reward_tv * cfg.local_tv_reward_scale
        reward_tv = jnp.clip(reward_tv, -cfg.local_tv_reward_max, cfg.local_tv_reward_max)
        reward_tv = jnp.where(collided, 0, reward_tv)

        # Turning, collision and constant rewards
        reward_turn = jnp.where(collided, 0, -cfg.turn_reward_scale * jnp.abs(ang_vel / cfg.max_ang_vel))
        reward_coll = jnp.where(wall, cfg.wall_collision_reward, 0) + \
                      jnp.where(obstacle, cfg.obstacle_collision_reward, 0) + \
                      jnp.where(robot, cfg.robot_collision_reward, 0)
        if cfg.constant_reward_always:
            reward_const = jnp.full((N,), cfg.constant_reward)
        else:
            reward_const = jnp.where(cov['new_share'] > 0, 0, cfg.constant_reward)

        # Termination and truncation
        t = state.t + 1
        non_new_steps = jnp.where(cov['any_new'], 0, state.non_new_steps + 1)
        success = coverage >= state.goal_coverage
        truncated = ~success & ((t >= cfg.max_episode_steps) | (non_new_steps >= cfg.max_non_new_steps))
        done = success | truncated
        reward_goal = jnp.where(success, cfg.goal_coverage_reward, 0)
        reward_trunc = jnp.where(truncated, -cfg.truncation_reward_scale * (1 - coverage), 0)

        rewards = reward_area + reward_ovrlp + reward_tv + reward_turn + reward_coll + \
                  reward_const + reward_goal + reward_trunc

        state = state.replace(
            pos=new_pos,
            heading=new_heading,
            prev_pos=state.pos,
            overlap=cov['overlap'],
            known=known,
            t=t,
            non_new_steps=non_new_steps,
            coverage_px=coverage_px,
            overlap_px=overlap_px,
            collisions=state.collisions + collided,
            ep_return=state.ep_return + rewards,
            lidar_dist=lidar_dist)
        info = dict(
            success=success,
            coverage=coverage,
            overlap=overlap_px / jnp.maximum(coverage_px, 1),
            length=t,
            ep_return=jnp.mean(state.ep_return),
            collisions=jnp.sum(state.collisions),
            map_idx=state.map_idx)
        return state, rewards, done, info

    # ------------------------------------------------------- observations

    def _observation(self, key, state: EnvState):
        cfg = self.cfg
        N = cfg.num_agents
        S = cfg.canvas_size
        ppm = cfg.pixels_per_meter
        k_pos, k_head, k_lidar = jax.random.split(key, 3)

        # Lidar
        dist = state.lidar_dist / ppm
        dist = dist + cfg.lidar_noise * jax.random.normal(k_lidar, dist.shape)
        lidar = jnp.clip(dist / cfg.lidar_range, 0, 1)

        # Shared full resolution maps
        ii = jnp.arange(S)[:, None]
        jj = jnp.arange(S)[None, :]
        outside = (ii >= state.size) | (jj >= state.size)
        border = (ii == 0) | (jj == 0) | (ii == state.size - 1) | (jj == state.size - 1)
        covered = state.overlap > 0
        if cfg.overlap_observation:
            coverage = jnp.tanh(0.2 * state.overlap.astype(jnp.float32))
        else:
            coverage = covered.astype(jnp.float32)
        known = state.known.astype(jnp.float32)
        channels = [coverage, known]
        pads = [0.0, 0.0 if cfg.all_unknown else 1.0]
        if cfg.frontier_observation:
            # see MowerEnv._compute_frontier_map
            known_b = (state.known | border | outside).astype(jnp.uint8)
            known_dil = max_pool(known_b, cfg.obstacle_dilation) > 0 if cfg.obstacle_dilation > 1 else known_b > 0
            cov_c = covered & ~known_dil
            free = ~(cov_c | known_dil)
            frontier = (max_pool(cov_c.astype(jnp.uint8), 3) > 0) & free
            channels.append(frontier.astype(jnp.float32))
            pads.append(0.0)

        # Pyramids: level n is average pooled by scale_factor ** n
        pyramids = []
        for ch in channels:
            levels = [ch]
            for _ in range(1, cfg.num_maps):
                levels.append(avg_pool(levels[-1], cfg.scale_factor))
            pyramids.append(levels)

        # Egocentric sampling grids (noisy pose), up is forward
        noisy_pos = state.pos + cfg.position_noise * jax.random.normal(k_pos, state.pos.shape)
        noisy_heading = state.heading + cfg.heading_noise * jax.random.normal(k_head, state.heading.shape)
        c0 = (cfg.input_size - 1) / 2
        rows = jnp.arange(cfg.input_size, dtype=jnp.float32)
        fwd = (c0 - rows)[:, None]                                           # H x 1
        right = (rows - c0)[None, :]                                         # 1 x W
        f_vec = jnp.stack([jnp.cos(noisy_heading), jnp.sin(noisy_heading)], axis=-1)   # N x 2
        r_vec = jnp.stack([jnp.sin(noisy_heading), -jnp.cos(noisy_heading)], axis=-1)  # N x 2
        npos_p = noisy_pos * ppm
        pos_p = state.pos * ppm
        other = ~jnp.eye(N, dtype=bool)

        maps = []
        for n in range(cfg.num_maps):
            f = cfg.scale_factor ** n
            # world coordinates of the output pixel centers [pixels]: N x H x W x 2
            world = npos_p[:, None, None, :] + f * (
                fwd[None, :, :, None] * f_vec[:, None, None, :] +
                right[None, :, :, None] * r_vec[:, None, None, :])
            u = world[..., 0] / f - 0.5
            v = world[..., 1] / f - 0.5
            for c, (levels, pad) in enumerate(zip(pyramids, pads)):
                val = bilinear(levels[n], u, v, pad)
                if cfg.frontier_observation and c == 2:
                    val = (val > 1e-6).astype(jnp.float32)
                maps.append(val)
            if cfg.robot_observation:
                # pixels that (approximately) intersect another robot
                d = jnp.linalg.norm(world[:, :, :, None, :] - pos_p[None, None, None, :, :], axis=-1)
                hit = (d <= cfg.radius_p + f * 0.7072) & other[:, None, None, :]
                maps.append(jnp.any(hit, axis=-1).astype(jnp.float32))
        maps = _quantize(jnp.stack(maps, axis=-1))                          # N x H x W x C

        # Global state for the critic (canvas frame)
        G = cfg.global_size
        g = S // G
        centers = (jnp.arange(G, dtype=jnp.float32) + 0.5) * g
        d = jnp.sqrt((centers[:, None, None] - pos_p[None, None, :, 0]) ** 2 +
                     (centers[None, :, None] - pos_p[None, None, :, 1]) ** 2)        # G x G x N
        robots_g = jnp.any(d <= cfg.radius_p + g * 0.7072, axis=-1)
        global_maps = jnp.stack([
            avg_pool(coverage, g),
            avg_pool((state.obstacles | outside).astype(jnp.float32), g),
            avg_pool(known, g),
            robots_g.astype(jnp.float32)], axis=-1)
        global_maps = _quantize(global_maps)

        coverage_frac = state.coverage_px / state.free_space
        agent = jnp.concatenate([
            pos_p / S,
            jnp.cos(state.heading)[:, None],
            jnp.sin(state.heading)[:, None],
            jnp.full((N, 1), coverage_frac),
            jnp.full((N, 1), state.t / cfg.max_episode_steps)], axis=-1)

        return {'maps': maps, 'lidar': lidar.astype(jnp.float32),
                'global': global_maps, 'agent': agent.astype(jnp.float32)}
