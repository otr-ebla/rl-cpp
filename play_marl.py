"""
Interactive pygame viewer of a trained MAPPO policy: the robots mow the
evaluation maps step by step, like a video game.

Examples:
  python play_marl.py --load experiments/marl_4robots/checkpoint
  python play_marl.py --load experiments/marl_4robots/checkpoint --num_agents 8 --map 5
  python play_marl.py --load experiments/marl_4robots/checkpoint --maps "maps/giant_mowing_*.png"
  python play_marl.py --load experiments/seeds/mappo_s0/checkpoint --maps "maps/test_*.png" "maps/eval_*.png"
  python play_marl.py --load experiments/seeds/mappo_s0/checkpoint --maps maps/corridor_120m_0.6m.png --num_agents 1 --start 60.15 60 0

Maps are grouped by the canvas size they need, and one simulation is compiled
per canvas size (the first time a map of that size is shown), so small maps
stay fast when large maps are also loaded.

Keys:
  space        pause / resume
  right / left next / previous map (new episode)
  .            one step (when paused)
  s / a        faster / slower (x1 = real time)
  n / p        next / previous map
  r            restart the episode on the same map
  tab          select the next robot (observation panel and crops)
  c            show the multi-scale crops of the selected robot on the map
               (thick edge = forward), also outlined in the observation panels
  m            switch the observation panels between the composite view and
               each scale shown separately, as received by the network
  l            toggle lidar rays
  t            toggle paths
  d            toggle deterministic / sampled actions
  b            limited communication: switch the map between the true global
               map and the belief of the selected robot
  q / Esc      quit

The map shows coverage (green, darker with more overlap), obstacles not yet
detected (gray), detected obstacles (black) and the frontier (magenta). The
panels on the right show the multi-scale egocentric observation of the
selected robot (up = forward), with the same colors as the single-agent
renderer (rlm/renderer.py).

With limited communication (--comm_range > 0), the communication range of the
selected robot is drawn as a circle and robots within range of each other are
linked by a line.
"""
import argparse
import glob
import math
import os
import time
from argparse import BooleanOptionalAction
from functools import partial

# Only allocate the GPU memory that is needed, so that the viewer can run next to a training
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from marl.config import EnvConfig
from marl.env import EnvParams, MultiMowerEnv, frontier_map
from marl.maps import build_bank, canvas_size_for, eval_map_files, load_map
from marl.mappo import actor_apply, load_checkpoint, make_networks

ROBOT_COLORS = [(228, 26, 28), (55, 126, 184), (255, 127, 0), (152, 78, 163),
                (166, 86, 40), (247, 129, 191), (77, 175, 74), (0, 160, 160)]
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (110, 110, 110)
PANEL_BG = (240, 240, 240)
LABEL_BG = (178, 178, 178)
CROP_COLORS = [(230, 25, 75), (245, 130, 48), (0, 130, 200), (145, 30, 180), (70, 240, 240), (240, 50, 230)]

# Map cell codes (see map_codes) -> colors
CODE_COLORS = np.array([
    (255, 255, 255),                                      # 0 free, not covered
    (170, 230, 170), (110, 200, 110), (60, 160, 60), (20, 110, 20),  # 1-4 covered 1, 2, 3, 4+ times
    (128, 128, 128),                                      # 5 obstacle not yet detected
    (0, 0, 0),                                            # 6 detected obstacle
    (255, 0, 255),                                        # 7 frontier
    (255, 255, 255)], dtype=np.uint8)                     # 8 outside the map

SPEEDS = [1, 2, 4, 8, 16, 32, math.inf]
MAX_FPS = 60
NEXT_MAP_DELAY = 3.0  # seconds the final state is shown before the next map


class Simulation:
    """Single environment with the policy, on the JAX device"""

    def __init__(self, env_cfg, actor, actor_params, env_params, start=None):
        self.env = MultiMowerEnv(env_cfg)
        self.cfg = env_cfg
        env = self.env
        N = env_cfg.num_agents
        if start is not None:
            # fixed start poses (x, y [m], heading [degrees] per robot) instead of the random spawn
            start = np.asarray(start, dtype=np.float32).reshape(N, 3)
            pos, heading = jnp.asarray(start[:, :2]), jnp.asarray(np.radians(start[:, 2]))
            env._spawn = lambda key, blocked, size: (pos, heading)

        @jax.jit
        def reset(key, map_logits):
            obs, state = env.reset(key, env_params.replace(map_logits=map_logits))
            return obs, state

        @jax.jit
        def step(key, obs, state, deterministic):
            k_act, k_step, k_obs = jax.random.split(key, 3)
            mean, log_std = actor_apply(actor, actor_params, obs)
            noise = jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            action = jnp.where(deterministic, mean, mean + noise)
            # no automatic reset, so that the final state can be shown
            state, rewards, done, info = env._step_state(k_step, state, action.reshape(N, -1))
            return env._observation(k_obs, state), state, rewards, done, info

        @partial(jax.jit, static_argnums=2)
        def map_codes(state, robot, belief):
            """
            (S, S) uint8 cell codes for display, see CODE_COLORS, of the true
            global map, or (belief) of the belief of robot (limited communication)
            """
            S = env_cfg.canvas_size
            ii = jnp.arange(S)[:, None]
            jj = jnp.arange(S)[None, :]
            outside = (ii >= state.size) | (jj >= state.size)
            border = (ii == 0) | (jj == 0) | (ii == state.size - 1) | (jj == state.size - 1)
            if belief:
                overlap, known = state.belief_overlap[robot], state.belief_known[robot]
            else:
                overlap, known = state.overlap, state.known
            frontier = frontier_map(overlap > 0, known, border | outside, env_cfg.obstacle_dilation)
            code = jnp.clip(overlap, 0, 4).astype(jnp.uint8)
            code = jnp.where(frontier, 7, code)
            code = jnp.where(state.obstacles & ~known, 5, code)
            code = jnp.where(known, 6, code)
            return jnp.where(outside, 8, code).astype(jnp.uint8)

        self._reset, self._step, self._map_codes = reset, step, map_codes

    def reset(self, key, map_idx, num_maps):
        logits = jnp.where(jnp.arange(num_maps) == map_idx, 0.0, -jnp.inf)
        self.obs, self.state = self._reset(key, logits)

    def step(self, key, deterministic):
        self.obs, self.state, rewards, done, info = self._step(key, self.obs, self.state, deterministic)
        return jax.device_get((rewards, done, info))

    def map_codes(self, robot=0, belief=False):
        return np.asarray(self._map_codes(self.state, robot, belief))


class Simulations:
    """One Simulation per canvas size, built the first time a map of that size is shown"""

    def __init__(self, env_cfg, actor, actor_params, maps, goal_coverage, start=None):
        self.env_cfg, self.actor, self.actor_params = env_cfg, actor, actor_params
        self.maps, self.goal_coverage, self.start = maps, goal_coverage, start
        self.canvas = [canvas_size_for([m[0].shape[0]], env_cfg) for m in maps]
        self._sims = {}

    def get(self, map_idx):
        """(Simulation, index of the map in its bank, number of maps in the bank)"""
        S = self.canvas[map_idx]
        members = [i for i, c in enumerate(self.canvas) if c == S]
        if S not in self._sims:
            cfg = EnvConfig.from_dict({**self.env_cfg.to_dict(), 'canvas_size': S})
            bank = build_bank([self.maps[i] for i in members], cfg)
            params = EnvParams(bank=bank, map_logits=jnp.zeros(len(members)),
                               goal_coverage=jnp.float32(self.goal_coverage))
            print(f'compiling the simulation for canvas {S} ({len(members)} maps)', flush=True)
            self._sims[S] = Simulation(cfg, self.actor, self.actor_params, params, self.start)
        return self._sims[S], members.index(map_idx), len(members)


class Viewer:

    def __init__(self, sims: Simulations, map_names, map_sizes, args):
        self.sims = sims
        self.cfg = sims.env_cfg
        self.map_names = map_names
        self.map_sizes = map_sizes
        self.args = args
        self.key = jax.random.PRNGKey(args.seed)
        self.map_idx = args.map % len(map_names)
        self.speed_idx = SPEEDS.index(args.speed) if args.speed in SPEEDS else 2
        self.deterministic = args.deterministic
        self.paused = False
        self.show_lidar = True
        self.show_paths = True
        self.selected = 0
        self.show_crops = False
        self.show_scales = False
        self.show_belief = False

        margin = 10
        self.map_rect = pygame.Rect(margin, margin, args.window, args.window)
        panel_w = 620
        self.panel_rect = pygame.Rect(self.map_rect.right + margin, margin, panel_w, args.window)
        pygame.init()
        self.screen = pygame.display.set_mode((self.panel_rect.right + margin, args.window + 2 * margin))
        pygame.display.set_caption('MARL mowing - [space] pause [s/a] speed [right/left] map [tab] robot '
                                   '[c] crops [m] scales [q] quit')
        self.font = pygame.font.SysFont('dejavusansmono,menlo,couriernew,monospace', 14)
        self.label_font = pygame.font.SysFont('dejavuserif,timesnewroman,times,serif', 14)
        self.big_font = pygame.font.SysFont('dejavusans,helvetica,arial', 20, bold=True)
        self.new_episode()

    # ------------------------------------------------------------ episode

    def new_episode(self):
        self.key, k = jax.random.split(self.key)
        self.sim, local_idx, num_local = self.sims.get(self.map_idx)
        self.cfg = self.sim.cfg
        self.sim.reset(k, local_idx, num_local)
        pos = np.asarray(self.sim.state.pos) * self.cfg.pixels_per_meter
        self.paths = [[tuple(p)] for p in pos]
        self.done = False
        self.done_time = None
        self.result = None
        self.last_rewards = np.zeros(self.cfg.num_agents)
        self.sim_steps_wall = time.perf_counter()

    def step(self):
        self.key, k = jax.random.split(self.key)
        rewards, done, info = self.sim.step(k, self.deterministic)
        self.last_rewards = rewards
        pos = np.asarray(self.sim.state.pos) * self.cfg.pixels_per_meter
        for path, p in zip(self.paths, pos):
            path.append(tuple(p))
        if done:
            self.done = True
            self.done_time = time.perf_counter()
            self.result = 'SUCCESS' if info['success'] else 'TRUNCATED'

    def change_map(self, delta):
        self.map_idx = (self.map_idx + delta) % len(self.map_names)
        self.new_episode()

    # ------------------------------------------------------------- loop

    def run(self):
        clock = pygame.time.Clock()
        next_step = time.perf_counter()
        while True:
            self.handle_events()
            now = time.perf_counter()
            if self.done:
                if not self.paused and now - self.done_time > NEXT_MAP_DELAY:
                    self.change_map(1)
                    next_step = now
            elif not self.paused:
                speed = SPEEDS[self.speed_idx]
                if math.isinf(speed):
                    # as many steps as fit in one frame
                    t_end = now + 1 / MAX_FPS
                    while not self.done and time.perf_counter() < t_end:
                        self.step()
                else:
                    interval = self.cfg.step_size / speed
                    next_step = max(next_step, now - 0.25)  # do not catch up after long pauses
                    while not self.done and next_step <= now:
                        self.step()
                        next_step += interval
            else:
                next_step = now
            self.draw()
            clock.tick(MAX_FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type != pygame.KEYDOWN:
                continue
            key = event.key
            if key in (pygame.K_q, pygame.K_ESCAPE):
                raise SystemExit
            elif key == pygame.K_SPACE:
                self.paused = not self.paused
            elif key == pygame.K_RIGHT:
                self.change_map(1)
            elif key == pygame.K_LEFT:
                self.change_map(-1)
            elif key == pygame.K_PERIOD and self.paused and not self.done:
                self.step()
            elif key == pygame.K_s:
                self.speed_idx = (self.speed_idx + 1) % len(SPEEDS)
            elif key == pygame.K_a:
                self.speed_idx = max(self.speed_idx - 1, 0)
            elif key == pygame.K_n:
                self.change_map(1)
            elif key == pygame.K_p:
                self.change_map(-1)
            elif key == pygame.K_r:
                self.new_episode()
            elif key == pygame.K_TAB:
                self.selected = (self.selected + 1) % self.cfg.num_agents
            elif key == pygame.K_c:
                self.show_crops = not self.show_crops
            elif key == pygame.K_m:
                self.show_scales = not self.show_scales
            elif key == pygame.K_l:
                self.show_lidar = not self.show_lidar
            elif key == pygame.K_t:
                self.show_paths = not self.show_paths
            elif key == pygame.K_d:
                self.deterministic = not self.deterministic
            elif key == pygame.K_b and self.cfg.limited_comm:
                self.show_belief = not self.show_belief

    # ------------------------------------------------------------ drawing

    def to_screen(self, x, y):
        """Map pixel coordinates (x right, y up) to screen coordinates"""
        k = self.map_rect.width / self.map_sizes[self.map_idx]
        return (float(self.map_rect.x + x * k), float(self.map_rect.y + (self.map_sizes[self.map_idx] - y) * k))

    def draw(self):
        self.screen.fill(WHITE)
        self.draw_map()
        self.draw_panel()
        pygame.display.flip()

    def draw_map(self):
        cfg = self.cfg
        state = jax.device_get(self.sim.state)
        size = self.map_sizes[self.map_idx]
        k = self.map_rect.width / size
        codes = self.sim.map_codes(self.selected, self.show_belief)[:size, :size]
        img = CODE_COLORS[codes][:, ::-1]           # indexed [x, y], y up -> pygame [x, y], y down
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(img))
        self.screen.blit(pygame.transform.scale(surf, self.map_rect.size), self.map_rect)

        pos = state.pos * cfg.pixels_per_meter
        radius = max(3.0, float(cfg.radius_p * k))
        if self.show_paths:
            for n, path in enumerate(self.paths):
                if len(path) > 1:
                    pts = [self.to_screen(*p) for p in path[-3000:]]
                    pygame.draw.lines(self.screen, ROBOT_COLORS[n % len(ROBOT_COLORS)], False, pts, 1)
        if self.show_lidar:
            angles = np.linspace(-cfg.lidar_fov / 2, cfg.lidar_fov / 2, cfg.lidar_rays) * np.pi / 180
            for n in range(cfg.num_agents):
                color = tuple(c // 2 + 127 for c in ROBOT_COLORS[n % len(ROBOT_COLORS)])
                start = self.to_screen(*pos[n])
                for a, d in zip(state.heading[n] + angles, state.lidar_dist[n]):
                    end = self.to_screen(pos[n, 0] + d * np.cos(a), pos[n, 1] + d * np.sin(a))
                    pygame.draw.line(self.screen, color, start, end, 1)
        if cfg.limited_comm:
            # communication range of the selected robot, links between robots in range
            comm_p = cfg.comm_range * cfg.pixels_per_meter
            self.screen.set_clip(self.map_rect)
            pygame.draw.circle(self.screen, ROBOT_COLORS[self.selected % len(ROBOT_COLORS)],
                               self.to_screen(*pos[self.selected]), comm_p * k, 1)
            self.screen.set_clip(None)
            for i in range(cfg.num_agents):
                for j in range(i + 1, cfg.num_agents):
                    if np.linalg.norm(pos[i] - pos[j]) <= comm_p:
                        pygame.draw.line(self.screen, GRAY, self.to_screen(*pos[i]), self.to_screen(*pos[j]), 2)
        for n in range(cfg.num_agents):
            color = ROBOT_COLORS[n % len(ROBOT_COLORS)]
            center = self.to_screen(*pos[n])
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, BLACK, center, radius, 3 if n == self.selected else 1)
            h = state.heading[n]
            tip = self.to_screen(pos[n, 0] + 2 * cfg.radius_p * np.cos(h), pos[n, 1] + 2 * cfg.radius_p * np.sin(h))
            pygame.draw.line(self.screen, BLACK, center, tip, 2)
        if self.show_crops:
            self.screen.set_clip(self.map_rect)
            self.draw_crops(pos[self.selected], state.heading[self.selected])
            self.screen.set_clip(None)
        pygame.draw.rect(self.screen, BLACK, self.map_rect, 1)

        if self.done:
            text = self.big_font.render(f'{self.result} - next map in {max(0, NEXT_MAP_DELAY - (time.perf_counter() - self.done_time)):.0f} s'
                                        if not self.paused else f'{self.result} (paused)', True, BLACK, PANEL_BG)
            self.screen.blit(text, text.get_rect(midtop=(self.map_rect.centerx, self.map_rect.y + 10)))

    def draw_panel(self):
        cfg = self.cfg
        state = jax.device_get(self.sim.state)
        r = self.panel_rect
        pygame.draw.rect(self.screen, PANEL_BG, r)
        coverage = state.coverage_px / state.free_space
        overlap = state.overlap_px / max(state.coverage_px, 1)
        speed = SPEEDS[self.speed_idx]
        lines = [
            f'map {self.map_idx + 1}/{len(self.map_names)}: {self.map_names[self.map_idx]}',
            f'robots {cfg.num_agents}   size {self.map_sizes[self.map_idx]} px',
            f'step {int(state.t)}   time {int(state.t) * cfg.step_size:.1f} s',
            f'coverage {100 * coverage:5.1f} %   overlap {100 * overlap:4.1f} %',
            f'collisions {int(state.collisions.sum())}',
            f'speed {"max" if math.isinf(speed) else f"x{speed}"}   {"PAUSED" if self.paused else ""}',
            f'actions {"deterministic" if self.deterministic else "sampled"}',
            (f'comm range {cfg.comm_range:g} m   [b] map: {f"belief of robot {self.selected}" if self.show_belief else "global"}'
             if cfg.limited_comm else 'comm range unlimited'),
            'robot  return   reward  coll.']
        for n in range(cfg.num_agents):
            mark = '>' if n == self.selected else ' '
            lines.append(f'{mark}{n:<4d} {state.ep_return[n]:8.1f} {self.last_rewards[n]:7.2f} {state.collisions[n]:5d}')
        y = r.y + 8
        for i, line in enumerate(lines):
            color = ROBOT_COLORS[(i - 9) % len(ROBOT_COLORS)] if i >= 9 else BLACK
            self.screen.blit(self.font.render(line, True, color), (r.x + 8, y))
            y += 18

        # Egocentric multi-scale observation of the selected robot
        y += 4
        hint = f'[c] crops {"on" if self.show_crops else "off"}   [m] {"composite" if self.show_scales else "separate scales"}'
        self.screen.blit(self.font.render(f'observation of robot {self.selected} (up = forward)   {hint}', True, BLACK),
                         (r.x + 8, y))
        y += 24
        area = pygame.Rect(r.x + 8, y, r.width - 16, r.bottom - y - 8)
        maps = np.asarray(self.sim.obs['maps'][self.selected]).astype(np.float32) / 255  # H x W x C
        if self.show_scales:
            self.draw_scales(maps, area)
        else:
            self.draw_composites(maps, area)

    # -------------------------------------------------------- observations

    def observation_kinds(self):
        """(name, channel within a scale) of the observation maps"""
        cfg = self.cfg
        kinds = [('coverage', 0), ('obstacles', 1)]
        if cfg.frontier_observation:
            kinds.append(('frontier', 2))
        if cfg.robot_observation:
            kinds.append(('robots', cfg.map_channels - 1))
        return kinds

    @staticmethod
    def colorize(kind, values):
        """Observation values in [0, 1] -> RGB in [0, 1], colors as in rlm/renderer.py"""
        img = np.ones(values.shape + (3,), dtype=np.float32)
        if kind == 'coverage':
            img[..., 0] = np.clip(1 - 1.5 * values, 0, 1)
            img[..., 1] = np.clip(2 - 1.5 * values, 0, 1)
            img[..., 2] = img[..., 0]
        elif kind == 'obstacles':
            img[:] = np.clip(1 - values, 0, 1)[..., None]
        elif kind == 'frontier':
            img[..., 1] = np.clip(1 - values, 0, 1)
        else:  # other robots
            img[..., 1] = np.clip(1 - values, 0, 1)
            img[..., 2] = img[..., 1]
        return img

    def blit_image(self, img, rect):
        """img: (rows, cols, 3) in [0, 1], row 0 at the top"""
        surf = pygame.surfarray.make_surface((255 * img + 0.5).astype(np.uint8).transpose(1, 0, 2))
        self.screen.blit(pygame.transform.scale(surf, rect.size), rect)
        pygame.draw.rect(self.screen, BLACK, rect.inflate(2, 2), 1)

    def draw_label(self, text, rect):
        label = self.label_font.render(text, True, BLACK)
        box = label.get_rect(topleft=(rect.x + 10, rect.y + 5)).inflate(8, 6)
        pygame.draw.rect(self.screen, LABEL_BG, box)
        self.screen.blit(label, label.get_rect(center=box.center))

    def composite(self, scales, side):
        """
        Multi-scale image at display resolution: the coarsest scale fills the
        panel and each finer scale is pasted at the center, at its true extent.
        scales: (num_maps, H, W), finest first
        """
        cfg = self.cfg
        img = np.zeros((side, side), dtype=np.float32)
        H = cfg.input_size
        for k in range(cfg.num_maps):
            size = int(round(side / cfg.scale_factor ** k))
            if size < 1:
                break
            idx = np.minimum((np.arange(size) + 0.5) * H / size, H - 1).astype(int)
            i1 = side // 2 - size // 2
            img[i1:i1 + size, i1:i1 + size] = scales[cfg.num_maps - 1 - k][np.ix_(idx, idx)]
        return img

    def draw_composites(self, maps, area):
        cfg = self.cfg
        kinds = self.observation_kinds()
        gap = 10
        side = int(min((area.width - gap) / 2, (area.height - gap) / 2))
        C = cfg.map_channels
        for i, (kind, c) in enumerate(kinds):
            rect = pygame.Rect(area.x + (i % 2) * (side + gap), area.y + (i // 2) * (side + gap), side, side)
            scales = np.stack([maps[..., s * C + c] for s in range(cfg.num_maps)])
            self.blit_image(self.colorize(kind, self.composite(scales, side)), rect)
            if self.show_crops:
                self.draw_scale_outlines(rect)
            pygame.draw.circle(self.screen, ROBOT_COLORS[self.selected % len(ROBOT_COLORS)], rect.center, 3)
            self.draw_label(f'{kind} observation', rect)

    def draw_scale_outlines(self, rect):
        """Region of each scale in a composite panel, thick edge = forward"""
        cfg = self.cfg
        side = rect.width
        for n in range(cfg.num_maps):
            size = int(round(side / cfg.scale_factor ** (cfg.num_maps - 1 - n)))
            if size < 2:
                continue
            i1 = side // 2 - size // 2
            box = pygame.Rect(rect.x + i1, rect.y + i1, size, size)
            color = CROP_COLORS[n % len(CROP_COLORS)]
            pygame.draw.rect(self.screen, color, box, 1)
            pygame.draw.line(self.screen, color, box.topleft, (box.right - 1, box.top), 3)

    def draw_scales(self, maps, area):
        """Each scale separately: rows are observation kinds, columns scales (finest first)"""
        cfg = self.cfg
        kinds = self.observation_kinds()
        gap, header_h = 8, 20
        C = cfg.map_channels
        tile = int(min((area.width - (cfg.num_maps - 1) * gap) / cfg.num_maps,
                       (area.height - header_h - (len(kinds) - 1) * gap) / len(kinds)))
        for n in range(cfg.num_maps):
            x = area.x + n * (tile + gap)
            color = CROP_COLORS[n % len(CROP_COLORS)]
            extent_m = cfg.input_size * cfg.scale_factor ** n * cfg.meters_per_pixel
            self.screen.blit(self.font.render(f'scale {n + 1}: {extent_m:g} m', True, color), (x, area.y))
            for row, (kind, c) in enumerate(kinds):
                rect = pygame.Rect(x, area.y + header_h + row * (tile + gap), tile, tile)
                self.blit_image(self.colorize(kind, maps[..., n * C + c]), rect)
                if self.show_crops:
                    pygame.draw.rect(self.screen, color, rect.inflate(2, 2), 2)
                    pygame.draw.line(self.screen, color, rect.topleft, (rect.right - 1, rect.top), 4)
                if n == 0:
                    self.draw_label(kind, rect)

    def draw_crops(self, pos_p, heading):
        """
        Regions of the map seen by each scale of the selected robot's
        observation, rotated with the robot. The thick edge is the forward
        direction, i.e. the top of the observation panels.
        """
        cfg = self.cfg
        f_vec = np.array([np.cos(heading), np.sin(heading)])
        r_vec = np.array([np.sin(heading), -np.cos(heading)])
        for n in range(cfg.num_maps):
            half = cfg.input_size / 2 * cfg.scale_factor ** n
            # front left, front right, back right, back left
            corners = [pos_p + half * (a * f_vec + b * r_vec) for a, b in [(1, -1), (1, 1), (-1, 1), (-1, -1)]]
            pts = [self.to_screen(*c) for c in corners]
            color = CROP_COLORS[n % len(CROP_COLORS)]
            pygame.draw.polygon(self.screen, color, pts, 2)
            pygame.draw.line(self.screen, color, pts[0], pts[1], 5)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--load', required=True, type=str, help='checkpoint directory')
    parser.add_argument('--maps', nargs='+', default=None,
                        help='globs of map images (default: mowing evaluation maps)')
    parser.add_argument('--map', default=0, type=int, help='index of the first map')
    parser.add_argument('--num_agents', default=None, type=int, help='default: as in training')
    parser.add_argument('--comm_range', default=None, type=float,
                        help='communication range [m], 0: unlimited (default: as in training)')
    parser.add_argument('--max_steps', default=5000, type=int)
    parser.add_argument('--goal_coverage', default=0.99, type=float)
    parser.add_argument('--deterministic', default=True, action=BooleanOptionalAction)
    parser.add_argument('--speed', default=4, type=float, help='initial speed, multiple of real time')
    parser.add_argument('--window', default=800, type=int, help='side of the map view [pixels]')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start', nargs='+', default=None, type=float,
                        help='fixed start poses instead of the random spawn: x y heading (meters, degrees) per robot')
    args = parser.parse_args()

    params, env_cfg, train_cfg = load_checkpoint(args.load)
    files = [f for p in args.maps for f in sorted(glob.glob(p))] if args.maps else eval_map_files()
    files = list(dict.fromkeys(files))
    assert len(files) > 0, 'No maps found'
    maps = [load_map(f) for f in files]
    overrides = env_cfg.to_dict()
    overrides.update(
        max_episode_steps=args.max_steps,
        max_non_new_steps=args.max_steps)
    if args.num_agents is not None:
        overrides['num_agents'] = args.num_agents
    if args.comm_range is not None:
        overrides['comm_range'] = args.comm_range
    env_cfg = EnvConfig.from_dict(overrides)

    actor, _ = make_networks(env_cfg, train_cfg)
    print(f'{len(files)} maps, {env_cfg.num_agents} robots, JAX device {jax.devices()[0]}', flush=True)
    if args.start is not None:
        assert len(args.start) == 3 * env_cfg.num_agents, '--start needs x y heading for each robot'
    sims = Simulations(env_cfg, actor, params['actor'], maps, args.goal_coverage, args.start)
    viewer = Viewer(sims, [os.path.basename(f) for f in files], [m[0].shape[0] for m in maps], args)
    try:
        viewer.run()
    except SystemExit:
        pygame.quit()


if __name__ == '__main__':
    main()
