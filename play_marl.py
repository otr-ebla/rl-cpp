"""
Interactive pygame viewer of a trained MAPPO policy: the robots mow the
evaluation maps step by step, like a video game.

Examples:
  python play_marl.py --load experiments/marl_4robots/checkpoint
  python play_marl.py --load experiments/marl_4robots/checkpoint --num_agents 8 --map 5
  python play_marl.py --load experiments/marl_4robots/checkpoint --maps "maps/giant_mowing_*.png"

Keys:
  space        pause / resume
  right arrow  one step (when paused)
  s / a        faster / slower (x1 = real time)
  n / p        next / previous map
  r            restart the episode on the same map
  tab          select the next robot (observation panel)
  l            toggle lidar rays
  t            toggle paths
  d            toggle deterministic / sampled actions
  q / Esc      quit

The map shows coverage (green, darker with more overlap), obstacles not yet
detected (gray), detected obstacles (black) and the frontier (magenta). The
panel on the right shows the multi-scale egocentric observation of the
selected robot, as received by the policy (up = forward).
"""
import argparse
import glob
import math
import os
import time
from argparse import BooleanOptionalAction

# Only allocate the GPU memory that is needed, so that the viewer can run next to a training
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from marl.config import EnvConfig
from marl.env import EnvParams, MultiMowerEnv, max_pool
from marl.maps import build_bank, canvas_size_for, eval_map_files, load_map
from marl.mappo import load_checkpoint, make_networks

ROBOT_COLORS = [(228, 26, 28), (55, 126, 184), (255, 127, 0), (152, 78, 163),
                (166, 86, 40), (247, 129, 191), (77, 175, 74), (0, 160, 160)]
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (110, 110, 110)
PANEL_BG = (240, 240, 240)

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

    def __init__(self, env_cfg, actor, actor_params, env_params):
        self.env = MultiMowerEnv(env_cfg)
        self.cfg = env_cfg
        env = self.env
        N = env_cfg.num_agents

        @jax.jit
        def reset(key, map_logits):
            obs, state = env.reset(key, env_params.replace(map_logits=map_logits))
            return obs, state

        @jax.jit
        def step(key, obs, state, deterministic):
            k_act, k_step, k_obs = jax.random.split(key, 3)
            mean, log_std = actor.apply(actor_params, obs['maps'], obs['lidar'])
            noise = jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            action = jnp.where(deterministic, mean, mean + noise)
            # no automatic reset, so that the final state can be shown
            state, rewards, done, info = env._step_state(k_step, state, action.reshape(N, -1))
            return env._observation(k_obs, state), state, rewards, done, info

        @jax.jit
        def map_codes(state):
            """(S, S) uint8 cell codes for display, see CODE_COLORS"""
            S = env_cfg.canvas_size
            ii = jnp.arange(S)[:, None]
            jj = jnp.arange(S)[None, :]
            outside = (ii >= state.size) | (jj >= state.size)
            border = (ii == 0) | (jj == 0) | (ii == state.size - 1) | (jj == state.size - 1)
            covered = state.overlap > 0
            # frontier as in MultiMowerEnv._observation
            known_b = (state.known | border | outside).astype(jnp.uint8)
            known_dil = max_pool(known_b, env_cfg.obstacle_dilation) > 0 if env_cfg.obstacle_dilation > 1 else known_b > 0
            cov_c = covered & ~known_dil
            frontier = (max_pool(cov_c.astype(jnp.uint8), 3) > 0) & ~(cov_c | known_dil)
            code = jnp.clip(state.overlap, 0, 4).astype(jnp.uint8)
            code = jnp.where(frontier, 7, code)
            code = jnp.where(state.obstacles & ~state.known, 5, code)
            code = jnp.where(state.known, 6, code)
            return jnp.where(outside, 8, code).astype(jnp.uint8)

        self._reset, self._step, self._map_codes = reset, step, map_codes

    def reset(self, key, map_idx, num_maps):
        logits = jnp.where(jnp.arange(num_maps) == map_idx, 0.0, -jnp.inf)
        self.obs, self.state = self._reset(key, logits)

    def step(self, key, deterministic):
        self.obs, self.state, rewards, done, info = self._step(key, self.obs, self.state, deterministic)
        return jax.device_get((rewards, done, info))

    def map_codes(self):
        return np.asarray(self._map_codes(self.state))


class Viewer:

    def __init__(self, sim: Simulation, map_names, map_sizes, args):
        self.sim = sim
        self.cfg = sim.cfg
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

        margin = 10
        self.map_rect = pygame.Rect(margin, margin, args.window, args.window)
        panel_w = 330
        self.panel_rect = pygame.Rect(self.map_rect.right + margin, margin, panel_w, args.window)
        pygame.init()
        self.screen = pygame.display.set_mode((self.panel_rect.right + margin, args.window + 2 * margin))
        pygame.display.set_caption('MARL mowing - [space] pause [s/a] speed [n/p] map [tab] robot [q] quit')
        self.font = pygame.font.SysFont('dejavusansmono,menlo,couriernew,monospace', 14)
        self.big_font = pygame.font.SysFont('dejavusans,helvetica,arial', 20, bold=True)
        self.new_episode()

    # ------------------------------------------------------------ episode

    def new_episode(self):
        self.key, k = jax.random.split(self.key)
        self.sim.reset(k, self.map_idx, len(self.map_names))
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
            elif key == pygame.K_RIGHT and self.paused and not self.done:
                self.step()
            elif key == pygame.K_s:
                self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1)
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
            elif key == pygame.K_l:
                self.show_lidar = not self.show_lidar
            elif key == pygame.K_t:
                self.show_paths = not self.show_paths
            elif key == pygame.K_d:
                self.deterministic = not self.deterministic

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
        codes = self.sim.map_codes()[:size, :size]
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
        for n in range(cfg.num_agents):
            color = ROBOT_COLORS[n % len(ROBOT_COLORS)]
            center = self.to_screen(*pos[n])
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, BLACK, center, radius, 3 if n == self.selected else 1)
            h = state.heading[n]
            tip = self.to_screen(pos[n, 0] + 2 * cfg.radius_p * np.cos(h), pos[n, 1] + 2 * cfg.radius_p * np.sin(h))
            pygame.draw.line(self.screen, BLACK, center, tip, 2)
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
            '',
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
        y += 8
        title = self.font.render(f'observation of robot {self.selected} (up = forward)', True, BLACK)
        self.screen.blit(title, (r.x + 8, y))
        y += 22
        maps = np.asarray(self.sim.obs['maps'][self.selected]).astype(np.float32) / 255  # H x W x C
        C = cfg.map_channels
        cell = (r.width - 3 * 8) // 2
        cell = min(cell, (r.bottom - y - 2 * 22) // 2)
        for s in range(cfg.num_maps):
            ch = maps[..., s * C:(s + 1) * C]
            img = np.ones(ch.shape[:2] + (3,), dtype=np.float32)
            img *= 1 - 0.6 * ch[..., :1] * np.array([1, 0.2, 1])        # coverage: green
            if cfg.frontier_observation:
                f = ch[..., 2:3]
                img = img * (1 - f) + f * np.array([1, 0, 1])          # frontier: magenta
            if cfg.robot_observation:
                rb = ch[..., -1:]
                img = img * (1 - rb) + rb * np.array([1, 0, 0])        # other robots: red
            img *= 1 - ch[..., 1:2]                                    # obstacles: black
            surf = pygame.surfarray.make_surface((255 * img).astype(np.uint8).transpose(1, 0, 2))
            x0 = r.x + 8 + (s % 2) * (cell + 8)
            y0 = y + (s // 2) * (cell + 22)
            self.screen.blit(pygame.transform.scale(surf, (cell, cell)), (x0, y0))
            pygame.draw.rect(self.screen, GRAY, (x0, y0, cell, cell), 1)
            # robot at the center, pointing up
            c = (x0 + cell / 2, y0 + cell / 2)
            pygame.draw.circle(self.screen, ROBOT_COLORS[self.selected % len(ROBOT_COLORS)], c, 3)
            label = self.font.render(f'scale {s} (x{cfg.scale_factor ** s})', True, BLACK)
            self.screen.blit(label, (x0, y0 + cell + 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--load', required=True, type=str, help='checkpoint directory')
    parser.add_argument('--maps', default=None, type=str, help='glob of map images (default: mowing evaluation maps)')
    parser.add_argument('--map', default=0, type=int, help='index of the first map')
    parser.add_argument('--num_agents', default=None, type=int, help='default: as in training')
    parser.add_argument('--max_steps', default=5000, type=int)
    parser.add_argument('--goal_coverage', default=0.99, type=float)
    parser.add_argument('--deterministic', default=True, action=BooleanOptionalAction)
    parser.add_argument('--speed', default=4, type=float, help='initial speed, multiple of real time')
    parser.add_argument('--window', default=800, type=int, help='side of the map view [pixels]')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    params, env_cfg, train_cfg = load_checkpoint(args.load)
    files = sorted(glob.glob(args.maps)) if args.maps else eval_map_files()
    assert len(files) > 0, 'No maps found'
    maps = [load_map(f) for f in files]
    overrides = env_cfg.to_dict()
    overrides.update(
        canvas_size=canvas_size_for([m[0].shape[0] for m in maps], env_cfg),
        max_episode_steps=args.max_steps,
        max_non_new_steps=args.max_steps)
    if args.num_agents is not None:
        overrides['num_agents'] = args.num_agents
    env_cfg = EnvConfig.from_dict(overrides)

    actor, _ = make_networks(env_cfg, train_cfg)
    bank = build_bank(maps, env_cfg)
    env_params = EnvParams(bank=bank, map_logits=jnp.zeros(len(maps)), goal_coverage=jnp.float32(args.goal_coverage))
    print(f'{len(files)} maps, {env_cfg.num_agents} robots, JAX device {jax.devices()[0]}', flush=True)
    sim = Simulation(env_cfg, actor, params['actor'], env_params)
    viewer = Viewer(sim, [os.path.basename(f) for f in files], [m[0].shape[0] for m in maps], args)
    try:
        viewer.run()
    except SystemExit:
        pygame.quit()


if __name__ == '__main__':
    main()
