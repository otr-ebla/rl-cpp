import math
import os
import time

import cv2
import numpy as np

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame


# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIDAR_COLOR = (0, 0, 255)
PATH_COLOR = (255, 255, 0)
LABEL_BG = (178, 178, 178)
CROP_COLORS = [(230, 25, 75), (245, 130, 48), (0, 130, 200), (145, 30, 180), (70, 240, 240), (240, 50, 230)]



def _map_colors():
    """
    Color lookup table for the full map: entries 0-255 are quantized coverage
    values in [0, 1], followed by unknown obstacles, known obstacles and
    frontier points.
    """
    overlap_tanh = np.tanh(0.5 * np.linspace(0, 1, 256))
    colors = np.empty((259, 3))
    colors[:256, 0] = np.clip(1 - 1.5*overlap_tanh, 0, 1)
    colors[:256, 1] = np.clip(2 - 1.5*overlap_tanh, 0, 1)
    colors[:256, 2] = colors[:256, 0]
    colors[256] = 0.5
    colors[257] = 0
    colors[258] = [1, 0, 1]
    return (255 * colors + 0.5).astype(np.uint8)


MAP_COLORS = _map_colors()

# Simulation speeds, as multiples of real time, cycled with the 's' key
SPEEDS = [1, 2, 4, 8, math.inf]
MAX_FPS_AT_INF = 30


class PygameRenderer:
    """
    Interactive pygame renderer for MowerEnv. Mimics the layout of the
    matplotlib renderer:
      'limited': only the full map
      'full':    full map + multi-scale coverage/obstacle/frontier observations

    Keys:
      c:       toggle the multi-scale crops (centered on and rotated with the agent)
      s:       cycle simulation speed x1 (real time), x2, x4, x8, max
      q/Esc:   quit
    """

    def __init__(self, env, mode):
        assert mode in ['full', 'limited']
        self.env = env
        self.mode = mode
        self.show_crops = False
        self.speed_idx = 0
        self.next_step_time = None
        self.last_draw_time = 0

        # Layout, similar to the matplotlib mosaics 'A' and 'AABC;AAD.'
        margin = 10
        if mode == 'limited':
            map_side = 780
            self.map_rect = pygame.Rect(margin, margin, map_side, map_side)
            size = (map_side + 2 * margin, map_side + 2 * margin)
            self.obs_rects = {}
            self.hud_rect = None
        else:
            title_h = 40
            cell = 280
            map_side = 2 * cell + margin
            self.map_rect = pygame.Rect(margin, title_h, map_side, map_side)
            x1 = self.map_rect.right + margin
            x2 = x1 + cell + margin
            y2 = title_h + cell + margin
            self.obs_rects = {
                'coverage': pygame.Rect(x1, title_h, cell, cell),
                'obstacles': pygame.Rect(x2, title_h, cell, cell),
                'frontier': pygame.Rect(x1, y2, cell, cell)}
            self.hud_rect = pygame.Rect(x2, y2, cell, cell)
            size = (x2 + cell + margin, title_h + map_side + margin)

        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.title_font = pygame.font.SysFont('dejavusans,helvetica,arial', 21)
        self.label_font = pygame.font.SysFont('dejavuserif,timesnewroman,times,serif', 14)
        self.hud_font = pygame.font.SysFont('dejavusansmono,menlo,couriernew,monospace', 14)
        self._update_caption()

    @property
    def speed(self):
        return SPEEDS[self.speed_idx]

    def _speed_str(self):
        return 'max' if math.isinf(self.speed) else f'x{self.speed}'

    def _update_caption(self):
        pygame.display.set_caption(f'MowerEnv - speed {self._speed_str()} - [c] crops [s] speed [q] quit')

    def close(self):
        pygame.quit()

    def render(self):
        """
        Called once per environment step. Handles input, paces the simulation
        according to the selected speed, and draws the current state.
        """
        env = self.env
        env.path.append(env.position_m)
        self._handle_events()

        # Pace the simulation
        now = time.perf_counter()
        if math.isinf(self.speed):
            self.next_step_time = None
            if now - self.last_draw_time < 1 / MAX_FPS_AT_INF:
                return
        else:
            interval = env.step_size / self.speed
            if self.next_step_time is None or self.next_step_time < now - interval:
                self.next_step_time = now
            if self.next_step_time > now:
                time.sleep(self.next_step_time - now)
            self.next_step_time += interval
        self.last_draw_time = time.perf_counter()

        self._draw()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.close()
                    raise SystemExit
                elif event.key == pygame.K_c:
                    self.show_crops = not self.show_crops
                elif event.key == pygame.K_s:
                    self.speed_idx = (self.speed_idx + 1) % len(SPEEDS)
                    self.next_step_time = None
                    self._update_caption()

    def _draw(self):
        env = self.env
        self.screen.fill(WHITE)
        self._draw_map()
        if self.mode == 'full':
            self._draw_observations()
            coverage = round(100 * env.coverage_in_percent)
            overlap = round(100 * env.overlap_in_percent)
            title = self.title_font.render(
                f'{env.elapsed_steps} steps' +
                f', {round(env.elapsed_time, 1)} s'
                f', {env.num_collisions} collisions' +
                f', {coverage}% coverage' +
                f', {overlap}% overlap', True, BLACK)
            self.screen.blit(title, title.get_rect(midtop=(self.screen.get_width() // 2, 8)))
            self._draw_label('full map', self.map_rect)
        self._draw_hud()
        pygame.display.flip()

    def _to_screen(self, i, j):
        """
        Converts continuous map pixel coordinates (i, j) to screen coordinates.
        The map is displayed with i to the right and j upwards.
        """
        k = self.map_rect.width / self.env.size_p
        return (self.map_rect.x + i * k, self.map_rect.y + (self.env.size_p - j) * k)

    def _draw_map(self):
        env = self.env
        rect = self.map_rect
        k = rect.width / env.size_p

        # Environment image with coverage/obstacles/frontier/agent
        # Colors are looked up from a table indexed by the quantized coverage,
        # with three extra entries for unknown obstacles, known obstacles and
        # frontier points (in increasing priority)
        label = (255 * env.coverage_map + 0.5).astype(np.uint16)
        np.putmask(label, env.unknown_obstacle_map > 0, 256)
        np.putmask(label, env.known_obstacle_map > 0, 257)
        np.putmask(label, env.frontier_map > 0, 258)
        img = np.take(MAP_COLORS, label, axis=0)
        cv2.circle(
            img,
            center=np.flip(env.position_p).astype(np.int32),
            radius=int(env.mower_radius * env.pixels_per_meter),
            color=(255, 0, 0),
            thickness=cv2.FILLED,
            lineType=env.line_type)
        surf = pygame.surfarray.make_surface(img[:, ::-1])
        self.screen.blit(pygame.transform.scale(surf, rect.size), rect)

        self.screen.set_clip(rect)

        # Path
        path = env.pixels_per_meter * np.array(env.path)
        if len(path) > 1:
            pts = [self._to_screen(p[0], p[1]) for p in path]
            pygame.draw.lines(self.screen, PATH_COLOR, False, pts, 2)

        # Lidar detections
        start = self._to_screen(*(env.pixels_per_meter * env.position_m))
        for n in range(env.lidar_rays):
            end = self._to_screen(env.lidar_pts[n, 0] + 0.5, env.lidar_pts[n, 1] + 0.5)
            pygame.draw.aaline(self.screen, LIDAR_COLOR, start, end)

        # Arrow for the heading
        length = env.pixels_per_meter * 2 * env.mower_radius
        tip_i = env.pixels_per_meter * env.position_m[0] + length * math.cos(env.heading)
        tip_j = env.pixels_per_meter * env.position_m[1] + length * math.sin(env.heading)
        tip = self._to_screen(tip_i, tip_j)
        pygame.draw.aaline(self.screen, BLACK, start, tip)
        head = max(k, 5)
        dx, dy = math.cos(env.heading), -math.sin(env.heading)
        pygame.draw.polygon(self.screen, BLACK, [
            (tip[0] + head * dx, tip[1] + head * dy),
            (tip[0] - 0.5 * head * dy, tip[1] + 0.5 * head * dx),
            (tip[0] + 0.5 * head * dy, tip[1] - 0.5 * head * dx)])

        if self.show_crops:
            self._draw_crops()

        self.screen.set_clip(None)
        pygame.draw.rect(self.screen, BLACK, rect.inflate(2, 2), 1)

    def _draw_crops(self):
        """
        Draws the regions of the full map covered by each scale of the
        multi-scale maps. The thick edge is the forward direction of the agent,
        which is the upwards direction in the observation panels.
        """
        env = self.env
        w = env.input_size
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, w, 1], [0, w, 1]], dtype=float).T
        for n in range(env.num_maps):
            scale = min(env.scale_factor ** n, env.size_p)
            inv = np.linalg.inv(env._get_transform_matrix(scale))
            xy = scale * (inv @ corners)[:2] # map coordinates, x = j and y = i
            pts = [self._to_screen(xy[1, c], xy[0, c]) for c in range(4)]
            color = CROP_COLORS[n % len(CROP_COLORS)]
            pygame.draw.polygon(self.screen, color, pts, 2)
            pygame.draw.line(self.screen, color, pts[1], pts[2], 5)

    def _draw_observations(self):
        env = self.env
        ob = env._get_latest_observation()
        side = self.obs_rects['coverage'].width

        coverage = self._multi_scale_image(ob['coverage'], side)
        img = np.empty((side, side, 3), dtype=np.float32)
        img[:, :, 0] = np.clip(1 - 1.5*coverage, 0, 1)
        img[:, :, 1] = np.clip(2 - 1.5*coverage, 0, 1)
        img[:, :, 2] = img[:, :, 0]
        self._blit_image(img, self.obs_rects['coverage'])
        self._draw_label('coverage observation', self.obs_rects['coverage'])

        obstacles = self._multi_scale_image(ob['obstacles'], side)
        img = np.repeat(np.clip(1 - obstacles, 0, 1)[:, :, None], 3, axis=2)
        self._blit_image(img, self.obs_rects['obstacles'])
        self._draw_label('obstacle observation', self.obs_rects['obstacles'])

        if env.frontier_observation:
            frontier = self._multi_scale_image(ob['frontier'], side)
            img = np.ones((side, side, 3), dtype=np.float32)
            img[:, :, 1] = np.clip(1 - frontier, 0, 1)
            self._blit_image(img, self.obs_rects['frontier'])
            self._draw_label('frontier observation', self.obs_rects['frontier'])

    def _multi_scale_image(self, ms_map, side):
        """
        Same as MowerEnv._get_image_from_multi_scale_map, but directly at the
        display resolution instead of the (much larger) full resolution.
        """
        env = self.env
        img = np.zeros((side, side), dtype=np.float32)
        for n in range(env.num_maps):
            size = int(round(side / env.scale_factor ** n))
            if size < 1:
                break
            interp = cv2.INTER_AREA if size < env.input_size else cv2.INTER_NEAREST
            map = cv2.resize(ms_map[env.num_maps-n-1].astype(np.float32), (size,)*2, interpolation=interp)
            i1 = side // 2 - size // 2
            img[i1:i1+size, i1:i1+size] = map
        return img

    def _blit_image(self, img, rect):
        img = (255 * img + 0.5).astype(np.uint8)
        self.screen.blit(pygame.surfarray.make_surface(img[:, ::-1]), rect)
        pygame.draw.rect(self.screen, BLACK, rect.inflate(2, 2), 1)

    def _draw_label(self, text, rect):
        label = self.label_font.render(text, True, BLACK)
        box = label.get_rect(topleft=(rect.x + 10, rect.y + 5)).inflate(8, 6)
        pygame.draw.rect(self.screen, LABEL_BG, box)
        self.screen.blit(label, label.get_rect(center=box.center))

    def _draw_hud(self):
        lines = [f'speed {self._speed_str()}',
                 '[s] speed  [c] crops',
                 f'crops {"on" if self.show_crops else "off"}  [q] quit']
        if self.hud_rect is not None:
            x, y = self.hud_rect.x + 10, self.hud_rect.bottom - 10 - len(lines) * 18
            for line in lines:
                self.screen.blit(self.hud_font.render(line, True, (90, 90, 90)), (x, y))
                y += 18
        else:
            text = self.hud_font.render(f'{self._speed_str()}  [s] speed  [c] crops', True, BLACK)
            box = text.get_rect(bottomleft=(self.map_rect.x + 6, self.map_rect.bottom - 6)).inflate(8, 4)
            bg = pygame.Surface(box.size, pygame.SRCALPHA)
            bg.fill((255, 255, 255, 190))
            self.screen.blit(bg, box)
            self.screen.blit(text, text.get_rect(center=box.center))
