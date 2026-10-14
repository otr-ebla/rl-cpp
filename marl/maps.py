"""
Maps for the multi-agent environment.

All maps are placed in the top-left corner of a fixed size canvas, so that
maps of different sizes can be stacked in one array (the map bank) and
simulated in parallel. Everything that only depends on the map, i.e. the
dilated obstacles and the collision map, is precomputed once per map here,
on the host, instead of at every reset on the GPU.
"""
import glob
import os

import jax.numpy as jnp
import numpy as np
from flax import struct
from PIL import Image
from scipy import ndimage

from marl.config import EnvConfig


RANDOM_GROUP = -1
EVAL_GROUP = -2


@struct.dataclass
class MapBank:
    """
    Stacked maps, K maps of the canvas size S x S.

    obstacles:  (K, S, S) bool, all obstacles (known and unknown) inside the map
    known:      (K, S, S) bool, obstacles known at the start of an episode
                (outside the map: the obstacle pad value)
    dilated:    (K, S, S) bool, dilated obstacles and map border, used to
                compute coverage metrics (free space = not dilated)
    blocked:    (K, S, S) bool, robot center positions in collision with an
                obstacle (obstacles dilated by the robot radius)
    size:       (K,) int32, map size [pixels]
    free_space: (K,) float32, number of free (not dilated) pixels
    """
    obstacles: jnp.ndarray
    known: jnp.ndarray
    dilated: jnp.ndarray
    blocked: jnp.ndarray
    size: jnp.ndarray
    free_space: jnp.ndarray

    @property
    def num_maps(self):
        return self.size.shape[0]


def load_map(filename):
    """
    Loads a map image, see MowerEnv._load_map.
    Returns (known, unknown) obstacle maps as bool arrays indexed [x, y].
    """
    img = np.array(Image.open(filename).convert('L'))
    img = np.fliplr(img.transpose(1, 0))
    assert img.ndim == 2 and img.shape[0] == img.shape[1], f'Invalid map shape: {filename}'
    assert set(np.unique(img)).issubset([0, 128, 255]), f'Invalid map image: {filename}'
    return img == 0, img == 128


def train_map_files(group, maps_dir='maps'):
    return sorted(glob.glob(os.path.join(maps_dir, f'train_{group}_*')))


def eval_map_files(maps_dir='maps', task='mowing'):
    files = glob.glob(os.path.join(maps_dir, f'eval_{task}_*'))
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0].rsplit('_', 1)[1]))


def _disc(radius):
    r = int(radius)
    ii, jj = np.mgrid[-r:r+1, -r:r+1]
    return ii ** 2 + jj ** 2 <= radius ** 2


def random_floor_plan(rng, known, mower_radius, pixels_per_meter):
    """NumPy port of MowerEnv._randomize_floor_plan, draws into known."""
    size_p = known.shape[0]
    min_room_size_p = int(10 * mower_radius * pixels_per_meter)
    max_room_size_p = int(32 * mower_radius * pixels_per_meter)
    min_wall_thickness_p = 2
    max_wall_thickness_p = int(2 * mower_radius * pixels_per_meter)
    min_gap = int(4 * mower_radius * pixels_per_meter)
    max_gap = int(8 * mower_radius * pixels_per_meter)
    if size_p <= 2 * min_room_size_p:
        return
    room_size_p = int(rng.integers(min_room_size_p, max_room_size_p + 1))
    num_walls = max(1, int(size_p / room_size_p) - 1)
    room_size_p = int(size_p / (num_walls + 1))
    wall_thickness_p = int(rng.integers(min_wall_thickness_p, max_wall_thickness_p + 1))
    vertical_stop = rng.uniform() < 0.5
    for n in range(num_walls):
        i1 = room_size_p * (n + 1) - wall_thickness_p // 2
        i2 = room_size_p * (n + 1) + wall_thickness_p
        if rng.uniform() < 0.9:
            known[i1:i2, :] = True
        if rng.uniform() < 0.9:
            known[:, i1:i2] = True
        stop_placed = False
        for m in range(num_walls + 1):
            g_min = min_gap
            g_max = min(max_gap, room_size_p - 2 * wall_thickness_p)
            j_min = room_size_p * m + wall_thickness_p
            j_max = room_size_p * (m + 1) - wall_thickness_p
            p_stop = 1 / (num_walls + 1 - m)
            place_stop = False
            if not stop_placed and rng.uniform() < p_stop:
                place_stop = True
                stop_placed = True
            if not vertical_stop or not place_stop:
                gap = int(rng.integers(g_min, g_max + 1))
                j1 = int(rng.integers(j_min, j_max - gap + 1))
                known[i1:i2, j1:j1+gap] = False
            if vertical_stop or not place_stop:
                gap = int(rng.integers(g_min, g_max + 1))
                j1 = int(rng.integers(j_min, j_max - gap + 1))
                known[j1:j1+gap, i1:i2] = False


def random_circular_obstacles(
        rng, known, unknown, use_known, use_unknown, max_known, max_unknown,
        obstacle_radius, mower_radius, pixels_per_meter):
    """NumPy port of MowerEnv._randomize_circular_obstacles."""
    size_p = known.shape[0]
    size_m = size_p / pixels_per_meter
    known_pos = rng.uniform(size=(max_known, 2))
    unknown_pos = rng.uniform(size=(max_unknown, 2))
    radius = 2 * mower_radius + obstacle_radius
    if size_m <= 4 * radius:
        return
    disc = _disc(int(obstacle_radius * pixels_per_meter))
    rd = disc.shape[0] // 2

    def free(pos_m):
        i1 = max(0, int((pos_m[0] - radius) * pixels_per_meter - 10))
        i2 = min(size_p, int((pos_m[0] + radius) * pixels_per_meter + 10))
        j1 = max(0, int((pos_m[1] - radius) * pixels_per_meter - 10))
        j2 = min(size_p, int((pos_m[1] + radius) * pixels_per_meter + 10))
        return not known[i1:i2, j1:j2].any() and not unknown[i1:i2, j1:j2].any()

    def draw(target, pos_m):
        ci, cj = (pos_m * pixels_per_meter).astype(int)
        i1, i2 = max(0, ci - rd), min(size_p, ci + rd + 1)
        j1, j2 = max(0, cj - rd), min(size_p, cj + rd + 1)
        target[i1:i2, j1:j2] |= disc[i1-ci+rd:i2-ci+rd, j1-cj+rd:j2-cj+rd]

    for n in range(max(max_known, max_unknown)):
        if use_known and n < max_known:
            pos_m = 2 * radius + known_pos[n] * (size_m - 4 * radius)
            if free(pos_m):
                draw(known, pos_m)
        if use_unknown and n < max_unknown:
            pos_m = 2 * radius + unknown_pos[n] * (size_m - 4 * radius)
            if free(pos_m):
                draw(unknown, pos_m)


def random_map(
        rng, cfg: EnvConfig, min_size_p=64, max_size_p=200, p_use_floor_plans=0.7,
        p_use_known_obstacles=0.7, p_use_unknown_obstacles=0.7,
        max_known_obstacles=100, max_unknown_obstacles=100, obstacle_radius=0.25):
    """
    Random map like MowerEnv._create_training_env, with the default sizes of
    the mowing task. Returns (known, unknown, has_floor_plan, has_obstacles).
    """
    max_size_p = min(max_size_p, cfg.canvas_size)
    size_p = int(rng.integers(min_size_p, max_size_p + 1))
    known = np.zeros((size_p, size_p), dtype=bool)
    unknown = np.zeros((size_p, size_p), dtype=bool)
    use_floor_plan = rng.uniform() < p_use_floor_plans
    if use_floor_plan:
        random_floor_plan(rng, known, cfg.mower_radius, cfg.pixels_per_meter)
    use_known = max_known_obstacles > 0 and rng.uniform() < p_use_known_obstacles
    use_unknown = max_unknown_obstacles > 0 and rng.uniform() < p_use_unknown_obstacles
    if use_known or use_unknown:
        random_circular_obstacles(
            rng, known, unknown, use_known, use_unknown, max_known_obstacles,
            max_unknown_obstacles, obstacle_radius, cfg.mower_radius, cfg.pixels_per_meter)
    return known, unknown, use_floor_plan, use_known or use_unknown


def prepare_map(known, unknown, cfg: EnvConfig):
    """
    Places one map on the canvas and precomputes the static maps.
    Returns a dict of numpy arrays, see MapBank.
    """
    size = known.shape[0]
    S = cfg.canvas_size
    assert size <= S, f'Map of size {size} does not fit in canvas of size {S}, increase canvas_size'

    obstacles = np.zeros((S, S), dtype=bool)
    obstacles[:size, :size] = known | unknown
    outside = np.ones((S, S), dtype=bool)
    outside[:size, :size] = False

    # Known obstacles at the start, see MowerEnv._set_all_unknown
    if cfg.all_unknown:
        obstacles[0, :size] = True; obstacles[size-1, :size] = True
        obstacles[:size, 0] = True; obstacles[:size, size-1] = True
        known_init = np.zeros((S, S), dtype=bool)
    else:
        known_init = outside.copy()
        known_init[:size, :size] = known

    # Dilated obstacles including the map border, see MowerEnv._initialize_metrics
    border = obstacles.copy()
    border[0, :size] = True; border[size-1, :size] = True
    border[:size, 0] = True; border[:size, size-1] = True
    if cfg.obstacle_dilation > 1:
        dilated = ndimage.maximum_filter(border, size=cfg.obstacle_dilation, mode='constant', cval=False)
    else:
        dilated = border
    dilated = dilated | outside
    free_space = float((~dilated).sum())

    # Robot center positions in collision, see MowerEnv._is_obstacle_collision
    disc = _disc(int(cfg.radius_p))
    blocked = ndimage.binary_dilation(obstacles, structure=disc) | outside

    return dict(obstacles=obstacles, known=known_init, dilated=dilated,
                blocked=blocked, size=size, free_space=free_space)


def build_bank(maps, cfg: EnvConfig):
    """
    maps: list of (known, unknown) obstacle maps
    Returns a MapBank on the default device.
    """
    prepared = [prepare_map(known, unknown, cfg) for known, unknown in maps]
    return MapBank(
        obstacles=jnp.asarray(np.stack([p['obstacles'] for p in prepared])),
        known=jnp.asarray(np.stack([p['known'] for p in prepared])),
        dilated=jnp.asarray(np.stack([p['dilated'] for p in prepared])),
        blocked=jnp.asarray(np.stack([p['blocked'] for p in prepared])),
        size=jnp.asarray([p['size'] for p in prepared], dtype=jnp.int32),
        free_space=jnp.asarray([p['free_space'] for p in prepared], dtype=jnp.float32))


def concat_banks(banks):
    return MapBank(*[jnp.concatenate([getattr(b, f) for b in banks]) for f in
                     ['obstacles', 'known', 'dilated', 'blocked', 'size', 'free_space']])


def canvas_size_for(map_sizes, cfg: EnvConfig):
    """Smallest valid canvas size that fits all maps"""
    m = cfg.scale_factor ** (cfg.num_maps - 1)
    m = m * cfg.global_size // np.gcd(m, cfg.global_size)
    return int(-(-max(map_sizes) // m) * m)
