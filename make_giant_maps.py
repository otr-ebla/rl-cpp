"""
Generates evaluation maps that are larger than the region covered by the
largest scale of the multi-scale maps (input_size * scale_factor^(num_maps-1)
pixels, i.e. 2048 pixels = 76.8 meters with the default parameters).

Two methods are available:
  tile:   a mosaic of the original evaluation maps for the task, randomly
          rotated/flipped, packed row by row and connected by corridors and
          door openings. Local structures are thus identical to the original
          evaluation maps.
  random: the procedural generator used for the training maps (random floor
          plans and circular obstacles), with the same obstacle density.

Regions of free space that the agent cannot reach are filled with obstacles.
The map size is rounded to a multiple of 64 pixels, so that the observations
can be computed efficiently, see MowerEnv._get_downsampled_map.

Example:
  python make_giant_maps.py --task exploration --method tile --size 108 --num 5
  python eval.py --load weights/exploration --maps "maps/giant_exploration_tile_*.png"
"""
import argparse
import glob
import math
import os
import random

import cv2
import numpy as np

from rlm.mower_env import MowerEnv


FREE, UNKNOWN, OBSTACLE = 255, 128, 0


def tile_map(task, size_p, ppm, mower_radius):
    """
    Mosaic of the original evaluation maps, randomly rotated/flipped and
    packed row by row, separated by free corridors. Door openings are cut in
    the outer walls of each map so that it can be entered from the corridors.
    """
    sources = sorted(glob.glob(f'maps/eval_{task}_*.png'))
    assert len(sources) > 0, f'No evaluation maps found for {task}'
    sources = [cv2.imread(f, flags=cv2.IMREAD_GRAYSCALE) for f in sources]
    corridor = int(1.5 * ppm)
    door = int(1.2 * ppm)
    img = np.full((size_p, size_p), FREE, dtype=np.uint8)
    y = corridor
    while y < size_p:
        x = corridor
        row_height = 0
        while x < size_p:
            src = np.rot90(random.choice(sources), k=random.randint(0, 3))
            if random.uniform(0, 1) < 0.5:
                src = np.fliplr(src)
            src = _cut_doors(src.copy(), door, int(2 * mower_radius * ppm))
            # crop the map at the borders of the mosaic
            h = min(src.shape[0], size_p - y)
            w = min(src.shape[1], size_p - x)
            img[y:y+h, x:x+w] = src[:h, :w]
            x += src.shape[1] + corridor
            row_height = max(row_height, src.shape[0])
        y += row_height + corridor
    return img


def _cut_doors(src, width, depth_free, doors_per_side=2, max_depth=40):
    """
    Cuts door openings through the outer wall on each side of a map, at
    positions where free space is reached within max_depth pixels.
    """
    n = src.shape[0]
    for side in range(4):
        view = np.rot90(src, k=side) # view of the side as the top rows
        candidates = []
        for x in range(0, n - width):
            column = view[:max_depth, x:x+width]
            free_rows = np.where((column == FREE).all(axis=1))[0]
            if len(free_rows) > 0 and free_rows[0] + depth_free < max_depth:
                candidates.append((x, free_rows[0]))
        random.shuffle(candidates)
        placed = []
        for x, depth in candidates:
            if len(placed) == doors_per_side:
                break
            if all(abs(x - p) > 2 * width for p in placed):
                view[:depth + depth_free, x:x+width] = FREE
                placed.append(x)
    return src


def random_map(task, size_p, ppm, mower_radius):
    """
    Uses the procedural generator of the training maps, with the number of
    circular obstacles scaled to keep the same density.
    """
    env = MowerEnv(exploration=task == 'exploration', meters_per_pixel=1 / ppm, mower_radius=mower_radius)
    typical_size_p = (env.min_size_p + env.max_size_p) / 2
    density_scale = (size_p / typical_size_p) ** 2
    env.max_known_obstacles = int(env.max_known_obstacles * density_scale)
    env.max_unknown_obstacles = int(env.max_unknown_obstacles * density_scale)
    env.size_p = size_p
    env.size_m = size_p / env.pixels_per_meter
    env.known_obstacle_map = np.zeros((size_p, size_p), dtype=float)
    env.unknown_obstacle_map = np.zeros((size_p, size_p), dtype=float)
    env.use_known_obstacles = True
    env.use_unknown_obstacles = True
    env._randomize_floor_plan()
    env._randomize_circular_obstacles()
    img = np.full((size_p, size_p), FREE, dtype=np.uint8)
    img[env.unknown_obstacle_map > 0] = UNKNOWN
    img[env.known_obstacle_map > 0] = OBSTACLE
    return img


def fill_unreachable(img, radius_p):
    """
    Fills free space that the agent cannot reach from the largest reachable
    region, taking the size of the agent into account.
    """
    free = (img == FREE).astype(np.uint8)
    free[0, :] = 0; free[-1, :] = 0; free[:, 0] = 0; free[:, -1] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_p + 1,)*2)
    centers = cv2.erode(free, kernel) # positions where the agent fits
    num, labels, stats, _ = cv2.connectedComponentsWithStats(centers, connectivity=8)
    assert num > 1, 'No free space large enough for the agent'
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    reachable = cv2.dilate((labels == largest).astype(np.uint8), kernel)
    img = img.copy()
    img[(img == FREE) & (reachable == 0)] = OBSTACLE
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', default='exploration', choices=['exploration', 'mowing'])
    parser.add_argument('--method', default='tile', choices=['tile', 'random'])
    parser.add_argument('--size', default=108, type=float, help='side length [m], rounded to a multiple of 64 pixels')
    parser.add_argument('--num', default=5, type=int, help='number of maps')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out', default='maps', type=str)
    parser.add_argument('--meters_per_pixel', default=0.0375, type=float)
    parser.add_argument('--mower_radius', default=0.15, type=float)
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--num_maps', default=4, type=int)
    parser.add_argument('--scale_factor', default=4, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    ppm = 1 / args.meters_per_pixel
    size_p = max(64, int(round(args.size * ppm / 64)) * 64)
    reach_p = args.input_size * args.scale_factor ** (args.num_maps - 1)
    print(f'Map side: {size_p} pixels = {size_p / ppm:g} m', flush=True)
    print(f'Largest scale: {reach_p} pixels = {reach_p / ppm:g} m', flush=True)
    if size_p <= reach_p:
        print('Warning: the largest scale is not smaller than the map side', flush=True)

    os.makedirs(args.out, exist_ok=True)
    for n in range(1, args.num + 1):
        if args.method == 'tile':
            img = tile_map(args.task, size_p, ppm, args.mower_radius)
        else:
            img = random_map(args.task, size_p, ppm, args.mower_radius)
        img = fill_unreachable(img, int(math.ceil(args.mower_radius * ppm)))
        filename = os.path.join(args.out, f'giant_{args.task}_{args.method}_{n}.png')
        # undo the transform in MowerEnv._load_map, so that the map is loaded as generated
        cv2.imwrite(filename, np.fliplr(img).T)
        print(f'  {filename}: {100 * (img == FREE).mean():.0f}% free space', flush=True)


if __name__ == '__main__':
    main()
