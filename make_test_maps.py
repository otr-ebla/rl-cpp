"""
Generate held-out random test maps for the multi-agent evaluation, with the
generator of the training maps (marl.maps.random_map) and a seed that is not
used in training (training uses the training seed, 0, 1, ...).

  python make_test_maps.py
  python make_test_maps.py --name test_large --num 100 --min_size 320 --max_size 640

Writes maps/<name>_<i>.png. The defaults give the two sets of collect_marl.py:
  test_small: 64-200 pixels (2.4-7.5 m), the sizes of the random training maps
  test_large: 320-640 pixels (12-24 m), larger than the training maps, where a
              limited communication range matters
"""
import argparse
import os

import numpy as np

from marl.config import EnvConfig
from marl.maps import load_map, random_map, save_map

SETS = {'test_small': (64, 200, 1_000_001), 'test_large': (320, 640, 1_000_002)}


def make(name, num, min_size, max_size, seed, maps_dir):
    rng = np.random.default_rng(seed)
    cfg = EnvConfig(canvas_size=-(-max_size // 64) * 64)
    for i in range(1, num + 1):
        known, unknown, _, _ = random_map(rng, cfg, min_size_p=min_size, max_size_p=max_size)
        filename = os.path.join(maps_dir, f'{name}_{i}.png')
        save_map(filename, known, unknown)
        k, u = load_map(filename)
        assert np.array_equal(k, known) and np.array_equal(u, unknown)
    print(f'{num} maps {name}_*.png, {min_size}-{max_size} pixels, seed {seed}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default=None, type=str, help='one set (default: test_small and test_large)')
    parser.add_argument('--num', default=100, type=int)
    parser.add_argument('--min_size', default=None, type=int)
    parser.add_argument('--max_size', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--maps_dir', default='maps', type=str)
    args = parser.parse_args()
    names = [args.name] if args.name else list(SETS)
    for name in names:
        min_size, max_size, seed = SETS.get(name, (None, None, None))
        make(name, args.num, args.min_size or min_size, args.max_size or max_size,
             args.seed if args.seed is not None else seed, args.maps_dir)


if __name__ == '__main__':
    main()
