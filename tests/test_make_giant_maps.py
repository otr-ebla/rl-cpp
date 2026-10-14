import random

import cv2
import numpy as np

import make_giant_maps
from rlm.mower_env import MowerEnv


def test_fill_unreachable():
    """
    Test that free space the agent cannot reach is filled with obstacles.
    """
    img = np.full((100, 100), make_giant_maps.FREE, dtype=np.uint8)
    img[:, 50] = make_giant_maps.OBSTACLE # wall splitting the map in two
    img[:, :20] = make_giant_maps.FREE
    img[40:60, 30:70] = make_giant_maps.OBSTACLE
    img[45:55, 45:55] = make_giant_maps.FREE # enclosed room
    img[0:2, 50] = make_giant_maps.FREE # opening too narrow for the agent
    filled = make_giant_maps.fill_unreachable(img, radius_p=3)
    assert (filled[45:55, 45:55] == make_giant_maps.OBSTACLE).all()
    left = (filled[:, :50] == make_giant_maps.FREE).sum()
    right = (filled[:, 51:] == make_giant_maps.FREE).sum()
    assert min(left, right) == 0 and max(left, right) > 0


def test_generated_maps(tmp_path):
    """
    Test that the generated maps have the requested size and can be loaded.
    """
    random.seed(0)
    np.random.seed(0)
    ppm = 1 / 0.0375
    for task in ['exploration', 'mowing']:
        for generate in [make_giant_maps.tile_map, make_giant_maps.random_map]:
            img = generate(task, 640, ppm, 0.15)
            img = make_giant_maps.fill_unreachable(img, 4)
            assert img.shape == (640, 640)
            assert set(np.unique(img)) <= {0, 128, 255}
            assert (img == make_giant_maps.FREE).mean() > 0.2
            filename = str(tmp_path / 'map.png')
            cv2.imwrite(filename, np.fliplr(img).T)
            env = MowerEnv(eval=True, exploration=task == 'exploration')
            env.eval_maps = [filename]
            env.reset()
            assert env.size_p == 640
            loaded = np.full(img.shape, make_giant_maps.FREE, dtype=np.uint8)
            loaded[env.unknown_obstacle_map > 0] = make_giant_maps.UNKNOWN
            # all obstacles are set as unknown in the evaluation maps
            assert ((loaded == make_giant_maps.FREE) == (img == make_giant_maps.FREE))[1:-1, 1:-1].all()
