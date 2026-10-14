"""
Progressive training levels for the mowing task, see MowerEnv._set_level.

The levels run on the host: after each update, the episodes that ended are
used to mark maps as completed, and the next level starts once every map of
the current level has been completed at least once (and, at the last level,
at least one random map with a floor plan and one with obstacles).
The level only changes the map sampling probabilities and the goal coverage,
which are traced parameters, so no recompilation is needed.
"""
import numpy as np

from marl.maps import RANDOM_GROUP


MOWING_LEVELS = [
    dict(goal_coverage=0.90, groups=[0], p_random=0.0),
    dict(goal_coverage=0.90, groups=[0, 1], p_random=0.0),
    dict(goal_coverage=0.95, groups=[0, 1], p_random=0.0),
    dict(goal_coverage=0.95, groups=[0, 1, 2], p_random=0.0),
    dict(goal_coverage=0.97, groups=[0, 1, 2], p_random=0.0),
    dict(goal_coverage=0.99, groups=[0, 1, 2], p_random=0.0),
    dict(goal_coverage=0.99, groups=[0, 1, 2, 3], p_random=0.0),
    dict(goal_coverage=0.99, groups=[0, 1, 2, 3], p_random=0.5),
]


class Curriculum:
    """
    groups:      (K,) group of each map in the bank (RANDOM_GROUP for random maps)
    floor_plan:  (K,) whether a random map has a floor plan
    obstacles:   (K,) whether a random map has circular obstacles
    """

    def __init__(self, groups, floor_plan, obstacles, start_level=1, enabled=True, levels=MOWING_LEVELS):
        self.levels = levels
        self.enabled = enabled
        self.groups = np.asarray(groups)
        self.floor_plan = np.asarray(floor_plan, dtype=bool)
        self.obstacles = np.asarray(obstacles, dtype=bool)
        self.level = min(start_level, len(levels)) if enabled else len(levels)
        self._start_level()

    @property
    def spec(self):
        return self.levels[self.level - 1]

    @property
    def goal_coverage(self):
        return self.spec['goal_coverage']

    def set_random_maps(self, floor_plan, obstacles):
        """Called when the random maps of the bank are regenerated"""
        is_random = self.groups == RANDOM_GROUP
        self.floor_plan[is_random] = floor_plan
        self.obstacles[is_random] = obstacles

    def map_logits(self):
        spec = self.spec
        fixed = np.isin(self.groups, spec['groups'])
        is_random = self.groups == RANDOM_GROUP
        p = np.zeros(len(self.groups))
        p_random = spec['p_random'] if is_random.any() else 0.0
        if fixed.any():
            p[fixed] = (1 - p_random) / fixed.sum()
        if p_random > 0:
            p[is_random] = p_random / is_random.sum()
        assert p.sum() > 0, 'No maps available for the current level'
        with np.errstate(divide='ignore'):
            return np.log(p).astype(np.float32)

    def _start_level(self):
        fixed = np.isin(self.groups, self.spec['groups'])
        self.completed = ~fixed
        uses_random = self.spec['p_random'] > 0 and (self.groups == RANDOM_GROUP).any()
        self.completed_floor_plan = not (uses_random and self.floor_plan.any())
        self.completed_obstacles = not (uses_random and self.obstacles.any())

    def update(self, map_idx, success):
        """
        map_idx, success: maps and outcomes of the episodes that ended.
        Returns True if the level changed.
        """
        if not self.enabled:
            return False
        map_idx = np.asarray(map_idx)[np.asarray(success, dtype=bool)]
        is_random = self.groups[map_idx] == RANDOM_GROUP
        self.completed[map_idx[~is_random]] = True
        if self.floor_plan[map_idx[is_random]].any():
            self.completed_floor_plan = True
        if self.obstacles[map_idx[is_random]].any():
            self.completed_obstacles = True
        if self.level < len(self.levels) and self.completed.all() and \
                self.completed_floor_plan and self.completed_obstacles:
            self.level += 1
            self._start_level()
            return True
        return False
