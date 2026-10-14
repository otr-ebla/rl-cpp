from dataclasses import dataclass, asdict, fields
import math


@dataclass(frozen=True)
class EnvConfig:
    """
    Static environment configuration. Changing any of these values changes
    array shapes or traced constants, i.e. it triggers a recompilation.
    Defaults follow rlm.mower_env.MowerEnv for the mowing task.

    :param num_agents: Number of robots
    :param canvas_size: Size of the (square) canvas in pixels that holds a map,
        maps are placed in the top-left corner. Must be a multiple of
        scale_factor ** (num_maps - 1) and of canvas_size / global_size
    :param input_size: Size of the egocentric maps [pixels], same for all scales
    :param num_maps: Number of scales of the multi-scale maps
    :param scale_factor: Scale factor between subsequent scales
    :param meters_per_pixel: World map resolution
    :param step_size: Time per step [s]
    :param constant_lin_vel: Whether the agent only predicts steering
    :param max_lin_vel: Maximum linear velocity [m/s]
    :param max_ang_vel: Maximum angular velocity [rad/s]
    :param steering_limits_lin_vel: Whether to suppress linear velocity based on steering
    :param mower_radius: Radius of cutting disc (and robot) [m]
    :param robot_separation: Minimum distance between robot centers at spawn [m]
    :param lidar_rays: Number of lidar rays
    :param lidar_range: Range of the lidar sensor [m]
    :param lidar_fov: Lidar field of view [degrees]
    :param position_noise: Std of the position noise used for the observations [m]
    :param heading_noise: Std of the heading noise used for the observations [rad]
    :param lidar_noise: Std of the lidar measurement noise [m]
    :param all_unknown: Whether all obstacles are initially unknown
    :param overlap_observation: Use overlap (True) or binary coverage (False) in the observation
    :param frontier_observation: Whether to use a frontier map as observation
    :param robot_observation: Whether to use a map of the other robots as observation
    :param obstacle_dilation: Dilation kernel size around obstacles (metrics, frontier)
    :param global_size: Size of the global maps given to the centralized critic
    :param max_episode_steps: Maximum episode length
    :param max_non_new_steps: Maximum steps in a row in which no robot covered new ground
    :param goal_coverage_reward: Reward for reaching the target coverage
    :param wall_collision_reward: Reward for wall collision
    :param obstacle_collision_reward: Reward for obstacle collision
    :param robot_collision_reward: Reward for colliding with another robot
    :param newly_visited_reward_scale: Reward multiplier when covering new ground
    :param newly_visited_reward_max: Maximum reward from covering new ground
    :param team_reward_weight: Weight of the team part of the area reward,
        0: each robot is rewarded for the area it covered itself,
        1: each robot gets the area covered by the team divided by num_agents
    :param overlap_reward_scale: Reward multiplier for overlapping a previously covered area
    :param overlap_reward_max: Maximum overlap reward
    :param overlap_reward_always: Whether to always use overlap reward, or only when the area reward is 0
    :param local_tv_reward_scale: Reward multiplier for local (incremental) total variation
    :param local_tv_reward_max: Maximum reward from local (incremental) total variation
    :param turn_reward_scale: Reward scale for turning penalty
    :param constant_reward: Constant reward at each time step
    :param constant_reward_always: Whether to always apply constant reward or only when no new space is covered
    :param truncation_reward_scale: Reward multiplier for episode truncation
    """
    num_agents: int = 4
    canvas_size: int = 320
    input_size: int = 32
    num_maps: int = 4
    scale_factor: int = 4
    meters_per_pixel: float = 0.0375
    step_size: float = 0.5
    constant_lin_vel: bool = True
    max_lin_vel: float = 0.26
    max_ang_vel: float = 1.0
    steering_limits_lin_vel: bool = True
    mower_radius: float = 0.15
    robot_separation: float = 0.6
    lidar_rays: int = 24
    lidar_range: float = 3.5
    lidar_fov: float = 345
    position_noise: float = 0.01
    heading_noise: float = 0.05
    lidar_noise: float = 0.05
    all_unknown: bool = True
    overlap_observation: bool = True
    frontier_observation: bool = True
    robot_observation: bool = True
    obstacle_dilation: int = 9
    global_size: int = 64
    max_episode_steps: int = 2000
    max_non_new_steps: int = 1000
    goal_coverage_reward: float = 0
    wall_collision_reward: float = -10
    obstacle_collision_reward: float = -10
    robot_collision_reward: float = -10
    newly_visited_reward_scale: float = 1
    newly_visited_reward_max: float = 2
    team_reward_weight: float = 0.5
    overlap_reward_scale: float = 0
    overlap_reward_max: float = 5
    overlap_reward_always: bool = False
    local_tv_reward_scale: float = 1
    local_tv_reward_max: float = 5
    turn_reward_scale: float = 0
    constant_reward: float = -0.1
    constant_reward_always: bool = True
    truncation_reward_scale: float = 0

    def __post_init__(self):
        assert self.num_agents >= 1
        assert self.canvas_size % self.scale_factor ** (self.num_maps - 1) == 0, \
            'canvas_size must be a multiple of scale_factor ** (num_maps - 1)'
        assert self.canvas_size % self.global_size == 0, \
            'canvas_size must be a multiple of global_size'
        assert self.input_size % 2 == 0
        assert self.obstacle_dilation % 2 == 1
        assert 0 <= self.team_reward_weight <= 1

    # Derived quantities

    @property
    def pixels_per_meter(self):
        return 1 / self.meters_per_pixel

    @property
    def radius_p(self):
        return self.mower_radius * self.pixels_per_meter

    @property
    def lidar_samples(self):
        return int(self.lidar_range * self.pixels_per_meter)

    @property
    def window(self):
        """
        Side of the local window around a robot in which coverage and total
        variation are computed. It holds the region swept in the last two
        steps plus a margin.
        """
        step_p = self.max_lin_vel * self.step_size * self.pixels_per_meter
        return 2 * int(math.ceil(self.radius_p + 2 * step_p + 3))

    @property
    def action_dim(self):
        return 1 if self.constant_lin_vel else 2

    @property
    def map_channels(self):
        return 2 + int(self.frontier_observation) + int(self.robot_observation)

    @property
    def obs_channels(self):
        return self.num_maps * self.map_channels

    @property
    def global_channels(self):
        # coverage, true obstacles, known obstacles, all robots
        return 4

    @property
    def agent_features(self):
        # position (2), heading (2), team coverage, time
        return 6

    @property
    def max_newly_visited(self):
        """Maximum area covered by one robot per step [pixels]"""
        width_m = 2 * self.mower_radius
        length_m = self.max_lin_vel * self.step_size
        return width_m * length_m * self.pixels_per_meter ** 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(frozen=True)
class TrainConfig:
    """
    MAPPO hyperparameters.

    The number of agent samples per update is num_envs * num_steps * num_agents.
    """
    seed: int = 0
    num_envs: int = 256
    num_steps: int = 128
    total_steps: int = 200_000_000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 10.0
    update_epochs: int = 5
    num_minibatches: int = 4
    huber_delta: float = 10.0
    use_value_norm: bool = True
    hidden_dims: int = 256
    grouped_convs: bool = True
    init_log_std: float = -0.5
    num_random_maps: int = 256
    random_maps_every: int = 50
    start_level: int = 1
    curriculum: bool = True
    log_every: int = 1
    save_every: int = 50

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
