from argparse import BooleanOptionalAction
from dataclasses import dataclass, asdict, fields
import json
import math
import os


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
    :param robot_observation: Whether to use a map of the other robots as
        observation channel (legacy, superseded by neighbor_observation)
    :param neighbor_observation: Whether the other robots are observed as a
        set of relative poses (see MultiMowerEnv), encoded by a permutation
        invariant network, so that the same parameters work for any number of
        robots. The global maps of the critic then have no robot channel, the
        critic gets the poses of all robots as a set instead
    :param robot_lidar_obstacles: Whether each robot also sees the pixels at
        which its own lidar hits another robot in its known obstacle channel
        (and frontier), as rlm.mower_env.MowerEnv adds all lidar detections to
        the obstacle map. Only in the observation of the current step, the
        detections of robots are not kept. Without it the other robots are
        only seen by the lidar (used to evaluate the single-agent policy)
    :param comm_range: Communication range between robots [m]. 0: unlimited,
        all robots share one global map. > 0: each robot keeps its own belief
        of the coverage and known obstacle maps, built from its own coverage
        and lidar, and the beliefs of the robots that are connected (directly
        within comm_range, or through other robots) are merged with OR (max of
        the overlap counts) every step. The robots in the observation are
        then only the connected ones
    :param no_communication: Whether the robots do not communicate at all
        (independent single robots): each robot keeps its own belief (as with
        comm_range > 0) built only from its own coverage and lidar, it gets
        no poses of the other robots, and its area, overlap and total
        variation rewards are computed on its own coverage instead of the
        team's. The other robots are still obstacles (lidar, collisions, and
        robot_lidar_obstacles). The team coverage only ends the episode and
        gives the metrics
    :param obstacle_dilation: Dilation kernel size around obstacles (metrics, frontier)
    :param global_size: Size of the global maps given to the centralized critic
    :param global_crop: Whether the global maps of the critic cover only the
        map (exact area average of the map on global_size x global_size
        cells, any map size) instead of the whole canvas (average pooling,
        the map in the top-left corner). A small map then gets a finer grid
        (a 4.8 m map: 7.5 cm instead of 19 cm cells on the 320 pixel canvas)
        and no cells outside of it. The positions of the critic's inputs
        (agent, team) are then relative to the map size instead of the
        canvas size, and the map side is added to the episode progress (as
        meters / 10), since the maps of all sizes look the same. Needs
        neighbor_observation
    :param workload_size: Size of the grid (over the map) of the geodesic
        Voronoi partition of the remaining work (0: off): every cell is
        assigned to the robot with the shortest path to it around the true
        obstacles (the owner), and the workload of a robot is the free area
        not yet covered in its cells [m^2]. Computed every step from the true
        state (centralized training), for the rewards and the critic. The
        cells must be smaller than the doors (two free cells), and the
        shortest paths are followed for at most workload_size steps: the
        farther cells get the closest robot in a straight line
    :param critic_workload: Whether the centralized critic gets privileged
        features of each robot from the true state (needs workload_size and
        neighbor_observation), appended to its agent features: the robot's
        workload [thousands of steps of maximum coverage] and its share of
        the team's workload, and with limited communication the free area
        covered by the team that the robot does not know as covered
        (fraction of the free space) and the fraction of the other robots
        it is connected to. The maximum workload (the potential of the
        workload reward, same units) is appended to the episode progress
    :param max_episode_steps: Maximum episode length
    :param max_non_new_steps: Maximum steps in a row in which no robot covered new ground
    :param flip_when_stuck: Whether to turn a robot by 180 degrees when it collided
        max_stuck_steps times in a row, as in rlm.mower_env.MowerEnv (used to
        evaluate policies that cannot get out of deadlocks by themselves)
    :param max_stuck_steps: Collisions in a row before flipping the robot
    :param collision_ends_episode: Whether any collision (wall, obstacle or
        other robot) of any robot ends the episode at once, as a failure (not
        a success even if the goal coverage is reached in the same step)
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
    :param team_overlap_reward_scale: Penalty for the area a robot sweeps in a
        step that was already covered by any robot (the true team coverage,
        also with limited communication: centralized training), relative to
        the maximum area a robot covers in a step (1: a step entirely on
        covered ground costs team_overlap_reward_scale)
    :param workload_reward_scale: Team reward (the same for all robots) for
        balancing the workloads (needs workload_size): potential-based
        shaping gamma * phi(s') - phi(s) with phi = -maximum workload of the
        robots, in steps of the maximum area a robot covers in a step, and
        phi = 0 at the end of an episode, so that it does not change the
        optimal policy (Ng et al., 1999)
    :param shaping_gamma: Discount factor of the shaping, must be the gamma of the training
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
    robot_observation: bool = False
    neighbor_observation: bool = True
    robot_lidar_obstacles: bool = False
    comm_range: float = 0
    no_communication: bool = False
    obstacle_dilation: int = 9
    global_size: int = 64
    global_crop: bool = False
    workload_size: int = 0
    critic_workload: bool = False
    max_episode_steps: int = 2000
    max_non_new_steps: int = 1000
    flip_when_stuck: bool = False
    max_stuck_steps: int = 5
    collision_ends_episode: bool = False
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
    team_overlap_reward_scale: float = 0
    workload_reward_scale: float = 0
    shaping_gamma: float = 0.99
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
        assert self.comm_range >= 0
        assert not self.global_crop or self.neighbor_observation, 'global_crop needs neighbor_observation'
        assert self.workload_size >= 0
        assert self.workload_reward_scale == 0 or self.workload_size > 0, 'workload_reward_scale needs workload_size'
        assert not self.critic_workload or (self.workload_size > 0 and self.neighbor_observation), \
            'critic_workload needs workload_size and neighbor_observation'

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
    def limited_comm(self):
        return self.comm_range > 0 or self.no_communication

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
        # coverage, true obstacles, known obstacles, all robots (without neighbor_observation)
        return 3 + int(not self.neighbor_observation)

    @property
    def neighbor_features(self):
        # bearing (2), closeness at the extent of each scale, relative heading (2)
        return 4 + self.num_maps

    @property
    def progress_features(self):
        # goal coverage, steps without new coverage / max_non_new_steps, (global_crop) map side,
        # (critic_workload) maximum workload
        return 2 + int(self.global_crop) + int(self.critic_workload)

    @property
    def team_features(self):
        # neighbor features (true poses), absolute position / canvas size (2)
        return self.neighbor_features + 2

    @property
    def agent_features(self):
        # position (2), heading (2), team coverage, time, (critic_workload) privileged features
        return 6 + self.privileged_features

    @property
    def privileged_features(self):
        # workload, share of the team's workload, (limited communication) unknown coverage, connected robots
        return (2 + 2 * int(self.limited_comm)) if self.critic_workload else 0

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
    MAPPO / IPPO hyperparameters.

    algo selects the critic. The actor and the rest of the training are the
    same for both, and in both the robots share the global maps:
      'mappo': centralized critic V(s, i) on the agent-specific global state
      'ippo':  independent (local) critic V(o_i) on the robot's own
               observation only, the same inputs as the actor
    In both cases the parameters are shared between the robots.

    critic_state selects the input of the centralized critic of MAPPO:
      'global':         global state only: global maps, poses of all robots,
                        episode progress (marl.networks.GlobalCritic)
      'agent_specific': global state together with each robot's own local
                        observation, as in the MAPPO paper (marl.networks.Critic)

    The number of agent samples per update is num_envs * num_steps * num_agents,
    they are split into minibatches of minibatch_size agent samples (all
    robots of minibatch_size / num_agents environment steps), so the memory
    used by the gradient computation does not depend on num_envs.
    The actor updates of an iteration stop once the approximate KL divergence
    to the policy that collected the data exceeds 1.5 * target_kl (0: off).
    The first minibatch is never stopped: its KL is measured before any update
    and is only numerical noise (TF32 convolutions with other batch sizes).
    squash_mean and min_log_std: see marl.networks.Actor.
    adaptive_lr: the learning rate of the actor starts at actor_lr and is
    adapted after each iteration from its final KL, to keep it near target_kl
    (see marl.mappo.MAPPO._adapt_actor_lr), instead of annealing it (the
    critic still uses anneal_lr). Useful with a sharp (nearly deterministic)
    initial policy, e.g. from scripts/multi_agent/init_from_sb3.py, where a fixed learning rate
    either freezes the actor (KL early stopping after the first minibatch)
    or moves it too far.
    critic_warmup: number of first iterations that only train the critic,
    with the actor frozen, e.g. when the actor is initialized from a trained
    policy (scripts/multi_agent/init_from_sb3.py) and the critic is new: the advantages of an
    untrained critic would otherwise move the actor away from it.
    stop_update: stops the training after this update (0: off), with the
    learning rates still annealed over total_steps, e.g. to give another run
    the same budget as a run that was stopped before total_steps.
    neighbor_dims, neighbor_layers, neighbor_heads: width, number of
    transformer blocks and attention heads of marl.networks.PoseEncoder.
    critic_gather: the centralized critic also samples (bilinearly) the
    feature maps of its global maps encoder at each robot's position, so
    that each V(s, i) gets features of the global maps around robot i (the
    last two convolutions, receptive fields of 10 and 18 global cells:
    1.9 and 3.4 m on the 320 pixel canvas), instead of relating the robot's pose to the flattened global
    features by itself (see marl.networks.global_maps_encoding).
    """
    algo: str = 'mappo'
    critic_state: str = 'global'
    seed: int = 0
    num_envs: int = 256
    num_steps: int = 128
    total_steps: int = 200_000_000
    stop_update: int = 0
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 10.0
    update_epochs: int = 4
    minibatch_size: int = 8192
    target_kl: float = 0.02
    critic_warmup: int = 0
    adaptive_lr: bool = False
    min_actor_lr: float = 1e-6
    max_actor_lr: float = 1e-3
    huber_delta: float = 10.0
    use_value_norm: bool = True
    hidden_dims: int = 256
    neighbor_dims: int = 64
    neighbor_layers: int = 2
    neighbor_heads: int = 4
    critic_gather: bool = False
    grouped_convs: bool = True
    init_log_std: float = -0.5
    squash_mean: bool = True
    min_log_std: float = -2.0
    num_random_maps: int = 256
    random_maps_every: int = 50
    start_level: int = 1
    curriculum: bool = True
    log_every: int = 1
    save_every: int = 50

    def __post_init__(self):
        assert self.algo in ('mappo', 'ippo'), "algo must be 'mappo' or 'ippo'"
        assert self.critic_state in ('global', 'agent_specific'), "critic_state must be 'global' or 'agent_specific'"
        assert not self.adaptive_lr or self.target_kl > 0, 'adaptive_lr needs target_kl > 0'

    @property
    def centralized_critic(self):
        return self.algo == 'mappo'

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def add_dataclass_args(parser, cls, title):
    """Add every field of the dataclass cls as a command line argument (booleans as --flag / --no-flag)"""
    group = parser.add_argument_group(title)
    for f in fields(cls):
        if f.type in (bool, 'bool'):
            group.add_argument(f'--{f.name}', default=f.default, action=BooleanOptionalAction)
        else:
            typ = {'int': int, 'float': float, 'str': str}.get(f.type, f.type)
            group.add_argument(f'--{f.name}', default=f.default, type=typ)


def read_config(path):
    """Settings (env and train) of a run directory, checkpoint directory or config.json"""
    if os.path.isdir(path):
        path = os.path.join(path, 'config.json')
    with open(path) as f:
        config = json.load(f)
    names = {f.name for cls in (EnvConfig, TrainConfig) for f in fields(cls)}
    return {k: v for k, v in {**config['env'], **config['train']}.items() if k in names}
