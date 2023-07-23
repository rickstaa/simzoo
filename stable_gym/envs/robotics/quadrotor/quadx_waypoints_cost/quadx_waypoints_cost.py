"""The QuadXWaypointsCost gymnasium environment."""
import copy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import utils
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv

from stable_gym import ENVS  # noqa: F401

EPISODES = 10  # Number of env episodes to run when __main__ is called.
RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


class QuadXWaypointsCost(QuadXWaypointsEnv, utils.EzPickle):
    r"""Custom QuadXWaypoints Bullet gymnasium environment.

    .. note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    Source:
        Modified version of the `QuadXWaypoints environment`_ found in the
        :PyFlyt:`PyFlyt package <>`. This environment was first described by `Tai et al. 2023`_.
        In this modified version:

        -   The reward has been changed to a cost. This was done by negating the reward always
            to be positive definite.
        -   A health penalty has been added. This penalty is applied when the quadrotor moves
            outside the flight dome or crashes. The penalty equals the maximum episode steps
            minus the steps taken or a user-defined penalty.
        -   The ``max_duration_seconds`` has been removed. Instead, the ``max_episode_steps``
            parameter of the :class:`gym.wrappers.TimeLimit` wrapper is used to limit
            the episode duration.

        The rest of the environment is the same as the original QuadXWaypoints environment.
        Please refer to the `original codebase <https://github.com/jjshoots/PyFlyt>`__,
        :PyFlyt:`the PyFlyt documentation <>` or the accompanying
        `article of Tai et al. 2023`_ for more information.

    .. _`QuadXWaypoints environment`: https://jjshoots.github.io/PyFlyt/documentation/gym_envs/quadx_envs/quadx_waypoints_env.html
    .. _`Tai et al. 2023`: https://arxiv.org/abs/2304.01305
    .. _`article of Tai et al. 2023`: https://arxiv.org/abs/2304.01305

    Modified cost:
        A cost, computed using the :meth:`QuadXWaypointsCost.cost` method, is given for each
        simulation step, including the terminal step. This cost is defined as the Euclidean
        error between the quadrotors' current position and the position of the current
        waypoint (i.e. :math:`p=x_{x,y,z}=[0,0,1]`). Additionally, a penalty is
        given for moving away from the waypoint, and a health penalty can also
        be included in the cost. This health penalty is added when the drone leaves the
        flight dome or crashes. It equals the ``max_episode_steps`` minus the
        number of steps taken in the episode or a fixed value. The cost is
        computed as:

        .. math::

            cost = 10 \times \| p_{drone} - p_{waypoint} \| - \min(3.0 \times (p_{old} - p_{drone}), 0.0) + p_{health}

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    How to use:
        .. code-block:: python

            import stable_gym
            import gymnasium as gym
            env = gym.make("QuadXWaypointsCost-v1")

    Attributes:
        state (numpy.ndarray): The current system state.
        agent_hz (int): The agent looprate.
        initial_physics_time (float): The simulation startup time. The physics time at
            the start of the episode after all the initialisation has been done.
    """  # noqa: E501

    def __init__(
        self,
        num_targets=4,
        use_yaw_targets=False,
        goal_reach_distance=0.2,
        goal_reach_angle=0.1,
        flight_dome_size=5.0,
        angle_representation="quaternion",
        agent_hz=30,
        render_mode=None,
        render_resolution=(480, 480),
        include_health_penalty=True,
        health_penalty_size=None,
        exclude_waypoint_targets_from_observation=False,
        only_observe_immediate_waypoint=True,
        exclude_waypoint_target_deltas_from_observation=True,
        only_observe_immediate_waypoint_target_delta=True,
        **kwargs,
    ):
        """Initialise a new QuadXWaypointsCost environment instance.

        Args:
            num_targets (int, optional): Number of waypoints in the environment. By
                default ``4``.
            use_yaw_targets (bool, optional): Whether to match yaw targets before a
                waypoint is considered reached. By default ``False``.
            goal_reach_distance (float, optional): Distance to the waypoints for it to
                be considered reached. By default ``0.2``.
            goal_reach_angle (float, optional): Angle in radians to the waypoints for
                it to be considered reached, only in effect if ``use_yaw_targets`` is
                used. By default ``0.1``.
            flight_dome_size (float, optional): Size of the allowable flying area. By
                default ``5.0``.
            angle_representation (str, optional): The angle representation to use.
                Can be ``"euler"`` or ``"quaternion"``. By default ``"quaternion"``.
            agent_hz (int, optional): Looprate of the agent to environment interaction.
                By default ``30``.
            render_mode (None | str, optional): The render mode. Can be ``"human"`` or
                ``None``. By default ``None``.
            render_resolution (tuple[int, int], optional): The render resolution. By
                default ``(480, 480)``.
            include_health_penalty (bool, optional): Whether to penalize the quadrotor
                if it becomes unhealthy (i.e. if it falls over). Defaults to ``True``.
            health_penalty_size (int, optional): The size of the unhealthy penalty.
                Defaults to ``None``. Meaning the penalty is equal to the max episode
                steps and the steps taken.
            exclude_waypoint_targets_from_observation (bool, optional): Whether to
                exclude the waypoint targets from the observation. Defaults to
                ``False``.
            only_observe_immediate_waypoint (bool, optional): Whether to only observe
                the immediate waypoint target. Defaults to ``True``.
            exclude_waypoint_target_deltas_from_observation (bool, optional): Whether
                to exclude the waypoint target deltas from the observation. Defaults to
                ``True``.
            only_observe_immediate_waypoint_target_delta (bool, optional): Whether to
                only observe the immediate waypoint target delta. Defaults to
                ``True``.
            **kwargs: Additional keyword arguments passed to the
                :class:`~PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env.QuadXWaypointsEnv`
        """
        assert "sparse_reward" not in kwargs, (
            "'sparse_reward' should not be passed to the 'QuadXWaypointsCost' "
            "environment as only 'dense' rewards are supported."
        )
        assert "max_duration_seconds" not in kwargs, (
            "'max_duration_seconds' should not be passed to the 'QuadXWaypointsCost' "
            "as we use gymnasium's 'max_episode_steps' parameter together with "
            "the 'TimeLimit' wrapper to limit the episode duration."
        )
        assert (
            not exclude_waypoint_targets_from_observation
            or not exclude_waypoint_target_deltas_from_observation
        ), (
            "Either 'exclude_waypoint_targets_from_observation' or "
            "'exclude_reference_error_from_observation' should be set to 'False' for "
            "the agent to be able to learn."
        )

        self.state = None
        self.initial_physics_time = None
        self._max_episode_steps_applied = False
        self._previous_num_targets_reached = 0
        self._episode_waypoint_targets = None
        self._current_immediate_waypoint_target = None
        self.agent_hz = agent_hz
        self._include_health_penalty = include_health_penalty
        self._health_penalty_size = health_penalty_size
        self._exclude_waypoint_targets_from_observation = (
            exclude_waypoint_targets_from_observation
        )
        self._only_observe_immediate_waypoint = only_observe_immediate_waypoint
        self._exclude_waypoint_target_deltas_from_observation = (
            exclude_waypoint_target_deltas_from_observation
        )
        self._only_observe_immediate_waypoint_target_delta = (
            only_observe_immediate_waypoint_target_delta
        )

        super().__init__(
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
            **kwargs,
        )

        # Create a flat observation space.
        # NOTE: The original observation space uses gym.spaces.Sequence for the
        # waypoints this space however is not supported by most RL algorithms.
        low = self.observation_space.spaces["attitude"].low
        high = self.observation_space.spaces["attitude"].high
        if not self._exclude_waypoint_targets_from_observation:
            waypoints_low = (
                self.observation_space.spaces["target_deltas"].feature_space.low[-1]
                if self._only_observe_immediate_waypoint
                else self.observation_space.spaces["target_deltas"].feature_space.low
            ) / 2
            waypoints_high = (
                self.observation_space.spaces["target_deltas"].feature_space.high[-1]
                if self._only_observe_immediate_waypoint
                else self.observation_space.spaces["target_deltas"].feature_space.high
            ) / 2
            low = np.append(
                low,
                waypoints_low,
            )
            high = np.append(
                high,
                waypoints_high,
            )
        if not self._exclude_waypoint_target_deltas_from_observation:
            reference_error_low = (
                self.observation_space.spaces["target_deltas"].feature_space.low[-1]
                if self._only_observe_immediate_waypoint_target_delta
                else self.observation_space.spaces["target_deltas"].feature_space.low
            )
            reference_error_high = (
                self.observation_space.spaces["target_deltas"].feature_space.high[-1]
                if self._only_observe_immediate_waypoint_target_delta
                else self.observation_space.spaces["target_deltas"].feature_space.high
            )
            low = np.append(
                low,
                reference_error_low,
            )
            high = np.append(
                high,
                reference_error_high,
            )
        self.observation_space = gym.spaces.Box(
            low,
            high,
            dtype=self.observation_space.spaces["attitude"].dtype,
            seed=self.observation_space.spaces["attitude"].np_random,
        )

        # NOTE: Done to ensure the args of the QuadXWaypointsCost class are also
        # pickled.
        # NOTE: Ensure that all args are passed to the EzPickle class!
        utils.EzPickle.__init__(
            self,
            num_targets,
            use_yaw_targets,
            goal_reach_distance,
            goal_reach_angle,
            flight_dome_size,
            angle_representation,
            agent_hz,
            render_mode,
            render_resolution,
            include_health_penalty,
            health_penalty_size,
            exclude_waypoint_targets_from_observation,
            only_observe_immediate_waypoint,
            exclude_waypoint_target_deltas_from_observation,
            only_observe_immediate_waypoint_target_delta,
            **kwargs,
        )

    def cost(self, env_completed, num_targets_reached):
        """Compute the cost of the current state.

        Args:
            env_completed (bool): Whether the environment is completed.
            num_targets_reached (int): The number of targets reached.

        Returns:
            (float): The cost.
        """
        # Check if target was reached or if the environment was completed.
        if env_completed or num_targets_reached > self._previous_num_targets_reached:
            self._previous_num_targets_reached = num_targets_reached
            return 0.0

        # Calculate step cost.
        cost = -min(
            3.0 * self.waypoints.progress_to_target(), 0.0
        )  # Penalize moving away from target.
        cost += 0.1 / self.distance_to_immediate

        return cost

    def step(self, action):
        """Take step into the environment.

        .. note::
            This method overrides the
            :meth:`~PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env.QuadXWaypointsEnv.step`
            method such that the new cost function is used.

        Args:
            action (np.ndarray): Action to take in the environment.

        Returns:
            (tuple): tuple containing:

                -   obs (:obj:`np.ndarray`): Environment observation.
                -   cost (:obj:`float`): Cost of the action.
                -   terminated (:obj:`bool`): Whether the episode is terminated.
                -   truncated (:obj:`bool`): Whether the episode was truncated. This
                    value is set by wrappers when for example a time limit is reached
                    or the agent goes out of bounds.
                -   info (:obj:`dict`): Additional information about the environment.
        """
        obs, _, terminated, truncated, info = super().step(action)

        # Re-calculate target deltas.
        # NOTE: Done since the QuadXWaypointsEnv environment removes the immediate
        # waypoint from the waypoints.targets list when it is reached and doesn't
        # expose the old value.
        target_deltas = self.compute_target_deltas()

        # Flatten observation by combining attitude, waypoints and/or reference error.
        obs_flatten = copy.copy(obs["attitude"])
        if not self._exclude_waypoint_targets_from_observation:
            if self._only_observe_immediate_waypoint:
                obs_flatten = np.append(obs_flatten, self.immediate_waypoint_target)
            else:
                obs_flatten = np.append(obs_flatten, self._episode_waypoint_targets)
        if not self._exclude_waypoint_target_deltas_from_observation:
            if self._only_observe_immediate_waypoint_target_delta:
                obs_flatten = np.append(
                    obs_flatten, target_deltas[self._previous_num_targets_reached]
                )
            else:
                obs_flatten = np.append(obs_flatten, target_deltas)
        obs = obs_flatten

        # Calculate the cost.
        cost = self.cost(
            env_completed=info["env_complete"],
            num_targets_reached=info["num_targets_reached"],
        )

        # Add optional health penalty at the end of the episode if requested.
        if self._include_health_penalty:
            if terminated and info["collision"] or info["out_of_bounds"]:
                if self._health_penalty_size is not None:
                    cost += self._health_penalty_size
                else:  # If not set add unperformed steps to the cost.
                    cost += self.time_limit_max_episode_steps - self.step_count

        self._previous_num_targets_reached = info["num_targets_reached"]
        self._current_immediate_waypoint_target = copy.copy(self.waypoints.targets[0])

        self.state = obs

        return obs, cost, terminated, truncated, info

    def compute_target_deltas(self):
        """Compute the waypoints target deltas.

        .. note::
            Needed because the `~PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env.QuadXWaypointsEnv`
            removes the immediate waypoint from the waypoint targets list when it is
            reached and doesn't expose the old value.

        Returns:
            (np.ndarray): The waypoints target deltas.
        """  # noqa: E501
        _, ang_pos, _, lin_pos, quarternion = super().compute_attitude()

        new_waypoints_targets = copy.copy(self.waypoints.targets)
        self.waypoints.targets = self._episode_waypoint_targets
        target_deltas = self.waypoints.distance_to_target(ang_pos, lin_pos, quarternion)
        self.waypoints.targets = new_waypoints_targets

        return target_deltas

    def reset(self, seed=None, options=None):
        """Reset gymnasium environment.

        Args:
            seed (int, optional): A random seed for the environment. By default
                ``None``.
            options (dict, optional): A dictionary containing additional options for
                resetting the environment. By default ``None``. Not used in this
                environment.

        Returns:
            (tuple): tuple containing:

                -   obs (:obj:`numpy.ndarray`): Initial environment observation.
                -   info (:obj:`dict`): Dictionary containing additional information.
        """
        # Apply TimeLimit wrapper 'max_episode_steps' to the environment if exists.
        if not self._max_episode_steps_applied:
            self._max_episode_steps_applied = True
            time_limit_max_episode_steps = self.time_limit_max_episode_steps
            if time_limit_max_episode_steps is not None:
                self.max_steps = self.time_limit_max_episode_steps

        obs, info = super().reset(seed=seed, options=options)

        # Store environment information.
        self.initial_physics_time = self.env.elapsed_time
        self._previous_num_targets_reached = 0
        self._episode_waypoint_targets = copy.copy(self.waypoints.targets)
        self._current_immediate_waypoint_target = copy.copy(self.waypoints.targets[0])

        # Re-calculate target deltas.
        # NOTE: Done since the QuadXWaypointsEnv environment removes the immediate
        # waypoint from the waypoints.targets list when it is reached and doesn't
        # expose the old value.
        target_deltas = self.compute_target_deltas()

        # Flatten observation by combining attitude, waypoints and/or reference error.
        obs_flatten = copy.copy(obs["attitude"])
        if not self._exclude_waypoint_targets_from_observation:
            if self._only_observe_immediate_waypoint:
                obs_flatten = np.append(obs_flatten, self.immediate_waypoint_target)
            else:
                obs_flatten = np.append(obs_flatten, self._episode_waypoint_targets)
        if not self._exclude_waypoint_target_deltas_from_observation:
            if self._only_observe_immediate_waypoint_target_delta:
                obs_flatten = np.append(
                    obs_flatten, target_deltas[self._previous_num_targets_reached]
                )
            else:
                obs_flatten = np.append(obs_flatten, target_deltas)
        obs = obs_flatten

        self.state = obs

        return obs, info

    @property
    def immediate_waypoint_target(self):
        """The immediate waypoint target."""
        return self._current_immediate_waypoint_target

    @property
    def time_limit_max_episode_steps(self):
        """The maximum number of steps that the environment can take before it is
        truncated by the :class:`gymnasium.wrappers.TimeLimit` wrapper.
        """
        time_limit_max_episode_steps = (
            self._time_limit_max_episode_steps
            if hasattr(self, "_time_limit_max_episode_steps")
            and self._time_limit_max_episode_steps is not None
            else gym.registry[self.spec.id].max_episode_steps
        )
        return time_limit_max_episode_steps

    @property
    def time_limit(self):
        """The maximum duration of the episode in seconds."""
        return self.max_steps * self.agent_hz

    @property
    def dt(self):
        """The environment step size.

        Returns:
            (float): The simulation step size. Returns ``None`` if the environment is
                not yet initialized.
        """
        return 1 / self.agent_hz

    @property
    def tau(self):
        """Alias for the environment step size. Done for compatibility with the
        other gymnasium environments.

        Returns:
            (float): The simulation step size. Returns ``None`` if the environment is
                not yet initialized.
        """
        return self.dt

    @property
    def t(self):
        """Environment time."""
        return (
            self.env.elapsed_time - self.initial_physics_time
            if hasattr(self, "env")
            else 0.0
        )

    @property
    def physics_time(self):
        """Returns the physics time."""  # noqa: E501
        return self.env.elapsed_time if hasattr(self, "env") else 0.0


if __name__ == "__main__":
    print("Setting up 'QuadXWaypointsCost' environment.")
    env = gym.make("QuadXWaypointsCost", render_mode="human")

    # Run episodes.
    episode = 0
    path, paths = [], []
    s, _ = env.reset()
    path.append(s)
    print(f"\nPerforming '{EPISODES}' in the 'QuadXWaypointsCost' environment...\n")
    print(f"Episode: {episode}")
    while episode + 1 <= EPISODES:
        action = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, terminated, truncated, _ = env.step(action)
        path.append(s)
        if terminated or truncated:
            paths.append(path)
            episode += 1
            path, reference = [], []
            s, _ = env.reset()
            path.append(s)
            print(f"Episode: {episode}")
    print("\nFinished 'QuadXWaypointsCost' environment simulation.")

    # Plot results per episode.
    print("\nPlotting episode data...")
    for i in range(len(paths)):
        path = paths[i]
        fig, ax = plt.subplots()
        print(f"\nEpisode: {i+1}")
        path = np.array(path)
        t = np.linspace(0, path.shape[0] * env.dt, path.shape[0])
        for j in range(path.shape[1]):  # NOTE: Change if you want to plot less states.
            ax.plot(t, path[:, j], label=f"State {j+1}")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"QuadXWaypointsCost episode '{i+1}'")
        ax.legend()
        print("Close plot to see next episode...")
        plt.show()

    print("\nDone")
