"""The CartPoleCost gymnasium environment."""

# NOTE: You can find the changes by searching for the ``NOTE:`` keyword.
import math

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled

EPISODES = 10  # Number of env episodes to run when __main__ is called.
RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


# TODO: Update solving criteria after training.
class CartPoleCost(gym.Env):
    r"""Custom CartPole Gymnasium environment.

    .. note::
        This environment can be used in a vectorized manner. Refer to the
        :gymnasium:`gym.vector <api/vector>` documentation for details.

    .. attention::
        You're currently on the ``han2020`` branch of the :stable-gym:`stable-gym <>`
        repository. This branch includes parameter modifications necessary to replicate
        the results of `Han et al.`_ 2020. For the most recent version of the package,
        please switch to the `main` branch.

    Source:
        This environment is a modified version of the CartPole environment from the
        Farma Foundation's :gymnasium:`Gymnasium <>` package, first used by `Han et al.`_
        in 2020. Modifications made by Han et al. include:

            - The action space is **continuous**, contrasting with the original **discrete**
              setting.
            - Offers an optional feature to confine actions within the defined action space,
              preventing the agent from exceeding set boundaries when activated.
            - The **reward** function is replaced with a (positive definite) **cost**
              function (negated reward), in line with Lyapunov stability theory.
            - Maximum cart force is increased from ``10`` to ``20``.
            - Episode length is reduced from ``500`` to ``250``.
            - A termination cost of :math:`c=100` is introduced for early episode
              termination, to promote cost minimization.
            - The terminal angle limit is expanded from the original ``12`` degrees to
              ``20`` degrees, enhancing recovery potential.
            - The terminal position limit is extended from ``2.4`` meters to ``10``
              meters, broadening the recovery range.
            - Velocity limits are adjusted from :math:`\pm \infty` to :math:`\pm 50`,
              accelerating training.
            - Angular velocity termination threshold is lowered from :math:`\pm \infty`
              to :math:`\pm 50`, likely for improved training efficiency.
            - Random initial state range is modified from ``[-0.05, 0.05]`` to ``[-5, 5]``
              for the cart position and ``[-0.2, 0.2]`` for all other states, allowing
              for expanded exploration.
            - The info dictionary is expanded to include the reference state, state of
              interest, and reference error.

        Additional modifications in our implementation:

            - Unlike the original environment's fixed cost threshold of ``100``, this
              version allows users to adjust the maximum cost threshold via the
              :obj:`max_cost` input, improving training adaptability.

    Observation:
        **Type**: Box(4) or Box(6)

        +-----+------------------------------+-----------------------+---------------------+
        | Num | Observation                  | Min                   | Max                 |
        +=====+==============================+=======================+=====================+
        | 0   | Cart Position                | -20                   | 20                  |
        +-----+------------------------------+-----------------------+---------------------+
        | 1   | Cart Velocity                | -50                   | 50                  |
        +-----+------------------------------+-----------------------+---------------------+
        | 2   | Pole Angle                   | ~ -.698 rad (-40 deg) | ~ .698 rad (40 deg) |
        +-----+------------------------------+-----------------------+---------------------+
        | 3   | Pole Angular Velocity        | -50rad                | 50rad               |
        +-----+------------------------------+-----------------------+---------------------+

        .. note::
            While the ranges above denote the possible values for observation space of
            each element, it is not reflective of the allowed values of the state space
            in an un-terminated episode. Particularly:

                -   The cart x-position (index 0) can be take values between
                    ``(-20, 20)``, but the episode terminates if the cart leaves the
                    ``(-10, 10)`` range.
                -   The pole angle can be observed between  ``(-0.698, .698)`` radians
                    (or **±40°**), but the episode terminates if the pole angle is not
                    in  the range ``(-.349, .349)`` (or **±20°**)

    Actions:
        **Type**: Box(1)

        +-----+----------------------+-----------------------+---------------------+
        | Num | Action               | Min                   | Max                 |
        +=====+======================+=======================+=====================+
        | 0   | The controller Force | -20                   | 20                  |
        +-----+----------------------+-----------------------+---------------------+

        .. note::
            The velocity that is reduced or increased by the applied force is not fixed
            and it depends on the angle the pole is pointing. The center of gravity of
            the pole varies the amount of energy needed to move the cart underneath it.

    Cost:
        A cost, computed using the :meth:`CartPoleCost.cost` method, is given for each
        simulation step, including the terminal step. This cost is the error
        between the cart position and angle and the zero position and angle. The cost
        is set to the maximum cost when the episode is terminated.The cost is defined as:

        .. math::

            cost = (x / x_{threshold})^2 + 20 * (\theta / \theta_{threshold})^2

    Starting State:
        The position is assigned a random value in ``[-5,5]`` and the other states are
        assigned a uniform random value in ``[-0.2..0.2]``.

    Episode Termination:
        -   Pole Angle is more than 20 degrees.
        -   Cart Position is more than 10 m (center of the cart reaches the edge of the
            display).
        -   Episode length is greater than 200.
        -   The cost is greater than a threshold (100 by default). This threshold can
            be changed using the ``max_cost`` environment argument.

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    How to use:
        .. code-block:: python

            import stable_gym
            import gymnasium as gym
            env = gym.make("stable_gym:CartPoleCost-v1")

        On reset, the ``options`` parameter allows the user to change the bounds used to
        determine the new random state when ``random=True``.

    Attributes:
        state (numpy.ndarray): The current state.
        t (float): Current time step.
        tau (float): The time step size. Also available as ``self.dt``.
        target_pos (float): The target position.
        constraint_pos (float): The constraint position.
        kinematics_integrator (str): The kinematics integrator used to update the state.
            Options are ``euler`` and ``semi-implicit euler``.
        theta_threshold_radians (float): The angle at which the pole is considered to be
            at a terminal state.
        x_threshold (float): The position at which the cart is considered to be at a
            terminal state.
        max_v (float): The maximum velocity of the cart.
        max_w (float): The maximum angular velocity of the pole.
        max_cost (float): The maximum cost.

    .. _`Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem`: https://ieeexplore.ieee.org/document/6313077
    .. _`Han et al.`: https://arxiv.org/abs/2004.14288
    """  # noqa: E501

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }  # Not used during training but in other gymnasium utilities.

    def __init__(
        self,
        render_mode=None,
        # NOTE: Custom environment arguments.
        max_cost=100.0,
        clip_action=True,
        action_space_dtype=np.float32,  # NOTE: Set to np.float32 as Han et al. 2020. Main branch uses np.float64  # noqa: E501
        observation_space_dtype=np.float32,  # NOTE: Set to np.float32 as Han et al. 2020. Main branch uses np.float64  # noqa: E501
    ):
        """Initialise a new CartPoleCost environment instance.

        Args:
            render_mode (str, optional): Gym rendering mode. By default ``None``.
            max_cost (float, optional): The maximum cost allowed before the episode is
                terminated. Defaults to ``100.0``.
            clip_action (str, optional): Whether the actions should be clipped if
                they are greater than the set action limit. Defaults to ``True``.
            action_space_dtype (union[numpy.dtype, str], optional): The data type of the
                action space. Defaults to ``np.float32``.
            observation_space_dtype (union[numpy.dtype, str], optional): The data type
                of the observation space. Defaults to ``np.float32``.
        """
        super().__init__()
        self.render_mode = render_mode
        assert max_cost > 0, "The maximum cost must be greater than 0."
        self.max_cost = max_cost
        self._clip_action = clip_action
        self._action_space_dtype = action_space_dtype
        self._observation_space_dtype = observation_space_dtype
        self._action_dtype_conversion_warning = False

        # NOTE: Compared to the original I store the initial values for the reset
        # function and replace the `self.total_mass` and `self.polemass_length` with
        # properties.
        self.gravity = self._gravity_init = (
            10.0  # NOTE: Set to 10 as Han et al. 2020. Main branch uses 9.8.  # noqa: E501
        )
        self.masscart = self._mass_cart_init = 1.0
        self.masspole = self._mass_pole_init = 0.1
        self.length = self._length_init = (
            1.0  # NOTE: The 0.5 of the original is moved to the `com_length` property.
        )
        self.force_mag = 20  # NOTE: Original uses 10.
        self.tau = 0.02
        self.kinematics_integrator = "euler"

        # Position and angle at which to fail the episode.
        self.theta_threshold_radians = (
            20 * 2 * math.pi / 360
        )  # NOTE: Original uses 12 degrees.
        self.x_threshold = 10  # NOTE: original uses 2.4.
        self.max_v = 50  # NOTE: Original uses np.finfo(np.float32).max (i.e. inf).
        self.max_w = 50  # NOTE: Original uses np.finfo(np.float32).max (i.e. inf).

        # Create observation space bounds.
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                self.max_v,
                self.theta_threshold_radians * 2,
                self.max_w,
            ],
        )
        self.action_space = spaces.Box(
            low=-self.force_mag,
            high=self.force_mag,
            shape=(1,),
            dtype=self._action_space_dtype,
        )  # NOTE: Original uses discrete version.
        self.observation_space = spaces.Box(
            -high, high, dtype=self._observation_space_dtype
        )

        # Clip the reward.
        # NOTE: Original does not do this. Here this is done because we want to decrease
        # the cost.
        self.reward_range = (0.0, max_cost)

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

        # NOTE: custom parameters that are not found in the original environment.
        self.t = 0
        self._action_clip_warning = False
        self._init_state = np.array(
            [0.1, 0.2, 0.3, 0.1], dtype=self._observation_space_dtype
        )  # Used when random is disabled in reset.
        self._init_state_range = {
            "low": [-5, -0.2, -0.2, -0.2],
            "high": [5, 0.2, 0.2, 0.2],
        }  # Used when random is enabled in reset.
        # NOTE: Original uses the following values in the reset function.
        # self._init_state_range = {
        #     "low": np.repeat(-0.05, 4),
        #     "high": np.repeat(0.05, 4),
        # }

    def set_params(self, length, mass_of_cart, mass_of_pole, gravity):
        """Sets the most important system parameters.

        Args:
            length (float): The pole length.
            mass_of_cart (float): Cart mass.
            mass_of_pole (float): Pole mass.
            gravity (float): The gravity constant.
        """
        self.length = length
        self.masspole = mass_of_pole
        self.masscart = mass_of_cart
        self.gravity = gravity

    def get_params(self):
        """Retrieves the most important system parameters.

        Returns:
            (tuple): tuple containing:

                -   length(:obj:`float`): The pole length.
                -   pole_mass (:obj:`float`): The pole mass.
                -   pole_mass (:obj:`float`): The cart mass.
                -   gravity (:obj:`float`): The gravity constant.
        """
        return self.length, self.masspole, self.masscart, self.gravity

    def reset_params(self):
        """Resets the most important system parameters."""
        self.length = self._length_init
        self.masspole = self._mass_pole_init
        self.masscart = self._mass_cart_init
        self.gravity = self._gravity_init

    def cost(self, x, theta):
        """Returns the cost for a given cart position (x) and a pole angle (theta).

            Args:
                x (float): The current cart position.
                theta (float): The current pole angle (rads).

        Returns:
            (tuple): tuple containing:

                -   cost (float): The current cost.
        """
        cost = np.square(x / self.x_threshold) + 20 * np.square(
            theta / self.theta_threshold_radians
        )

        return cost

    def step(self, action):
        """Take step into the environment.

        Args:
            action (numpy.ndarray): The action we want to perform in the environment.

        Returns:
            (tuple): tuple containing:

                -   obs (:obj:`np.ndarray`): Environment observation.
                -   cost (:obj:`float`): Cost of the action.
                -   terminated (:obj:`bool`): Whether the episode is terminated.
                -   truncated (:obj:`bool`): Whether the episode was truncated. This
                    value is set by wrappers when for example a time limit is reached or
                    the agent goes out of bounds.
                -   info (:obj:`dict`): Additional information about the environment.
        """
        # Convert action to correct data type if needed.
        if action.dtype != self._action_space_dtype:
            if not self._action_dtype_conversion_warning:
                logger.warn(
                    "The data type of the action that is supplied to the "
                    f"'ros_gazebo_gym:{self.spec.id}' environment ({action.dtype}) "
                    "does not match the data type of the action space "
                    f"({self._action_space_dtype.__name__}). The action data type will "
                    "be converted to the action space data type."
                )
                self._action_dtype_conversion_warning = True
            action = action.astype(self._action_space_dtype)

        # Clip action if needed.
        # NOTE: This is not done in the original environment.
        if self._clip_action:
            # Throw warning if clipped and not already thrown.
            if not self.action_space.contains(action) and not self._action_clip_warning:
                logger.warn(
                    f"Action '{action}' was clipped as it is not in the action_space "
                    f"'high: {self.action_space.high}, low: {self.action_space.low}'."
                )
                self._action_clip_warning = True

            force = np.clip(
                action, self.action_space.low, self.action_space.high
            ).item()
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid"
            force = action.item()
        assert self.state is not None, "Call reset before using step method."

        # Get the new state by solving 3 first-order differential equations.
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self._com_length
            * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)
        self.t = self.t + self.tau  # NOTE: Not done in the original environment.

        # Calculate cost.
        # NOTE: Different cost function compared to the original.
        cost = self.cost(x, theta)

        # Define stopping criteria.
        terminated = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold_radians
            or cost < self.reward_range[0]  # NOTE: Added compared to original.
            or cost > self.reward_range[1]  # NOTE: Added compared to original.
        )

        # Handle termination.
        if terminated:
            # Ensure cost is at max cost.
            cost = self.max_cost  # NOTE: Different cost compared to the original.

            # Throw warning if already done.
            if self.steps_beyond_terminated is None:
                # Pole just fell!
                self.steps_beyond_terminated = 0
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behaviour."
                    )
                self.steps_beyond_terminated += 1

        # Render environment if requested.
        if self.render_mode == "human":
            self.render()

        # Create observation and info dict.
        obs = np.array(self.state, dtype=self._observation_space_dtype)
        info_dict = dict(
            reference=np.array([0.0, 0.0], dtype=self._observation_space_dtype),
            state_of_interest=np.array([x, theta], dtype=self._observation_space_dtype),
            reference_error=np.array([-x, -theta], dtype=self._observation_space_dtype),
        )

        # NOTE: The original returns an empty info dict.
        return (
            obs,
            cost,
            terminated,
            False,
            info_dict,
        )

    def reset(self, seed=None, options=None, random=True):
        """Reset gymnasium environment.

        Args:
            seed (int, optional): A random seed for the environment. By default
                ``None``.
            options (dict, optional): A dictionary containing additional options for
                resetting the environment. By default ``None``. Not used in this
                environment.
            random (bool, optional): Whether we want to randomly initialise the
                environment. By default True.

        Returns:
            (tuple): tuple containing:

                -   obs (:obj:`numpy.ndarray`): Initial environment observation.
                -   info (:obj:`dict`): Dictionary containing additional information.
        """
        super().reset(seed=seed)

        # Initialise custom bounds while ensuring that the bounds are valid.
        # NOTE: If you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low = np.array(
            (
                options["low"]
                if options is not None and "low" in options
                else self._init_state_range["low"]
            ),
            dtype=self._observation_space_dtype,
        )
        high = np.array(
            (
                options["high"]
                if options is not None and "high" in options
                else self._init_state_range["high"]
            ),
            dtype=self._observation_space_dtype,
        )
        assert (
            self.observation_space.contains(
                np.append(
                    low,
                    np.zeros(
                        self.observation_space.shape[0] - low.shape[0],
                        dtype=self._observation_space_dtype,
                    ),
                )
            )
        ) and (
            self.observation_space.contains(
                np.append(
                    high,
                    np.zeros(
                        self.observation_space.shape[0] - low.shape[0],
                        dtype=self._observation_space_dtype,
                    ),
                )
            )
        ), (
            "Reset bounds must be within the observation space bounds "
            f"({self.observation_space})."
        )

        # Set random initial state and reset several env variables.
        self.state = (
            self.np_random.uniform(low=low, high=high, size=(4,))
            if random
            else self._init_state
        )
        self.steps_beyond_terminated = None
        self.t = 0.0

        # Retrieve observation and info_dict.
        obs = np.array(self.state, dtype=self._observation_space_dtype)
        x, _, theta, _ = self.state
        info_dict = dict(
            reference=np.array([0.0, 0.0], dtype=self._observation_space_dtype),
            state_of_interest=np.array([x, theta], dtype=self._observation_space_dtype),
            reference_error=np.array([-x, -theta], dtype=self._observation_space_dtype),
        )

        # Render environment reset if requested.
        if self.render_mode == "human":
            self.render()

        # NOTE: The original returns an empty info dict.
        return obs, info_dict

    def render(self):
        """Render one frame of the environment."""
        if self.render_mode is None:
            assert self.spec is not None
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("stable_gym:{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = scale * 0.1  # NOTE: Original uses 10.0.
        polelen = scale * self.length  # NOTE: Original uses scale * (2 * self.length)
        cartwidth = scale * 0.5  # NOTE: Original uses 50.0
        cartheight = scale * 0.3  # NOTE: Original uses 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART.
        carty = 100  # TOP OF CART.
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """Close down the viewer"""
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    @property
    def total_mass(self):
        """Property that returns the full mass of the system."""
        return self.masspole + self.masscart

    @property
    def _com_length(self):
        """Property that returns the position of the center of mass."""
        return self.length * 0.5  # half the pole's length

    @property
    def polemass_length(self):
        """Property that returns the pole mass times the COM length."""
        return self.masspole * self._com_length

    # Aliases.
    # NOTE: Added because the original environment doesn't use the pythonic naming.
    @property
    def pole_mass_length(self):
        """Alias for :attr:`polemass_length`."""
        return self.polemass_length

    @property
    def mass_pole(self):
        """Alias for :attr:`masspole`."""
        return self.masspole

    @property
    def mass_cart(self):
        """Alias for :attr:`masscart`."""
        return self.masscart

    @property
    def dt(self):
        """Property that also makes the timestep available under the :attr:`dt`
        attribute.
        """
        return self.tau

    @property
    def physics_time(self):
        """Returns the physics time. Alias for :attr:`.t`."""
        return self.t


if __name__ == "__main__":
    print("Setting up 'CartPoleCost' environment.")
    env = gym.make("stable_gym:CartPoleCost", render_mode="human")

    # Run episodes.
    episode = 0
    path, paths = [], []
    s, info = env.reset()
    path.append(s)
    print(f"\nPerforming '{EPISODES}' in the 'CartPoleCost' environment...\n")
    print(f"Episode: {episode}")
    while episode + 1 <= EPISODES:
        action = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, terminated, truncated, info = env.step(action)
        path.append(s)
        if terminated or truncated:
            paths.append(path)
            episode += 1
            path = []
            s, info = env.reset()
            path.append(s)
            print(f"Episode: {episode}")
    print("\nFinished 'CartPoleCost' environment simulation.")

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
        ax.set_title(f"CartPoleCost episode '{i+1}'")
        ax.legend()
        print("Close plot to see next episode...")
        plt.show()

    print("\nDone")
    env.close()
