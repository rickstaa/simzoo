"""Modified version of the cart-pole environment in v0.28.1 of the `gymnasium library`_.
In this modified version:

-   The action space is continuous, wherein the original version it is discrete.
-   The reward is replaced with a cost. This cost is defined as the difference between a
    state variable and a reference value (error).
-   Some of the environment parameters were changed slightly.

You can find the changes by searching for the ``NOTE:`` keyword.

.. _`gymnasium library`: https://gymnasium.farama.org/environments/classic_control/cart_pole/>
"""  # noqa: E501
# IMPROVEMENT: The multi-instance logic can be replaced with the new vectorized envs see
# https://gymnasium.farama.org/api/vector.
import math

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled

if __name__ == "__main__":
    from simzoo.envs.classic_control.cartpole_cost.cartpole_cost_disturber import (
        CartPoleDisturber,
    )
else:
    from .cartpole_cost_disturber import CartPoleDisturber

RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


# TODO: Add solving criteria after training.
class CartPoleCost(gym.Env, CartPoleDisturber):
    """Custom cartPole gymnasium environment.

    .. Note::
        Can also be used in a vectorized manner. See the `gym.vector`_ documentation.

    .. _gym.vector: https://gymnasium.farama.org/api/vector/

    Description:
        This environment was based on the cart-pole environment described by Barto,
        Sutton, and Anderson in
        `Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem`_.
        A pole is attached by an un-actuated joint to a cart, which moves along a
        frictionless track. The pendulum is placed upright on the cart and the goal
        is to balance the pole by applying forces in the left and right direction
        on the cart.

    Source:
        This environment corresponds to the version that is included in the Farama
        Foundation gymnasium package. It is different from this version in the fact
        that:

        -   The action space is continuous, wherein the original version it is discrete.
        -   The reward is replaced with a cost. This cost is defined as the difference
            between a state variable and a reference value (error).
        -   Some of the environment parameters were changed slightly.

    Observation:
        **Type**: Box(4)

        +-----+-----------------------+-----------------------+---------------------+
        | Num | Observation           | Min                   | Max                 |
        +=====+=======================+=======================+=====================+
        | 0   | Cart Position         | -20                   | 20                  |
        +-----+-----------------------+-----------------------+---------------------+
        | 1   | Cart Velocity         | -50                   | 50                  |
        +-----+-----------------------+-----------------------+---------------------+
        | 2   | Pole Angle            | ~ -.698 rad (-40 deg) | ~ .698 rad (40 deg) |
        +-----+-----------------------+-----------------------+---------------------+
        | 3   | Pole Angular Velocity | -50rad                | 50rad               |
        +-----+-----------------------+-----------------------+--------------------+

        .. Note::
            While the ranges above denote the possible values for observation space of
            each element, it is not reflective of the allowed values of the state space
            in an un-terminated episode. Particularly:
                -   The cart x-position (index 0) can be take values between
                    ``(-20, 20)``, but the episode terminates if the cart leaves the
                    ``(-10, 10)`` range.
                -   The pole angle can be observed between  ``(-0.698, .698)`` radians
                    (or **±40°**), but the episode terminates if the pole angle is
                    not in the range `(-.349, .349)` (or **±20°**)

    Actions:
        **Type**: Box(1)

        +-----+----------------------+
        | Num | Action               |
        +=====+======================+
        | 0   | The controller Force |
        +-----+----------------------+

        .. Note::
            The velocity that is reduced or increased by the applied force is not fixed
            and it depends on the angle the pole is pointing. The center of gravity of
            the pole varies the amount of energy needed to move the cart underneath it.

    Cost:
        A cost, computed using the :meth:`CartPoleCost.cost` method, is given for each
        simulation step including the terminal step. This cost is defined as a error
        between a state variable and a reference value.

    Starting State:
        All observations are assigned a uniform random value in ``[-0.2..0.2]``

    Episode Termination:
        -   Pole Angle is more than 20 degrees.
        -   Cart Position is more than 5 m (center of the cart reaches the edge of the
            display).
        -   Episode length is greater than 200.

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    Arguments:

        ```python
        import simzoo
        env = simzoo.make("CartPoleCost-v1")
        ```

        On reset, the `options` parameter allows the user to change the bounds used to
        determine the new random state when ``random=True``.

    Attributes:
        state (numpy.ndarray): Array containing the current state.
        t (float): Current time step.
        dt (float): Seconds between state updates.
        target_pos (float): The target position.
        const_pos (float): The constraint position.
        kinematics_integrator (str): The kinematics integrator used to update the state.
            Options are ``euler`` and ``semi-implicit euler``.
        theta_threshold_radians (float): The angle at which the pole is considered to be
            at a terminal state.
        x_threshold (float): The position at which the cart is considered to be at a
            terminal state.
        max_v (float): The maximum velocity of the cart.
        max_w (float): The maximum angular velocity of the pole.
        cost_range (gym.spaces.Box): The range of the cost.

    .. _`Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem`: https://ieeexplore.ieee.org/document/6313077
    """  # noqa: E501

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }  # Not used during training but in other gymnasium utilities.

    def __init__(
        self,
        render_mode=None,
        # NOTE: Custom environment arguments.
        task_type="stabilization",
        reference_type="constant",
        clip_action=True,
    ):
        """Constructs all the necessary attributes for the CartPoleCost instance.

        Args:
            render_mode (str, optional): Gym rendering mode. By default ``None``.
            task_type (str, optional): The task you want the agent to perform (options
                are "reference_tracking" and "stabilization"). Defaults to
                "stabilization".
            reference_type (str, optional): The type of reference you want to use
                (``constant`` or ``periodic``), by default ``periodic``.
            clip_action (str, optional): Whether the actions should be clipped if
                they are greater than the set action limit. Defaults to ``True``.
        """
        super().__init__()  # NOTE: Initialize disturber superclass.

        # NOTE: Compared to the original I store the initial values for the reset
        # function and replace the `self.total_mass` and `self.polemass_length` with
        # properties.
        self.gravity = self._gravity_init = 9.8
        self.masscart = self._mass_cart_init = 1.0
        self.masspole = self._mass_pole_init = 0.1
        self.length = (
            self._length_init
        ) = 1.0  # NOTE: The 0.5 of the original is moved to the `com_length` property.
        self.force_mag = 20  # NOTE: Original uses 10.
        self.tau = 0.02
        self.kinematics_integrator = "euler"

        # Position and angle at which to fail the episode.
        self.theta_threshold_radians = (
            20 * 2 * math.pi / 360
        )  # NOTE: original uses 12 degrees.
        self.x_threshold = 10  # NOTE: original uses 2.4.
        self.max_v = 50  # NOTE: Original uses np.finfo(np.float32).max (i.e. inf).
        self.max_w = 50  # NOTE: Original uses np.finfo(np.float32).max (i.e. inf).

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                self.max_v,
                self.theta_threshold_radians * 2,
                self.max_w,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )  # NOTE: Original uses discrete version.
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Clip the reward.
        # NOTE: Original does not do this. Here this is done because we want to decrease
        # the cost.
        self.cost_range = spaces.Box(
            np.array([0.0], dtype=np.float32),
            np.array([100], dtype=np.float32),
            dtype=np.float32,
        )

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

        # NOTE: custom parameters that are not found in the original environment.
        self.t = 0
        self.task_type = task_type
        self.reference_type = reference_type
        self._action_clip_warning = False
        self._clip_action = clip_action
        self._init_state = np.array(
            [0.1, 0.2, 0.3, 0.1]
        )  # Used when random is disabled in reset.
        self._init_state_range = {
            "low": [-2, -0.2, -0.2, -0.2],
            "high": [2, 0.2, 0.2, 0.2],
        }  # Used when random is enabled in reset.
        # NOTE: Original uses the following values in the reset function.
        # self._init_state_range = {
        #     "low": [-0.2, -0.05, -0.05, -0.05],
        #     "high": [0.2, 0.05, 0.05, 0.05],
        # }
        self.const_pos = 4.0  # Reference constraint.
        self.target_pos = 0.0  # Reference target.

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

                - length(:obj:`float`): The pole length.
                - pole_mass (:obj:`float`): The pole mass.
                - pole_mass (:obj:`float`): The cart mass.
                - gravity (:obj:`float`): The gravity constant.
        """
        return self.length, self.masspole, self.masscart, self.gravity

    def reset_params(self):
        """Resets the most important system parameters."""
        self.length = self._length_init
        self.masspole = self._mass_pole_init
        self.masscart = self._mass_cart_init
        self.gravity = self._gravity_init

    def reference(self, t):
        """Returns the current value of the periodic reference signal that is tracked by
        the cart-pole system.

        Args:
            t (float): The current time step.

        Returns:
            float: The current reference value.
        """
        if self.reference_type == "periodic":
            return self.target_pos + 7 * np.sin((2 * np.pi) * t / 200)
        else:
            return self.target_pos

    def cost(self, x, theta):
        """Returns the cost for a given cart position (x) and a pole angle (theta).

            Args:
                x (float): The current cart position.
                theta (float): The current pole angle (rads).

        Returns:
            (tuple): tuple containing:

                - cost (float): The current cost.
                - r_1 (float): The current position reference.
                - r_2 (float): The cart_pole angle reference.
        """
        if self.task_type.lower() == "reference_tracking":
            # Calculate cost (reference tracking task)
            stab_cost = x**2 / 100 + 20 * (theta / self.theta_threshold_radians) ** 2
            ref = [self.reference(self.t), 0.0]
            ref_cost = abs(x - ref[0])
            # ref_cost = np.square(x - ref[0])
            cost = stab_cost + ref_cost
        else:
            # Calculate cost (stabilization task)
            cost = (
                x**2 / 100 + 20 * (theta / self.theta_threshold_radians) ** 2
            )  # Stabilization task
            ref = np.array([0.0, 0.0])

        return cost, ref

    def step(self, action):
        """Take step into the environment.

        Args:
            action (numpy.ndarray): The action we want to perform in the environment.

        Returns:
            (tuple): tuple containing:

                - obs (:obj:`numpy.ndarray`): The current state
                - cost (:obj:`numpy.float64`): The current cost.
                - terminated (:obj:`bool`): Whether the episode was done.
                - truncated (:obj:`bool`): Whether the episode was truncated. This value
                    is set by wrappers when for example a time limit is reached or the
                    agent goes out of bounds.
                - info_dict (:obj:`dict`): Dictionary with additional information.
        """
        # Clip action if needed
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

        # Increment time step
        # NOTE: This is not done in the original environment.
        self.t = self.t + self.tau

        # Calculate cost
        cost, ref = self.cost(x, theta)

        # Define stopping criteria
        terminated = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold_radians
            or cost > self.cost_range.high  # NOTE: Added compared to original.
            or cost < self.cost_range.low  # NOTE: Added compared to original.
        )

        # Handle termination.
        if terminated:
            cost = 100.0

            # Throw warning if already done
            if self.steps_beyond_terminated is None:
                # Pole just fell!
                self.steps_beyond_terminated = 0
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1

        # Render environment if requested
        if self.render_mode == "human":
            self.render()

        # Return state, cost, terminated, truncated and info_dict
        violation_of_constraint = bool(abs(x) > self.const_pos)
        violation_of_x_threshold = bool(x < -self.x_threshold or x > self.x_threshold)
        return (
            np.array(self.state, dtype=np.float32),
            cost,
            terminated,
            False,
            dict(
                cons_pos=self.const_pos,
                cons_theta=self.theta_threshold_radians,
                target=self.target_pos,
                violation_of_x_threshold=violation_of_x_threshold,
                violation_of_constraint=violation_of_constraint,
                reference=ref,
                state_of_interest=theta,
            ),
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
            Tuple[numpy.ndarray, dict]: Tuple containing:
                - numpy.ndarray: Array containing the current observations.
                - dict: Dictionary containing additional information.
        """
        super().reset(seed=seed)

        # Initialize custom bounds while ensuring that the bounds are valid.
        # NOTE: If you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low = np.array(
            options["low"]
            if options is not None and "low" in options
            else self._init_state_range["low"],
            dtype=np.float32,
        )
        high = np.array(
            options["high"]
            if options is not None and "high" in options
            else self._init_state_range["high"],
            dtype=np.float32,
        )
        assert (self.observation_space.contains(low)) and (
            self.observation_space.contains(high)
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

        # Create info dict.
        x, _, theta, _ = self.state
        _, ref = self.cost(x, theta)
        violation_of_constraint = bool(abs(x) > self.const_pos)
        violation_of_x_threshold = bool(x < -self.x_threshold or x > self.x_threshold)
        info_dict = dict(
            cons_pos=self.const_pos,
            cons_theta=self.theta_threshold_radians,
            target=self.target_pos,
            violation_of_x_threshold=violation_of_x_threshold,
            violation_of_constraint=violation_of_constraint,
            reference=ref,
            state_of_interest=theta,
        )

        # Render environment reset if requested.
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), info_dict

    def render(self):
        """Render one frame of the environment."""
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
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
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
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

    # Aliases
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


if __name__ == "__main__":
    print("Setting up CartPoleCost environment.")
    env = gym.make("CartPoleCost", render_mode="human")

    # Take T steps in the environment
    T = 1000
    path = []
    t1 = []
    s = env.reset(
        options={
            "low": [-2, -0.2, -0.2, -0.2],
            "high": [2, 0.2, 0.2, 0.2],
        }
    )
    print(f"Taking {T} steps in the CartPoleCost environment.")
    for i in range(int(T / env.dt)):
        action = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()
        path.append(s)
        t1.append(i * env.dt)
    print("Finished CartPoleCost environment simulation.")

    # Plot results
    print("Plot results.")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(t1, np.array(path)[:, 0], color="orange", label="x")
    ax.plot(t1, np.array(path)[:, 1], color="magenta", label="x_dot")
    ax.plot(t1, np.array(path)[:, 2], color="sienna", label="theta")
    ax.plot(t1, np.array(path)[:, 3], color="blue", label=" theat_dot1")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.ioff()
    plt.show()

    print("done")
