"""A gymnasium environment for a synthetic oscillatory network of transcriptional
regulators called a repressilator. A repressilator is a three-gene regulatory network
where the dynamics of mRNA and proteins follow an oscillatory behavior
(see https://www-nature-com.tudelft.idm.oclc.org/articles/35002125).
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.utils import colorize

if __name__ == "__main__":
    from oscillator_disturber import OscillatorDisturber
else:
    from .oscillator_disturber import OscillatorDisturber

RANDOM_STEP = False  # Use random steps in __main__


class Oscillator(gym.Env, OscillatorDisturber):
    """Synthetic oscillatory network

    .. note::
        This gymnasium environment inherits from the
        :class:`~bayesian_learning_control.simzoo.simzoo.common.disturber.Disturber`
        in order to be able to use it with the Robustness Evaluation tool of the
        Bayesian Learning Control package (BLC). For more information see
        `the BLC documentation <https://rickstaa.github.io/bayesian-learning-control/control/robustness_eval.html>`_.

    Description:
        The goal of the agent in the oscillator environment is to act in such a way that
        one of the proteins of the synthetic oscillatory network follows a supplied
        reference signal.

    Observation:
        **Type**: Box(4)

        +-----+-----------------------------------------------+----------------------+--------------------+
        | Num | Observation                                   | Min                  | Max                |
        +=====+===============================================+======================+====================+
        | 0   | Lacl mRNA concentration                       | -100                 | 100                |
        +-----+-----------------------------------------------+----------------------+--------------------+
        | 1   | tetR mRNA concentration                       | -100                 | 100                |
        +-----+-----------------------------------------------+----------------------+--------------------+
        | 2   || lacI (repressor) protein concentration       | -100                 | 100                |
        |     || (Inhibits transcription tetR gene)           |                      |                    |
        +-----+-----------------------------------------------+----------------------+--------------------+
        | 3   || tetR (repressor) protein concentration       | -100                 | 100                |
        |     || (Inhibits transcription CI)                  |                      |                    |
        +-----+-----------------------------------------------+----------------------+--------------------+
        | 2   | The value of the reference for protein 1      | -100                 | 100                |
        +-----+-----------------------------------------------+----------------------+--------------------+
        | 3   || The error between the current value of       | -100                 | 100                |
        |     || protein 1 and the reference                  |                      |                    |
        +-----+-----------------------------------------------+----------------------+--------------------+

    Actions:
        **Type**: Box(3)

        +-----+---------------------------------------------------------------------------+
        | Num | Action                                                                    |
        +=====+===========================================================================+
        | 0   || Number of Lacl proteins produced during continuous growth under repressor|
        |     || saturation (Leakiness).                                                  |
        +-----+---------------------------------------------------------------------------+
        | 1   || Number of tetR proteins produced during continuous growth under repressor|
        |     || saturation (Leakiness).                                                  |
        +-----+---------------------------------------------------------------------------+
        | 2   || Number of CI proteins produced during continuous growth under repressor  |
        |     || saturation (Leakiness).                                                  |
        +-----+---------------------------------------------------------------------------+

    Cost:
        A cost, computed as the sum of the squared differences between the estimated and the actual states:

        .. math::

            C = \\abs{p_1 - r_1}

    Starting State:
        All observations are assigned a uniform random value in ``[-0.05..0.05]``

    Episode Termination:
        -   When the step cost is higher than 100.

    Solved Requirements:
        Considered solved when the average cost is lower than 300.

    Attributes:
        state (numpy.ndarray): The current system state.
        t (float): The current time step.
        dt (float): The environment step size.
        sigma (float): The variance of the system noise.
    """  # noqa: E501

    instances = []

    def __init__(
        self,
        render_mode=None,
        reference_type="periodic",
        clipped_action=True,
    ):
        """Constructs all the necessary attributes for the oscillator instance.

        Args:
            render_mode (str, optional): The render mode you want to use. Defaults to
                ``None`` as it is not used in this environment.
            reference_type (str, optional): The type of reference you want to use
                (``constant`` or ``periodic``), by default ``periodic``.
            clipped_action (str, optional): Whether the actions should be clipped if
                they are greater than the set action limit. Defaults to ``True``.
        """
        # Display warning if render_mode is not None.
        if render_mode is not None:
            print(
                colorize(
                    (
                        "WARNING: The `render_mode` argument is not used in the "
                        "Oscillator environment."
                    ),
                    "yellow",
                    bold=True,
                )
            )

        super().__init__()  # Setup disturber
        self.__class__.instances.append(self)
        self._instance_nr = len(self.__class__.instances)
        self._action_clip_warning = False
        self._clipped_action = clipped_action

        self.reference_type = reference_type
        self.t = 0.0
        self.dt = 1.0
        self.sigma = 0.0
        self._init_state = np.array(
            [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
        )  # Initial state when random is disabled

        # Print environment information
        print(
            colorize(
                (
                    f"INFO: Oscillator environment {self._instance_nr} is using a "
                    f"'{reference_type}' reference."
                ),
                "green",
                bold=True,
            )
        )

        # Set oscillator network parameters
        self.K = 1.0
        self.c1 = 1.6
        self.c2 = 0.16
        self.c3 = 0.16
        self.c4 = 0.06
        self.b1 = 1.0
        self.b2 = 1.0
        self.b3 = 1.0

        # Set angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([100, 100, 100, 100, 100, 100, 100, 100], dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.reward_range = spaces.Box(
            np.array([0.0], dtype=np.float32),
            np.array([100], dtype=np.float32),
            dtype=np.float32,
        )

        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

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
        if self._clipped_action:
            if (
                (action < self.action_space.low).any()
                or (action > self.action_space.high).any()
                and not self._action_clip_warning
            ):
                print(
                    colorize(
                        (
                            f"WARNING: Action '{action}' was clipped as it is not in "
                            f"the action_space 'high: {self.action_space.high}, "
                            f"low: {self.action_space.low}'."
                        ),
                        "yellow",
                        bold=True,
                    )
                )
                self._action_clip_warning = True
            u1, u2, u3 = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            u1, u2, u3 = action

        # Perform action in the environment and return the new state
        # NOTE: The new state is found by solving 3 first-order differential equations.
        m1, m2, m3, p1, p2, p3 = self.state
        m1_dot = self.c1 / (self.K + np.square(p3)) - self.c2 * m1 + self.b1 * u1
        p1_dot = self.c3 * m1 - self.c4 * p1
        m2_dot = self.c1 / (self.K + np.square(p1)) - self.c2 * m2 + self.b2 * u2
        p2_dot = self.c3 * m2 - self.c4 * p2
        m3_dot = self.c1 / (self.K + np.square(p2)) - self.c2 * m3 + self.b3 * u3
        p3_dot = self.c3 * m3 - self.c4 * p3

        # Calculate mRNA concentrations
        # Note: Use max to make sure concentrations can not be negative.
        m1 = np.max(
            [
                m1
                + m1_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        m2 = np.max(
            [
                m2
                + m2_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        m3 = np.max(
            [
                m3
                + m3_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )

        # Calculate protein concentrations
        # Note: Use max to make sure concentrations can not be negative.
        p1 = np.max(
            [
                p1
                + p1_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        p2 = np.max(
            [
                p2
                + p2_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        p3 = np.max(
            [
                p3
                + p3_dot * self.dt
                + self.np_random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )

        # Retrieve state
        self.state = np.array([m1, m2, m3, p1, p2, p3])
        self.t = self.t + self.dt  # Increment time step

        # Calculate cost
        r1 = self.reference(self.t)
        cost = np.square(p1 - r1)
        # cost = (abs(p1 - r1)) ** 0.2

        # Define stopping criteria
        terminated = bool(cost > self.reward_range.high or cost < self.reward_range.low)

        # Return state, cost, terminated, truncated and info_dict
        return (
            np.array([m1, m2, m3, p1, p2, p3, r1, p1 - r1]),
            cost,
            terminated,
            False,
            dict(reference=r1, state_of_interest=p1 - r1),
        )

    def reset(
        self,
        seed=None,
        random=True,
    ):
        """Reset gymnasium environment.

        Args:
            seed (int, optional): A random seed for the environment. By default
                ``None``.
            random (bool, optional): Whether we want to randomly initialise the
                environment. By default True.

        Returns:
            Tuple[numpy.ndarray, dict]: Tuple containing:
                numpy.ndarray: Array containing the current observations.
                dict: Dictionary containing additional information.
        """
        super().reset(seed=seed)

        # Return random initial state
        self.state = (
            self.np_random.uniform(low=0, high=1, size=(6,))
            if random
            else self._init_state
        )
        self.t = 0.0
        m1, m2, m3, p1, p2, p3 = self.state
        r1 = self.reference(self.t)
        return np.array([m1, m2, m3, p1, p2, p3, r1, p1 - r1]), dict(
            reference=r1, state_of_interest=p1 - r1
        )

    def reference(self, t):
        """Returns the current value of the periodic reference signal that is tracked by
        the Synthetic oscillatory network.

        Args:
            t (float): The current time step.

        Returns:
            float: The current reference value.
        """
        if self.reference_type == "periodic":
            return 8 + 7 * np.sin((2 * np.pi) * t / 200)
        else:
            return 8

    def render(self, mode="human"):
        """Render one frame of the environment.

        Args:
            mode (str, optional): Gym rendering mode. The default mode will do something
                human friendly, such as pop up a window.

        NotImplementedError: Will throw a NotImplimented error since the render method
            has not yet been implemented.

        Note:
            This currently is not yet implemented.
        """
        raise NotImplementedError(
            "No render method was implemented yet for the Oscillator environment."
        )


if __name__ == "__main__":
    print("Setting up oscillator environment.")
    env = gym.make("Oscillator")

    # Take T steps in the environment
    T = 60000
    path = []
    t1 = []
    s = env.reset()
    print(f"Taking {T} steps in the oscillator environment.")
    for i in range(int(T / env.dt)):
        action = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, terminated, truncated, info = env.step(action)
        path.append(s)
        t1.append(i * env.dt)
    print("Finished oscillator environment simulation.")

    # Plot results
    print("Plot results.")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.plot(t1, np.array(path)[:, 0], color="orange", label="mRNA1")
    # ax.plot(t1, np.array(path)[:, 1], color="magenta", label="mRNA2")
    # ax.plot(t1, np.array(path)[:, 2], color="sienna", label="mRNA3")
    ax.plot(t1, np.array(path)[:, 3], color="blue", label="protein1")
    # ax.plot(t1, np.array(path)[:, 4], color="cyan", label="protein2")
    # ax.plot(t1, np.array(path)[:, 5], color="green", label="protein3")
    # ax.plot(t1, np.array(path)[:, 0:3], color="blue", label="mRNA")
    # ax.plot(t1, np.array(path)[:, 3:6], color="blue", label="protein")
    ax.plot(t1, np.array(path)[:, 6], color="yellow", label="reference")
    ax.plot(t1, np.array(path)[:, 7], color="red", label="error")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.ioff()
    plt.show()

    print("done")
