"""Module that register the Simzoo gym environments.
"""

import importlib

import gym
from gym.envs.registration import register

# Create import prefix as stand-alone package or name_space package (mlc)
if importlib.util.find_spec("simzoo") is not None:
    namespace_prefix = ""
else:
    namespace_prefix = "bayesian_learning_control.simzoo."

ENVS = {
    "name": ["Oscillator-v1", "Ex3EKF-v1", "CartPoleCost-v0"],
    "module": [
        "simzoo.envs.biological.oscillator.oscillator:Oscillator",
        "simzoo.envs.classic_control.ex3_ekf.ex3_ekf:Ex3EKF",
        "simzoo.envs.classic_control.cart_pole_cost.cart_pole_cost:CartPoleCost",
    ],
    "max_step": [800, 800, 800],
}

for idx, env in enumerate(ENVS["name"]):
    if (
        env not in gym.envs.registry.env_specs
    ):  # NOTE: Required because we use namespace packages
        register(
            id=env,
            entry_point=namespace_prefix + ENVS["module"][idx],
            max_episode_steps=ENVS["max_step"][idx],
        )
