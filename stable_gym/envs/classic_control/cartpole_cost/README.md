# CartPoleCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/eb3d4f34-1429-4597-a51f-16aea0e7def2" alt="CartPole Cost environment" width="400px">
</div>

<!--alex ignore joint-->

:::{attention}
You're currently on the `han2020` branch of the [stable-gym](https://github.com/rickstaa/stable-gym) repository. This branch includes parameter modifications necessary to replicate the results of [Han et al. 2020](https://arxiv.org/abs/2004.14288). For the most recent version of the package, please switch to the `main` branch.
:::

An unactuated joint attaches a pole to a cart, which moves along a frictionless track. This environment is a modified version of the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) found in the Gymnasium package, with several key alterations:

* The action space is **continuous**, contrasting with the original **discrete** setting.
* Offers an optional feature to confine actions within the defined action space, preventing the agent from exceeding set boundaries when activated.
* The **reward** function is replaced with a (positive definite) **cost** function (negated reward), in line with Lyapunov stability theory.
* Maximum cart force is increased from `10` to `20`.
* Episode length is reduced from `500` to `250`.
* A termination cost of `c=100` is introduced for early episode termination, to promote cost minimization.
* The terminal angle limit is expanded from the original `12` degrees to `20` degrees, enhancing recovery potential.
* The terminal position limit is extended from `2.4` meters to `10` meters, broadening the recovery range.
* Velocity limits are adjusted from ±∞ to ±50, accelerating training.
* Angular velocity termination threshold is lowered from ±∞ to ±50, likely for improved training efficiency.
* Random initial state range is modified from `[-0.05, 0.05]` to `[-5, 5]` for the cart position and `[-0.2, 0.2]` for all other states, allowing for expanded exploration.
* The info dictionary is expanded to include the reference state, state of interest, and reference error.

Additional modifications in our implementation:

* Unlike the original environment's fixed cost threshold of `100`, this version allows users to adjust the maximum cost threshold improving training adaptability.
* The gravity constant is adjusted back from `10` to the real-world value of `9.8`, aligning it closer with the original CartPole environment.
* The data types for action and observation spaces are set to `np.float64`, diverging from the `np.float32` used by Han et al. 2020. This aligns the Gymnasium implementation with the original CartPole environment.

These modifications were first described in [Han et al. 2020](https://arxiv.org/abs/2004.14288) and further adapted in our version for enhanced training and exploration.

## Observation space

The environment returns the following observation:

* $x$ - Cart Position.
* $x_{dot}$ - Cart Velocity.
* $w$ - Pole angle.
* $w_{dot}$ - Pole angle velocity.

## Action space

* **u1:** The x-force applied on the cart.

## Episode Termination

An episode is terminated when:

* Pole Angle is more than 20 degrees.
* Cart Position is more than 10 m (center of the cart reaches the edge of the
  display).
* Episode length is greater than 200.
* The cost is greater than a set threshold (100 by default). This threshold can be changed using the `max_cost` environment argument.

## Environment goals

The goal is similar to the original `CartPole-v1` environment. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's control force. This must be done while the cart does not violate set position constraints. These constraints are defined in the cost function.

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between the current cart position and angle and the zero position and angle:

$$
cost = (x / x_{threshold})^2 + 20 * (\theta / \theta_{threshold})^2
$$

The cost is between `0` and a set threshold value in both tasks, and the maximum cost is used when the episode is terminated.

## Environment step return

In addition to the observations, the cost and a termination and truncation boolean, the environment also returns an info dictionary:

```python
[observation, cost, termination, truncation, info_dict]
```

The info dictionary contains the following keys:

* **reference**: The set cart position and angle reference (i.e. the zero position and angle).
* **state\_of\_interest**: The state that should track the reference (SOI).
* **reference\_error**: The error between SOI and the reference.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:CartPoleCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
