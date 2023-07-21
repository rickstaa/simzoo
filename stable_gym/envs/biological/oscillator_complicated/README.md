# Oscillator Complicated gymnasium environment

A more challenging (i.e. complicated) version of the [Oscillator environment](https://rickstaa.dev/stable-gym/envs/biological/oscillator.html). This version adds an extra 4th protein and its accompanying mRNA transcription concentration to the environment. The light signal of an additional action input induces the mRNA transcription of this extra protein.

## Observation space

By default, the environment returns the following observation:

*   $m_1$ - The lacl mRNA transcripts concentration.
*   $m_2$ - The tetR mRNA transcripts concentration.
*   $m_3$ - The CI mRNA transcripts concentration.
*   $m_4$ - Extra protein mRNA transcripts concentration.
*   $p_1$ - The lacI (repressor) protein concentration (Inhibits transcription of tetR gene).
*   $p_2$ - The tetR (repressor) protein concentration (Inhibits transcription of CI gene).
*   $p_3$ - The CI (repressor) protein concentration (Inhibits transcription of extra protein gene).
*   $p_4$ - Extra protein concentration (Inhibits transcription of lacI gene).
*   $r$ - The reference we want to follow.
*   $r_{error}$ - The error between the state of interest (i.e. $p_1$) and the reference.

The last two variables can be excluded from the observation space by setting the `exclude_reference_from_observation` and `exclude_reference_error_from_observation` environment arguments to `True`. Please note that the environment needs the reference or the reference error to be included in the observation space to function correctly. If both are excluded, the environment will raise an error.

## Action space

*   $u_1$ - Relative intensity of the light signal that induces the Lacl mRNA gene expression.
*   $u_2$ - Relative intensity of the light signal that induces the tetR mRNA gene expression.
*   $u_3$ - Relative intensity of the light signal that induces the expression of the CI mRNA gene.
*   $u_4$ - Relative intensity of the light signal that induces the expression of the extra protein mRNA gene.

## Episode Termination

An episode is terminated when the maximum step limit is reached, or the step cost exceeds 100.

## Environment goal

The agent's goal in the oscillator environment is to act in such a way that one of the proteins of the synthetic oscillatory network follows a supplied reference signal.

## Cost function

The Oscillator environment uses the absolute difference between the reference and the state of interest as the cost function:

$$
cost = (p_1 - r_1)^2
$$

## Environment step return

In addition to the observations, the cost and a termination and truncation boolean, the environment also returns an info dictionary:

```python
[observation, info_dict]
```

The info dictionary contains the following keys:

*   **reference**: The set cart position reference.
*   **state\_of\_interest**: The state that should track the reference (SOI).
*   **reference\_error**: The error between SOI and the reference.
*   **reference\_constraint\_position**: A user-specified constraint they want to watch.
*   **reference\_constraint\_error**: The error between the SOI and the set reference constraint.
*   **reference\_constraint\_violated**: Whether the reference constraint was violated.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:OscillatorComplicated-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
