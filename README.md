# Stable Gym

[![Stable Gym CI](https://github.com/rickstaa/stable-gym/actions/workflows/stable_gym.yml/badge.svg)](https://github.com/rickstaa/stable-gym/actions/workflows/stable_gym.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/rickstaa/stable-gym)](https://github.com/rickstaa/stable-gym/releases)
[![Python 3](https://img.shields.io/badge/Python->=3.8-brightgreen)](https://www.python.org/)
[![codecov](https://codecov.io/gh/rickstaa/stable-gym/branch/main/graph/badge.svg?token=RFM3OELQ3L)](https://codecov.io/gh/rickstaa/stable-gym)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![DOI](https://zenodo.org/badge/287501190.svg)](https://zenodo.org/badge/latestdoi/287501190)

A Python package that contains several [gymnasium environments](https://gymnasium.farama.org/)
with cost functions compatible with (stable) RL agents. It was initially created for the stable RL
algorithms in the [Stable Learning Control](https://github.com/rickstaa/stable-learning-control) package but can be
used with any RL agent requiring a **positive definite cost function**. For more information about stable
RL agents see the [Stable Learning Control documentation](https://rickstaa.dev/stable-learning-control).

## Installation and Usage

Please see the accompanying [documentation](https://rickstaa.dev/stable-gym) for information on installing and using this package.

## Contributing

<!--alex ignore husky-hooks-->

We use Husky pre-commit hooks and github actions to enforce high code quality. Before contributing to this repository, please check the [contribution guidelines](CONTRIBUTING.md).
