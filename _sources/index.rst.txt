.. Stable Gym documentation master file, created by
   sphinx-quickstart on Tue Jun  6 12:20:55 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
Welcome to Stable Gym's documentation
=====================================

.. image:: /images/cart_pole_header.jpeg
   :alt: Stable Gym
   :width: 80%
   :align: center

The :stable-gym:`Stable Gym <>` package contains several :gymnasium:`gymnasium environments <>` with
positive definite cost functions, designed for compatibility with stable RL agents. It was initially
created for the stable RL algorithms in the :stable-learning-control:`Stable Learning Control <>` package
but can be used with any RL agent requiring a **positive definite cost function**. For more information
about stable RL agents see the :stable-learning-control:`Stable Learning Control documentation <>`.

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Usage

   get_started/install.rst
   get_started/usage.rst
   envs/envs.rst

.. toctree::
   :maxdepth: 2
   :caption: Development

   dev/contributing.rst
   dev/add_new_envs.rst
   dev/doc_dev.rst
   dev/license.rst

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   autoapi/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
