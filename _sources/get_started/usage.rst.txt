==========
How to use
==========

The environments in the :stable-gym:`Stable Gym <>` package can be imported like any other 
:gymnasium:`gymnasium environments <>`. You can then use the :func:`gym.vector.make` function to create an instance of
the environment. Here's a bare minimum example of using one of the :ref:`Stable Gym environment <envs>`. This will run an instance of the
:ref:`Oscillator-v1 <oscillator>` environment for 1000 timesteps. You should see the observations
being printed to the console. More examples can be found in the :stable-gym:`Stable Gym examples folder <tree/main/examples>`.

.. literalinclude:: ../../../examples/use_stable_gym.py
    :language: python
    :linenos:

.. important::

    Some of the environments in this package do not have a render method.
