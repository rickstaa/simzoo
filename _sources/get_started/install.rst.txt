============
Installation
============

Pre-requisites
--------------

* `Python >= 3.7 <https://www.python.org/>`_.
* `pip >= 19.0.3 <https://pypi.org/project/pip/>`_.
* `conda <https://docs.conda.io/en/latest/>`_ (optional but recommended).

Installing Python
-----------------

You are recommended to install Python through Anaconda. Anaconda is a library that includes Python and many valuable packages for
Python, as well as an environment manager called Conda, makes package management simple. To install Anaconda, please follow the 
installation instructions on the `official website <https://docs.continuum.io/free/anaconda/install//>`_. After downloading and 
installing Anaconda3 (at the time of writing, `Anaconda3-2023.3.1`_), create a Conda Python env for organizing packages used 
in the ``stable_gym`` package:

.. code-block:: bash

    conda create -n stable_gym python=3.7

To use Python from the environment you just created, activate the environment with:

.. code-block:: bash

    conda activate stable_gym

.. note::
    Alternatively, you can use Python's `venv <https://docs.python.org/3/library/venv.html>`_ package to create a virtual environment. 

.. _`Anaconda3-2023.3.1`: https://repo.anaconda.com/archive/

Installing stable_gym
---------------------

After you successfully setup your Python environment, you can use pip to install the ``stable_gym`` package and its dependencies in
this environment:

.. code-block:: bash

    pip install .
