Usage
=====


Installation
------------

To use evoke, first install it using pip:

.. code-block:: console

   (.venv) $ pip install evoke


The simplest case: reproducing a figure from the literature
-----------------------------------------------------------

The easiest thing to do with evoke is recreate a figure from the signalling game literature.
The parameters required to create some of these figures are included in the ``examples`` folder.

For example, to create figure 5.2 from *Signals* (Skyrms 2010), you would run:

.. code-block:: python

   Skyrms2010_5_2()

This creates an **object** which runs an evolutionary simulation with parameters as close as possible to those described by Bryan Skyrms for the figure in the book.

In this case evoke creates a figure very close to the original.
In other cases there might be a range of random properties that can cause deviation from the figures in the literature.
It's often a good idea to create the same figure multiple times, to get a feel for the range of variation that can be produced by the reported parameters.


Creating simulations
--------------------

If you want to create custom simulations, you need to create two kinds of object:

- **game**: This describes properties of the game, including the probabilities of each observable state, the number of signals available, and the payoff matrices of sender and receiver.
- **evolve**: This describes properties of the evolutionary scenario, especially whether the 'agents' are populations evolving via selection or individuals learning via reinforcement.

Games and evolve objects can be mixed and matched.
This allows you to see differences between evolution and reinforcement learning, by taking the same game and plugging it into different evolve objects.

