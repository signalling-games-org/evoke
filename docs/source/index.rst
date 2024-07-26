.. Evoke documentation master file, created by
   sphinx-quickstart on Fri Aug 18 12:31:09 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Evoke: A Python package for evolutionary signalling games
=========================================================

**Evoke** is a Python library for evolutionary simulations of signalling games.
It is particularly oriented towards reproducing results and figures from the literature, and offers a simple and intuitive API.

* `Tutorial <https://colab.research.google.com/drive/1AwUCP05lpITAP7_EZD7loGv3unhnwvhM#forceEdit=true&sandboxMode=true>`_ | *Google Colab*
* `Documentation <https://evoke.readthedocs.io/en/latest/>`_ **You are here!** | *ReadTheDocs*
* `Package <https://pypi.org/project/evoke-signals/>`_ | *PyPI*
* `Source code <https://github.com/signalling-games-org/evoke>`_ | *GitHub*

.. note::

   This project is under active development.

Introduction
------------

A **game** is a formal representation of interactions between agents.
`Game theory <https://en.wikipedia.org/wiki/Game_theory>`_ is the study of agential interactions, which has the aim of understanding the success of different strategies and the dynamics of strategy change over time.
While the agents in traditional games might learn strategies via such dynamics as reinforcement learning, `evolutionary game theory <https://en.wikipedia.org/wiki/Evolutionary_game_theory>`_ studies how strategies change over time in populations undergoing evolutionary change such as natural selection.

A **signalling game** is a game in which the actions available to the players include sending and responding to signals.
Signalling games can be studied in the traditional reinforcement-learning paradigm or in the evolutionary paradigm.
Evoke offers methods for both kinds of game dynamic.

Creating a signalling game requires defining the following:

- The **agents** who will interact in the game
- The **strategies** of the agents, which determine what actions the agents will take under different circumstances
- The **payoffs** of the agents, which determine the benefit or penalty accruing to the agents as a consequence of their actions.

The following section shows how each of these components can be defined for a simple signalling game.

A simple signalling game
------------------------

Consider two agents, a sender and a receiver, who need to cooperate in order to achieve a common goal.
On each round of the game, one out of two possible states of the world obtains with 50/50 probability.
The sender observes which state occurs, and sends one of two possible signals to the receiver.
Upon receipt of the signal, the receiver performs one of two possible acts.
Payoffs to sender and receiver are identical and are determined by whether or not the receiver's act matches the state observed by the sender.
Payoffs can be represented in a table:

============= =======  =======
 *Payoffs*     Act 1    Act 2 
============= =======  =======
 **State 1**     1        0   
 **State 2**     0        1   
============= =======  =======

We have two agents, the basic actions available to them, and the payoffs associated with the outcomes that result from the combination of these actions.
What about strategies?
An agent's strategy describes which actions it performs in which circumstance.
In this simple game the sender has four deterministic or "pure" strategies, corresponding to the different ways it can produce signals in response to the states of the world it observes:

- State 1 => Signal 1; State 2 => Signal 2
- State 1 => Signal 2; State 2 => Signal 1
- State 1 => Signal 1; State 2 => Signal 1
- State 1 => Signal 2; State 2 => Signal 2

The latter two strategies are effectively useless to the receiver, since they entail sending the same signal no matter what the state of the world.
Nevertheless these useless strategies technically belong in the list of pure strategies.

Similarly, the receiver has four pure strategies available:

- Signal 1 => Act 1; Signal 2 => Act 2
- Signal 1 => Act 2; Signal 2 => Act 1
- Signal 1 => Act 1; Signal 2 => Act 1
- Signal 1 => Act 2; Signal 2 => Act 2

There are two combinations of sender and receiver strategies that will yield maximum payoffs: both play their first strategy (in which case State 1 => Signal 1 => Act 1 and State 2 => Signal 2 => Act 2), or both play their second (in which case State 1 => Signal 2 => Act 1 and State 2 => Signal 1 => Act 2).
Because signals in these games are assumed to be arbitrary, there is no reason to prefer one of these strategy combinations to another.
For the agents there is no sense in which Signal 1 more "naturally" goes together with State 1 or Act 1; Signal 2 would serve just as well in linking these two states.
What *is* important (from the perspective of agents wanting to maximise their payoffs) is that states get matched to their appropriate acts.
Any combination of signals that allows them to do this reliably is advantageous.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   usage
   examples
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
