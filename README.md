![PyPI - Python Version](https://img.shields.io/pypi/pyversions/evoke-signals)

# Overview

+ [Tutorial](https://colab.research.google.com/drive/1AwUCP05lpITAP7_EZD7loGv3unhnwvhM#forceEdit=true&sandboxMode=true) | *Google Colab*
+ [Documentation](https://evoke.readthedocs.io/en/latest/) | *ReadTheDocs*
+ [Package](https://pypi.org/project/evoke-signals/) | *PyPI*
+ [Source code](https://github.com/signalling-games-org/evoke) | *GitHub*

evoke enables users to recreate signalling game simulations from the academic literature.

It comprises a library of methods from **evolutionary game theory** `evoke/src/` and an ever-growing collection of **user-friendly examples** `evoke/examples/`.

## Online tutorial

See the interactive tutorial on [Google Colab](https://colab.research.google.com/drive/1AwUCP05lpITAP7_EZD7loGv3unhnwvhM#forceEdit=true&sandboxMode=true).

## Requirements

evoke currently requires Python versions 3.9-3.11.
Further requirements are listed in `pyproject.toml`.

## Installation

Install with pip: `pip install evoke_signals`.

## Basic usage

### Creating your own signalling games

Creating and running your own simulations requires you to specify payoff matrices, player strategies, and learning or evolutionary dynamics.
See the [tutorial](https://colab.research.google.com/drive/1AwUCP05lpITAP7_EZD7loGv3unhnwvhM#forceEdit=true&sandboxMode=true) or the [documentation](https://evoke.readthedocs.io/en/latest/) for details and examples.

### Recreating figures from the literature

You can also recreate figures from the signalling game literature.
For example, suppose you want to recreate Figure 1.1 from page 11 of _Signals_ (Skyrms 2010).
The figure depicts the replicator dynamics of a population repeatedly playing a two-player cooperative game where each agent in the population either always plays sender or always plays receiver.
Senders observe one of two randomly chosen states of the world and produce one of two signals.
Receivers observe the signal and produce one of two acts.
If the act matches the state, both players gain a payoff.

The x-axis gives the proportion of receivers mapping the first signal to the second act and the second signal to the first act.
The y-axis gives the proportion of senders mapping the first state to the second signal and the second state to the first signal.

The points at which the population is achieving the greatest coordination are thus (0,0) and (1,1), so we would expect to see the arrows in the grid pointing towards those two corners.
Skyrms's figure shows exactly that.

For copyright reasons we can't show the original figure here.
Fortunately, recreating the figure is as easy as importing the relevant class and creating an instance of it:

```
from evoke.examples.skyrms2010signals import Skyrms2010_1_1
fig1_1 = Skyrms2010_1_1()
```

![Example of Skyrms 2010 Figure 1.1](https://github.com/signalling-games-org/evoke/blob/main/docs/tutorials/figures/skyrms2010_1_1.png?raw=true)

If you check page 11 of _Signals_ you will see this plot closely matches Skyrms's Figure 1.1.

### Recreating figures from the literature with different parameters

One of the useful features of evoke is that it allows you to re-run existing figures with different data.
In this way you can see how the results of a simulation would change if the parameters were tweaked.

Let's take Figure 3.3 of Skyrms (2010:40) as an example.
The figure depicts the mutual information between signal and state over successive trials of a two-player cooperative game in which agents learn via reinforcement.
There are two states, two signals and two acts.
Typically the mutual information will increase over time as the agents learn to use specific signals as indicators of specific states.

Once again we can create the figure just by creating an instance of the object.
Instead of the 100 iterations Skyrms uses, we will run the model for 1000 iterations:

```
from evoke.examples.skyrms2010signals import Skyrms2010_3_3
fig3_3 = Skyrms2010_3_3(iterations=1000)
````

![Example of Skyrms 2010 Figure 3.3](https://github.com/signalling-games-org/evoke/blob/main/docs/tutorials/figures/skyrms2010_3_3.png?raw=true)

Running for 100 iterations would sometimes lead to high information transmission and sometimes not, due to the stochastic nature of the simulation.
In contrast, running for 1000 iterations almost always leads to appreciable information transmission, as in the figure shown here.

Figures from the literature that are currently in the evoke library can be found in the `examples/` directory.

## Contributions

We welcome contributions, especially those that add to the stock of example figures.
See [CONTRIBUTIONS.md](https://github.com/signalling-games-org/evoke/blob/main/CONTRIBUTING.md) for information about contributing to evoke.

## References

Skyrms, B. (2010). *Signals: Evolution, Learning, and Information.* Oxford University Press. [https://doi.org/10.1093/acprof:oso/9780199580828.001.0001](https://doi.org/10.1093/acprof:oso/9780199580828.001.0001)
