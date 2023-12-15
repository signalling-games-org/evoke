# Overview

+ [Tutorial](https://colab.research.google.com/drive/1AwUCP05lpITAP7_EZD7loGv3unhnwvhM#forceEdit=true&sandboxMode=true) | *Google Colab*
+ [Documentation](https://evoke.readthedocs.io/en/latest/) | *ReadTheDocs*
+ [Package](https://pypi.org/project/evoke-signals/) | *PyPI*
+ [Source code](https://github.com/signalling-games-org/evoke) | *GitHub*

evoke enables users to recreate signalling game simulations from the academic literature.

It comprises a library of methods from **evolutionary game theory** `evoke/src/` and an ever-growing collection of **user-friendly examples** `evoke/examples/`.

## Online tutorial

See the interactive tutorial on [Google Colab](https://colab.research.google.com/drive/1AwUCP05lpITAP7_EZD7loGv3unhnwvhM#forceEdit=true&sandboxMode=true).

## Installation

Install with pip: `pip install evoke_signals`.

## Basic usage

To create one of the example figures, simply import the relevant class and create an instance of it:

```
from evoke.examples.skyrms2010signals import Skyrms2010_1_1
fig1_1 = Skyrms2010_1_1()
```

![Example of Skyrms 2010 Figure 1.1](https://github.com/signalling-games-org/evoke/blob/main/docs/tutorials/figures/skyrms2010_1_1.png?raw=true)

Certain figures allow you to specify your own parameters:

```
from evoke.examples.skyrms2010signals import Skyrms2010_3_3
fig3_3 = Skyrms2010_3_3(iterations=1000)
````

![Example of Skyrms 2010 Figure 3.3](https://github.com/signalling-games-org/evoke/blob/main/docs/tutorials/figures/skyrms2010_3_3.png?raw=true)

See the `examples/` directory for a collection of available figures.
