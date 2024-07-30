---
title: 'Evoke: A Python package for evolutionary signalling games'
tags:
  - Python
  - evolutionary game theory
  - signalling games
  - sender-receiver framework
  - evolutionary simulations
authors:
  - name: Stephen Francis Mann
    orcid: 0000-0002-4136-8595
    equal-contrib: true
    affiliation: "1, 2"
  - name: Manolo Martínez
    corresponding: true
    orcid: 0000-0002-6194-7121
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: LOGOS Research Group, Universitat de Barcelona, Spain
    index: 1
  - name: Max Planck Institute for Evolutionary Anthropology, Leipzig, Germany
    index: 2
date: 22 July 2024
bibliography: paper.bib
---

# Summary

**Evoke** is a Python library for evolutionary simulations of signalling games.
It offers a simple and intuitive API that can be used to analyze arbitrary game-theoretic models, and to easily reproduce and customize well-known results and figures from the literature.

A signalling game is a special kind of mathematical game, a formal representation of interactions between agents
In a signalling game, the actions available to the players include sending and responding to signals.
The agents in games traditionally studied in game theory develop strategies via such dynamics as reinforcement learning.
In contrast, evolutionary game theory investigates how strategies change over time in populations undergoing evolutionary change such as natural selection.
Signalling games can be studied in the traditional reinforcement-learning paradigm or in the evolutionary paradigm.
Evoke offers methods for both kinds of game dynamic.
Users are able to create signalling games and simulate the evolution of agents' strategies over time, using a range of game types and evolutionary or learning dynamics.

Evoke also allows the user to recreate and customize figures from the signalling game literature.
Examples provided with Evoke include figures from @skyrms2010signals and @godfrey-smith2013communication.
More examples from well-known books and papers will be added to the built-in library, and users are also able to create both reconstructions of models from the literature and their own models and results.

# Statement of need

While there are Python packages devoted to game theory, such as Nashpy [@nashpyproject], and evolutionary game theory, such as EGTtools [@Fernandez2020], to our knowledge there has not yet been a Python package dedicated to the study of signalling games in the context of both evolution and reinforcement learning.
That is the gap Evoke is intended to fill.

In the evolutionary game theory literature, models and results are often developed with proprietary code.
Evaluating and re-running models can be difficult for readers, because custom-made software is often not developed with other users in mind.
Sometimes the model code is not available at all.

It would be preferable to have a common framework that different users can share.
When new results are presented in a research article, readers of that article could run the model and check the results for themselves.
Readers could also vary the parameters to obtain results that were not reported in the original article, lending an air of interactivity to published papers.

Built-in examples already shipped with Evoke include figures from @skyrms2010signals.
These examples allow the user to change some of the input parameters to Skyrms's figures to see how different parameter values yield different results.
In a small way, this makes the book "interactive": in addition to the static figures on the page, the user can play with the models in order to get a sense of the range of outcomes each model can generate.

# Acknowledgements

This work was supported by Juan de la Cierva grant FJC2020-044240-I and María de Maeztu grant CEX2021-001169-M funded by MICIU/AEI/10.13039/501100011033.

# References