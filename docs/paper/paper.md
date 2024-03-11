---
title: 'evoke: A Python package for evolutionary signalling games'
tags:
  - Python
  - evolutionary game theory
  - signalling games
  - sender-receiver framework
  - evolutionary simulations
authors:
  - name: Manolo Martínez
    corresponding: true
    orcid: 0000-0002-6194-7121
    equal-contrib: true
    affiliation: 1
  - name: Stephen Francis Mann
    orcid: 0000-0002-4136-8595
    equal-contrib: true
    affiliation: "1, 2"
affiliations:
  - name: LOGOS Research Group, Universitat de Barcelona, Spain
    index: 1
  - name: Max Planck Institute for Evolutionary Anthropology, Leipzig, Germany
    index: 2
date: 11 March 2024
bibliography: paper.bib
---

# Summary

**evoke** is a Python library for evolutionary simulations of signalling games.
It is particularly oriented towards reproducing results and figures from the literature, and offers a simple and intuitive API.

The easiest thing to do with evoke is recreate a figure from the signalling game literature.
Examples provided with evoke include figures from @skyrms2010signals and @godfrey-smith2013communication.
More examples from well-known books and papers will be added to the built-in library, and users are also able to create both reconstructions of models from the literature and their own models and results.

# Statement of need

In the evolutionary game theory literature, models and results are often developed with proprietary code.
Evaluating and re-running models can be difficult for readers, because custom-made software is often not developed with other users in mind.
Sometimes the model code is not available at all.

It would be preferable to have a common framework that different users can share.
When new results are presented in a research article, readers of that article could run the model and check the results for themselves.
Readers could also vary the parameters to obtain results that were not reported in the original article, lending an air of interactivity to published papers.

Built-in examples already shipped with evoke include figures from @skyrms2010signals.
These examples allow the user to change some of the input parameters to Skyrms's figures to see how different parameter values yield different results.
In a small way, this makes the book "interactive": in addition to the static figures on the page, the user can play with the models in order to get a sense of the range of outcomes each model can generate.

# Acknowledgements

This work was supported by Juan de la Cierva grant FJC2020-044240-I and María de Maeztu grant CEX2021-001169-M funded by MCIN/AEI/10.13039/501100011033.

# References