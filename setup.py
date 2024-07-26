#!/usr/bin/env python
from setuptools import setup

setup(
    name="Evoke",
    version="0.1",
    install_requires=["numpy", "scipy"],
    description="Signalling games in python",
    author="Stephen Mann, Manolo Martínez",
    author_email="mail@manolomartinez.net",
    url="https://github.com/signalling-games-org/evoke",
    packages=["evoke"],
    license="GPLv3",
)
