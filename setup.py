from setuptools import setup,find_packages
import sys
from os import path

setup(name = "city_graph",
      packages=find_packages('.'),
      description="framework for representing a city and moving in it",
      url="https://github.com/MPI-IS/CityGraph.git",
      author="Jean Claude Passy, Ivan Oreshnikov, Vincent Berenz",
      install_requires = ["networkx"]
)

