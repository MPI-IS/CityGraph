from pathlib import Path

from setuptools import setup, find_packages


def _get_version():
    """"Convenient function to get the version of this package."""

    version_path = Path() / 'city_graph' / 'version.py'
    if not version_path.exists:
        return None
    with open(version_path) as version_file:
        ns = {}
        exec(version_file.read(), ns)

    return ns['__version__']


setup(name="city_graph",
      version=_get_version(),
      packages=find_packages(),
      description="Framework for representing a city and moving in it",
      url="https://github.com/MPI-IS/CityGraph.git",
      author="Jean Claude Passy, Ivan Oreshnikov, Vincent Berenz",
      install_requires=["networkx"],
      )
