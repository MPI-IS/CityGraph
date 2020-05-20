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


dependencies = (
    "descartes",
    "geopandas",
    "networkx",
    "numpy",
    "osmnx",
    "scipy",
    "shapely"
)

setup(name="city_graph",
      version=_get_version(),
      packages=find_packages(),
      description="Framework for representing a city and moving around it",
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      url="https://github.com/MPI-IS/CityGraph.git",
      maintainer='Software Workshop - Max Planck Institute for Intelligent Systems',
      maintainer_email="jean-claude.passy@tuebingen.mpg.de",
      author="Jean Claude Passy, Ivan Oreshnikov, Vincent Berenz",
      install_requires=dependencies,
      license='BSD',
      python_requires='>=3.5',
      scripts=["demos/city_graph_demo"],
      )
