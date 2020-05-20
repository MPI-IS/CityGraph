import os
import pickle

EXTENSION = ".citygraph"


def _fix_path(path):
    # if path is None: set it to current directory
    # then check that path is an existing directory
    # (raises a FileNotFoundError if not)
    if path is None:
        path = os.getcwd()
    if not os.path.isdir(path):
        raise FileNotFoundError("CityGraph.city_io: " + path + " does not exist")
    return path


def _get_abs_path(city, path):
    # city : either a string (city name) or an instance of City
    # path : absolute path to a folder with read/write access
    if isinstance(city, str):
        city_name = city
    else:
        city_name = city.name

    return os.sep.join([path,
                        city_name]) + EXTENSION


def is_saved(city, path=None):
    """
    Returns True if a city of the same name as already been saved.

    :param city: city or city name
    :type city: :py:class:`City<city_graph.city.City>` or str
    :param str path: path of the folder where cities are saved. default: current directory
    :returns: True if a city of the same name has already been saved.
    :rtype: bool
    """
    # set path to current directory if None.
    # raise Exception if path does not exist
    path = _fix_path(path)
    # path to the file
    path = _get_abs_path(city, path)
    return os.path.isfile(path)


def save(city, path=None, overwrite=False):
    """
    Save the city in a file.

    :param obj or str city: city to save (:py:class:`City<city_graph.city.City>` or str)
    :param str path: path of the folder where cities are saved. default: currrent directory
    :param bool overwrite: if True, will overwrite any saved city of the same name.
        default: False
    :returns: the path of the file into which the city was saved
    :raises: :py:class:`FileNotFoundError`: if path does not exist
    :raises: :py:class:`FileExists`: if overwrite is False
        and a city of the same name has already been saved.
    """
    # set path to current directory if None.
    # raise Exception if path does not exist
    path = _fix_path(path)

    # path to the file
    path = _get_abs_path(city, path)

    # file already exist, and overwrite is false:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError("CityGraph.city_io: can not save in", path, "(aleady exists)")

    with open(path, "wb") as f:
        pickle.dump(city, f)

    return path


def load(city_name, path=None):
    """
    :param str city_name: name of the city to load
    :param str path: path of the folder where cities are saved. default: current directory
    :raises: :py:class:`FileNotFoundError`: if no city of this name has been saved
    :returns: An instance of :py:class:`City<city_graph.city.City>`
    """
    if not is_saved(city_name, path):
        raise FileNotFoundError("loading city: " + path + " does not exist")

    path = _get_abs_path(city_name, _fix_path(path))

    with open(path, "rb") as f:
        city = pickle.load(f)

    return city
