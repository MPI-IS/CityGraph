import os
import pickle

EXTENSION = ".citygraph"


def _get_abs_path(city, path):
    if isinstance(city, str):
        city_name = city
    else:
        city_name = city.name
    return os.sep.join([path,
                        city_name]) + EXTENSION


def is_saved(city, path="/tmp/"):
    """
    Returns True if a city of the same name as already been saved.

    :param obj or str city: city (:py:class:`.city.City`) or city name (str)
    :param str path: path of the folder where cities are saved. default: /tmp/
    :returns: True if a city of the same name has already been saved.
    """
    if isinstance(city, str):
        path = _get_abs_path(city, path)
    else:
        path = _get_abs_path(city.name, path)
    return os.path.isfile(path)


def save(city, path="/tmp/", overwrite=False):
    """
    Save the city in a file.

    :param obj or str city: city to save (:py:class:`.city.City` or str)
    :param str path: path of the folder where cities are saved. default: /tmp/
    :param bool overwrite: if True, will overwrite any saved city of the same name.
        default: False
    :returns: the path of the file into which the city was saved
    :raises: :py:class:`FileNotFoundError`: if path does not exist
    :raises: :py:class:`FileExists`: if overwrite is False
        and a city of the same name has already been saved.
    """
    path = str(path)

    if not os.path.isdir(path):
        raise FileNotFoundError("saving city: " + path + " does not exist")

    path = _get_abs_path(city, path)

    if not overwrite:
        if os.path.isfile(path):
            raise FileExistsError("saving city: " + path + " already exist")

    with open(path, "wb") as f:
        pickle.dump(city, f)

    return path


def load(city_name, path="/tmp/"):
    """
    :param str city_name: name of the city to load
    :param str path: path of the folder where cities are saved. default: /tmp/
    :raises: :py:class:`FileNotFoundError`: if no city of this name has been saved
    :returns: An instance of :py:class:`.city.City`
    """
    if not is_saved(city_name, path):
        raise FileNotFoundError("loading city: " + path + " does not exist")

    path = _get_abs_path(city_name, path)

    with open(path, "rb") as f:
        city = pickle.load(f)

    return city
