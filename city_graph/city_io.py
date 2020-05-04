import os
import pickle

EXTENSION = ".citygraph"


def _get_abs_path(folder, city_name):
    return os.path.abspath(
        os.path.join(folder, city_name)) + EXTENSION


def is_saved(city, folder):
    """
    Returns True if a city of the same name as already been saved.

    :param obj or str city: city (:py:class:`.city.City`) or city name (str)
    :param str folder: path of the folder where cities are saved. default: /tmp/
    :returns: True if a city of the same name has already been saved.
    """
    return os.path.isfile(_get_abs_path(city, folder))


def save(city, folder, overwrite=False):
    """
    Save the city in a file.

    :param obj or str city: city to save (:py:class:`.city.City` or str)
    :param str folder: path of the folder where cities are saved. default: /tmp/
    :param bool overwrite: if True, will overwrite any saved city of the same name.
        default: False
    :returns: the path of the file into which the city was saved
    :raises: :py:class:`FileNotFoundError`: if path does not exist
    :raises: :py:class:`FileExists`: if overwrite is False
        and a city of the same name has already been saved.
    """
    path = os.path.join(folder)

    if not os.path.isdir(path):
        raise FileNotFoundError("saving city: " + path + " does not exist")

    path = _get_abs_path(city.name, path)

    if not overwrite:
        if os.path.isfile(path):
            raise FileExistsError("saving city: " + path + " already exist")

    with open(path, "wb") as f:
        pickle.dump(city, f)

    return path


def load(city_name, folder):
    """
    :param str city_name: name of the city to load
    :param str folder: path of the folder where cities are saved. default: /tmp/
    :raises: :py:class:`FileNotFoundError`: if no city of this name has been saved
    :returns: An instance of :py:class:`.city.City`
    """
    if not is_saved(city_name, folder):
        raise FileNotFoundError("loading city: " + folder + " does not exist")

    path = _get_abs_path(city_name, folder)

    with open(path, "rb") as f:
        city = pickle.load(f)

    return city
