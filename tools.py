"""
Utilities toolbox
"""

# --------------------------
# Angle Wrapper
# Simple toolbox for wrapping angles. Supports numeric types, lists, and other iterables. Works in both degrees and radians and is solely dependent on base Python.

import math


def wrap_to_pi(angle):
    """
    Wraps the given angle(s) to +/- pi.

    Parameters
    ----------
    :param angle: The values to wrap.
    :type angle: int, float, list[int], list[float], iterable[int], iterable[float]

    Returns
    --------
    :return: The wrapped values
    :rtype: int, float, list[int], list[float], iterable[int], iterable[float]

    Raises
    -------
    :raise TypeError: If the input data type is unsupported.
    """
    if isinstance(angle, (int, float)):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    elif isinstance(angle, list):
        return [wrap_to_pi(a) for a in angle]
    elif isinstance(angle, tuple):
        return tuple(wrap_to_pi(a) for a in angle)
    elif hasattr(
        angle, "__iter__"
    ):  # Check if it's an iterable (including numpy array)
        return [wrap_to_pi(a) for a in angle]
    else:
        raise TypeError("Unsupported data type for angle")


def wrap_to_2pi(angle):
    """
    Wraps the given angle(s) to 2 pi.

    Parameters
    ----------
    :param angle: The values to wrap.
    :type angle: int, float, list[int], list[float], iterable[int], iterable[float]

    Returns
    --------
    :return: The wrapped values
    :rtype: int, float, list[int], list[float], iterable[int], iterable[float]

    Raises
    -------
    :raise TypeError: If the input data type is unsupported.
    """
    if isinstance(angle, (int, float)):
        return angle % (2 * math.pi)
    elif isinstance(angle, list):
        return [wrap_to_2pi(a) for a in angle]
    elif isinstance(angle, tuple):
        return tuple(wrap_to_2pi(a) for a in angle)
    elif hasattr(angle, "__iter__"):
        return [wrap_to_2pi(a) for a in angle]
    else:
        raise TypeError("Unsupported data type for angle")


def wrap_to_180(angle):
    """
    Wraps the given angle(s) to 180.

    Parameters
    ----------
    :param angle: The values to wrap.
    :type angle: int, float, list[int], list[float], iterable[int], iterable[float]

    Returns
    --------
    :return: The wrapped values
    :rtype: int, float, list[int], list[float], iterable[int], iterable[float]

    Raises
    -------
    :raise TypeError: If the input data type is unsupported.
    """
    if isinstance(angle, (int, float)):
        return (angle + 180) % 360 - 180
    elif isinstance(angle, list):
        return [wrap_to_180(a) for a in angle]
    elif isinstance(angle, tuple):
        return tuple(wrap_to_180(a) for a in angle)
    elif hasattr(angle, "__iter__"):
        return [wrap_to_180(a) for a in angle]
    else:
        raise TypeError("Unsupported data type for angle")


def wrap_to_360(angle):
    """
    Wraps the given angle(s) to 360.

    Parameters
    ----------
    :param angle: The values to wrap.
    :type angle: int, float, list[int], list[float], iterable[int], iterable[float]

    Returns
    --------
    :return: The wrapped values
    :rtype: int, float, list[int], list[float], iterable[int], iterable[float]

    Raises
    -------
    :raise TypeError: If the input data type is unsupported.
    """
    if isinstance(angle, (int, float)):
        return angle % 360
    elif isinstance(angle, list):
        return [wrap_to_360(a) for a in angle]
    elif isinstance(angle, tuple):
        return tuple(wrap_to_360(a) for a in angle)
    elif hasattr(angle, "__iter__"):
        return [wrap_to_360(a) for a in angle]
    else:
        raise TypeError("Unsupported data type for angle")
