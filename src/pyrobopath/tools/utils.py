from itertools import tee
from .types import ArrayLike3


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def string_rgb(text: str, rgb: ArrayLike3):
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"
