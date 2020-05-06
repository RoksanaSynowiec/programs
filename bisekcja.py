
from mpmath import nsum, inf
import numpy

C = 0.577215


def f(x):
    S = nsum(lambda n: (2 * n * x + x * x) / (n * (n + x) * (n + x)), [1, inf]) - C
    return S


def bisekcja(x, y, eps):
    if f(x) * f(y) < 0:
        pierwiastek = x
        while (y - x) >= eps:
            pierwiastek = (y + x) / 2.0
            if f(x) * f(pierwiastek) < 0:
                y = pierwiastek
            elif f(pierwiastek) * f(y) < 0:
                x = pierwiastek
            else:
                break

    elif f(x) == 0:
        pierwiastek = x

    elif f(y) == 0:
        pierwiastek = y

    return pierwiastek


print(bisekcja(0,1, 0.0000001))
