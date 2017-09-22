import os
import sys
import math
import numpy as np

# Class to hold the ranges
class Manager:
    def __init__(self, lst):
        self.lst = lst
        self.eps = 0.2 * np.std(lst)
        self.length = len(lst)
        self.min_n = math.sqrt(len(lst))
        self.ranges = []

# Bins which will be updated as we go along
class Bin:
    def __init__(self):
        self.low = float('Inf')
        self.high = -float('Inf')
        self.span = 0.0
        self.n = 0
        self.terms = []

    def __str__(self):
        return "{span=%f, low=%f, n=%d, high=%f}" %(self.span, self.low,
                                                    self.n, self.high)

def update(r, value, f):
    r.n += 1

    val = f(value)
    if r.low > val:
        r.low = val
    if r.high < val:
        r.high = val

    r.terms.append(value)
    r.span = r.high - r.low
    return r

# This function expects a list and function to extract which value of x
def main(lst, f):
    r = Manager(sorted(lst, key=lambda l: f(l)))
    r.ranges.append(Bin())

    for index, value in enumerate(r.lst):
        r.ranges[-1] = update(r.ranges[-1], value, f)
        temp = r.ranges[-1]
        if index > 1 and temp.n > r.min_n and temp.span > r.eps and (r.length - index) > r.min_n:
            r.ranges.append(Bin())

    return r.ranges
