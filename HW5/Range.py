from Sample import create as sc, update as su
from Num import c

class Range:
    def __init__(self, all):
        self._all = all
        self.n = 0
        self.hi = -2 ** 63
        self.lo = 2 ** 63
        self.span = 2 ** 64

    def __str__(self):
        return "span=%f, lo=%f, n=%f, hi=%f" % (self.span, self.lo, self.n, self.hi)

class Range1:
    def __init__(self, x, size):
        self.x = x
        self.cohen = 0.2
        self.m = 0.5
        self.size = size
        self.ranges = []

    def __str__(self):
        return str(self.size)

def create():
    _all = sc(512)
    return Range(_all)

def nextRange(i):
    i.now = create()
    i.ranges.append(i.now)

def rangeManager(t, x):
    _ = Range1(x, len(t))
    nextRange(_)
    c1 = c()
    _.num = c1.updates(t, _.x)
    _.hi = _.num.hi
    _.enough = _.size**_.m
    _.epsilon = _.num.sd*_.cohen
    return _

def update(i, one, x):
    #print x
    su(i._all, one)
    i.n = i.n + 1
    if x > i.hi:
        i.hi = x
    if x < i.lo:
        i.lo = x
    i.span = i.hi - i.lo
    return x

def main(t, x, last=-1):
    #t = list(filter(lambda v : v != '?', t))
    t = sorted(t, key=lambda v: x(v))
    i = rangeManager(t, x)
    for j, one in enumerate(t):
        x1 = x(one)
        #print x1
        update(i.now, one, x1)
        if j > 1 and x1 > last and i.now.n > i.enough and i.now.span > i.epsilon and (i.num.n - j) > i.enough  and (i.num.hi - x1) > i.epsilon :
            nextRange(i)
        last = x1

    return i.ranges
