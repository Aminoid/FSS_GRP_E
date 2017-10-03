import math
import Random as R

class c:
    def __init__(self, most):
        self._all = []
        self.n = 0
        self.most = most
    def __str__(self):
        return str(self.most)

def create(most):
    return c(512)

def update(i, x):
    i.n = i.n + 1
    r = R.r();
    if len(i._all) < i.most:
        i._all.append(x)
    elif r < len(i._all)/i.n:
        i._all[math.floor(1+r*len(i._all))] = x
    return x
