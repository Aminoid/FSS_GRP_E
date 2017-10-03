import config

the = config

class c:
    def __init__(self):
        self.n = 0
        self.mu = 0
        self.m2 = 0
        self.sd = 0
        self.hi = -1e32
        self.lo = 1e32
        self.w = 1

    def updates(self, t, f = None, all = None):
        all = all or c()
        def fun(x):
            return x
        f = f or fun
        for _,one in enumerate(t):
            all.update(f(one))
        return all

    def update(self, x):
        if x != the.ignore:
            self.n = self.n + 1
            if x < self.lo:
                self.lo = x
            if x > self.hi:
                self.hi = x
            delta = x - self.mu
            self.mu = self.mu + delta/self.n
            self.m2 = self.m2 + delta*(x-self.mu)
            if self.n > 1:
                self.sd = (self.m2/(self.n-1))**0.5


    def norm(self, x):
        if x == the.ignore:
            return x
        return (x - self.lo) / (self.hi - self.lo + 1e-32)