import math
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
        #print all.lo
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

    def ttest1(self,df,first,last,crit):
        if df <= first:
            return crit[first]
        elif df >= last:
            return crit[last]
        else:
            n1 = first
            while n1 < last:
                n2 = n1*2
                if df >= n1 and df <= n2:
                    old,new = crit[n1], crit[n2]
                    return old + (new - old) * (df - n1) / (n2 - n1)
                n1 = n1*2

    def ttest(self,i,j):
        t = (i.mu - j.mu) / math.sqrt(max(10 ** -64, i.sd ** 2 / i.n + j.sd ** 2 / j.n))
        a = i.sd ** 2 / i.n
        b = j.sd ** 2 / j.n
        df = (a + b) ** 2 / (10 ** -64 + a ** 2 / (i.n - 1) + b ** 2 / (j.n - 1))
        list = [0]*97
        list[3] = 3.182
        list[6] = 2.447
        list[12] = 2.179
        list[24] = 2.064
        list[48] = 2.011
        list[96] = 1.985
        c = self.ttest1(math.floor(df + 0.5),
                   3,
                   96,
                   list )
        return abs(t) > c

    def hedges(self,i,j):
        nom = (i.n - 1) * i.sd ** 2 + (j.n - 1) * j.sd ** 2
        denom = (i.n - 1) + (j.n - 1)
        sp = (nom / denom) ** 0.5
        g = abs(i.mu - j.mu) / sp
        c = 1 - 3.0 / (4 * (i.n + j.n - 2) - 1)
        return (g * c) > 0.38

    def same(self,i,j):
        return not (self.hedges(i, j) and self.ttest(i, j))