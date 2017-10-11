import config
the = config
class s:
    def __init__(self):
        self.n = 0
        self.nk = 0
        self.counts = {}
        self.most = 0
        self.mode = None
        self._ent = None

    def updates(self, lst,f):
        for _, one in enumerate(lst):
            self.update(self, f(one))

    def discretize(self, x):
            r = None
            if x == the.ignore:
                return x
            if not self.bins:
                return x
            for b in self.bins:
                r = b.label
                if x < b.most:
                    break

            return r

    def update(self,x):
        if x != '?':
            self._ent = None
            self.n = self.n + 1
            if not self.counts.get(x):
                self.nk = self.nk + 1
                self.counts[x] = 0
            seen = self.counts[x] + 1
            self.counts[x] = seen
            if seen > self.most:
                self.most, self.mode = seen, x