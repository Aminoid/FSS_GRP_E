class s:
    def __init__(self):
        self.n = 0
        self.nk = 0
        self.counts = []
        self.most = 0
        self.mode = None
        self._ent = None
    def __str__(self):
        return str(self.sd)

def create():
    return s()

def updates(lst,f,i):
    i = i or create()
    for _, one in enumerate(lst):
        update(i, f(one))
    return i

def update(i, x):
    if x != '?':
        i._ent = None
        i.n = i.n + 1
        if not i.counts[x]:
            i.nk = i.nk + 1
            i.counts[x] = 0
        seen = i.counts[x] + 1
        i.counts[x] = seen
        if seen > i.most:
            i.most, i.mode = seen, x