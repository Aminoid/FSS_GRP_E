import math
import Range as Ra
import copy
from Lists import copy as lc
from Num import c
from Sym import s

class Label:
    def __init__(self, num, i):
        self.most = num
        self.label = i

    def __str__(self):
        return "most=%f, label=%f" % (self.most, self.label)


def labels(nums):
    out = []
    for i in range(1, len(nums)+1):
        out.append(Label(nums[i], i))
    return out

def same(_):
    return _

def sd(_):
    return _.sd

def ent(i):
    if i._ent == None:
        e = 0.0
        for _,f in enumerate(i.counts):
            e = e - (float(f)/i.n) * math.log((float(f)/i.n), 2)
        i._ent = e
    return i._ent

def below(x,y):
    return x*1.00 < y

def above(x,y):
    return x > y*1.00

def last(x):
    return x[len(x)-1]

def main(things,x,y = None,   nump=None, lessp=None):

    if y == None:
        y = last

    if nump == None:
        nump = True

    if lessp == None:
        lessp = True

    ## To DO, check if correct
    better = above
    if lessp == True:
        better = below

    what = s()
    z = ent
    if nump == True:
        what = c()
        z = sd
    #z = nump and sd or ent
    breaks, ranges = {}, things
    def data(j):
        return ranges[j]._all._all

    #To Do
    #Correct this

    def memo(here, stop, _memo, b4=None, inc=None):
        if (stop > here):
            inc = 1
        else:
            inc = -1
        if (here != stop):
            m = memo(here + inc, stop, _memo)
            b4 = copy.deepcopy(m)
            #b4 = m
            #b4 = m
        _memo[here] = what.updates(data(here), y, b4)
        return _memo[here]

    def combine(lo, hi, all, bin, lvl):
        best = z(all)
        lmemo, rmemo = {}, {}
        memo(hi, lo, lmemo)  # summarize i+1 using i
        memo(lo, hi, rmemo)  # summarize i using i+1
        cut=None
        lbest=None
        rbest=None
        for j in range(lo, hi):
            l = lmemo[j]
            r = rmemo[j + 1]
            tmp = (float(l.n) / all.n)*z(l)  + (float(r.n) / all.n)*z(r)
            if (better(tmp, best)):
                cut = j
                best = tmp
                lbest = copy.deepcopy(l)
                rbest = copy.deepcopy(r)

        if cut != None:
            bin = combine(lo, cut, lbest, bin, lvl + 1) + 1
            bin = combine(cut + 1, hi, rbest, bin, lvl + 1)
        else:  # -- no cut found, mark everything "lo" to "hi" as "bin"
            if bin not in breaks:
                breaks[bin] = -1e32
            if ranges[hi].hi > breaks[bin]:
                breaks[bin] = ranges[hi].hi
        return bin
    combine(1, len(ranges)-1, memo(1, len(ranges)-1, {}), 1, 0)
    return breaks





if __name__ == "__main__":
    v = [10,9,8,6,'?']
    main(v)