import math
import Range as Ra
import copy
from Lists import copy as lc
from Num import c
from Sym import s

class Label:
    def __init__(self, nums, i):
        self.most = nums[i]
        self.label = i

    def __str__(self):
        return "most=%f, label=%f" % (self.most, self.label)


def labels(nums):
    out = []
    for i in range(0, len(nums)):
        out.append(Label(nums, i))
    return out

def same(_):
    return _

def sd(_):
    return _.sd

def ent(i):
    if i._ent == None:
        e = 0
        for _,f in enumerate(i.counts):
            e = e - (f/i.n) * math.log((f/i.n), 2)
        i._ent = e
    return i._ent

def below(x,y):
    return x*1.00 < y

def above(x,y):
    return x > y*1.00

def last(x):
    return x[len(x)-1]

def main(things,x,y,   nump=None, lessp=None):

    if y == None:
        y = last

    if nump == None:
        nump = True

    if lessp == None:
        lessp = True

    ## To DO, check if correct
    better = above
    if lessp != None:
        better = below

    what = s()
    z = ent
    if nump == True:
        what = c()
        z = sd
    #z = nump and sd or ent
    breaks, ranges = [], things
    def data(j):
        return ranges[j-1]._all._all

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
        lmemo, rmemo = [None]*(hi+1), [None]*(hi+1)
        memo(hi, lo, lmemo)  # summarize i+1 using i
        memo(lo, hi, rmemo)  # summarize i using i+1
        cut=-1
        lbest=None
        rbest=None
        for j in range(lo, hi):
            l = lmemo[j]
            r = rmemo[j + 1]
            tmp = float(l.n / float(all.n)) * z(l) + float(r.n / float(all.n)) * z(r)
            if (better(tmp, best)):
                cut = j
                best = tmp
                lbest = l
                rbest = r

        if cut != -1:
            bin = combine(lo, cut, lbest, bin, lvl + 1) + 1
            bin = combine(cut + 1, hi, rbest, bin, lvl + 1)
        else:  # -- no cut found, mark everything "lo" to "hi" as "bin"
            breaks.append(-10 ^ 32)
            if ranges[hi].hi > breaks[bin-1]:
                breaks[len(breaks)-1] = ranges[hi].hi
        return bin
    combine(0, len(ranges)-1, memo(0, len(ranges)-1, [None]*len(ranges)), 1, 0)
    return breaks





if __name__ == "__main__":
    v = [10,9,8,6,'?']
    main(v)