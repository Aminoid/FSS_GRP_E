import math
import Range as Ra

class Label:
    def __init__(self, nums, i):
        self.most = nums[i]
        self.label = i

    def __str__(self):
        return "most=%d, label=%d" % (self.most, self.label)


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

def main(things,x,y,   nump, lessp):
    y = y or last # klass is last arg
    # To Do
    ##Correct this
    nump = nump == nil and true or nump
    lessp = lessp == nil and true or lessp
    better = lessp and below or above
    # To Do
    ##Correct this
    what = nump and NUM or sym
    z = nump and sd or ent
    breaks, ranges = [], Ra.main(things, x)
    def data(j):
        return ranges[j]._all._all

    #To Do
    #Correct this
    def copy(t):
        return type(t) != 'table' and t or collect(t,copy)

    def memo(here, stop, _memo, b4, inc):
        if (stop > here):
            inc = 1
        else:
            inc = -1
        if (here != stop):
            b4 = copy(memo((here + inc), stop, _memo))
        _memo[here] = what.updates(data(here), y, b4)
        return _memo[here]

    def combine(lo, hi, all, bin, lvl):
        best = z(all)
        lmemo, rmemo = [], []
        memo(hi, lo, lmemo)  # summarize i+1 using i
        memo(lo, hi, rmemo)  # summarize i using i+1
        cut=0.0
        lbest=0.0
        rbest=0.0
        for j in range(lo, hi):
            l = lmemo[j]
            r = rmemo[j + 1]
            tmp = l.n / all.n * z(l) + r.n / all.n * z(r)
            if (better(tmp, best)):
                cut = j
                best = tmp
                lbest = copy(l)
                rbest = copy(r)
        if (cut):
            bin = combine(lo, cut, lbest, bin, lvl + 1) + 1
            bin = combine(cut + 1, hi, rbest, bin, lvl + 1)
        else:  # -- no cut found, mark everything "lo" to "hi" as "bin"
            breaks[bin] = breaks[bin] or -10 ^ 32
            if (ranges[hi].hi > breaks[bin]):
                breaks[bin] = ranges[hi].hi
        return bin
    combine(1, len(ranges), memo(1, len(ranges), []), 1, 0)
    return labels(breaks)





if __name__ == "__main__":
    v = [10,9,8,6,'?']
    main(v)
