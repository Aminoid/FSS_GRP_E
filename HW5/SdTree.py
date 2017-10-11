from __future__ import print_function
import Num as num
import Lists as lst
import Tbl as tbl
import itertools
import config
import Tbl
from Tbl import Tbl

the = config

class Range:
    def __init__(self, t, yfun, pos, attr, val):
        self._t = t
        self._kids = []
        self.yfun = yfun
        self.pos = pos
        self.attr = attr
        self.val = val
        self.stats = num.c().updates(t.rows, yfun)

def create(t, yfun, pos=None, attr=None, val=None):
    return Range(t, yfun, pos, attr, val)

def fun(x): return x.val

def order(t, y):

    def xpect(col):
        tmp = 0
        for _, x in col.nums.items():
            tmp = tmp + x.sd * x.n / col.n
        return tmp

    def whatif(head, y):
        class Detail:
            def __init__(self, head):
                self.pos = head.pos
                self.what = head.txt
                self.nums = {}
                self.n = 0

        col = Detail(head)
        for _, row in enumerate(t.rows):
            x = row.cells[col.pos]
            if x != '?':
                col.n = col.n + 1
                p = col.nums.get(x) or num.c()
                p.update(y(row))
                col.nums[x] = p

        class KeyVal:
            def __init__(self, key, val):
                self.key = key
                self.val = val

        return KeyVal(xpect(col), col)

    out = []
    for _,h in enumerate(t.x.cols):
        out.append(whatif(h,y))
    out = sorted(out, key=lambda x: x.key)
    return lst.collect(out, fun)

def grow1(above, yfun, rows, lvl, b4, pos=None, attr=None, val=None):
    def pad(): return ':20'.format(itertools.repeat('| ', lvl))
    def likeAbove():
        return above._t.copy(rows)
    if len(rows) >= 2:
        if lvl <= 10:
            here = above if (lvl == 0) else create(likeAbove(), yfun, pos, attr, val)
            if here.stats.sd < b4:
                if lvl > 0:
                    above._kids.append(here)
                cuts = order(here._t, yfun)
                cut = cuts[1].val
                kids = {}
                for _, r in enumerate(rows):
                    val = r.cells[cut.pos]
                    if val != '?':
                        rows1 = kids.get(val) if kids.get(val) != None else []
                        rows1.append(r)
                        kids[val] = (rows1)
                for val, rows1 in kids.items():
                    if len(rows1) < len(rows):
                        grow1(here, yfun, rows1, lvl + 1, here.stats.sd, cut.pos, cut.what, val)

def grow(t, y):
  yfun = y
  root = create(t,yfun)
  grow1(root, yfun, t.rows,0,2**32)
  return root

def leaf(tr,cells, bins, lvl):
  lvl=lvl if lvl != 0 else 0
  for j,kid in enumerate(tr._kids):
    pos,val = kid.pos, kid.val
    if cells[kid.pos] == kid.val:
      return leaf(kid, cells, bins, lvl+1)
  return tr

def tprint(tr, lvl=0):
      def pad():
          return "| " * (lvl - 1)

      def left(x):
          return "%-20s" % x

      lvl = lvl or 0
      suffix = ""
      if len(tr._kids) == 0 or lvl == 0:
          suffix = "n=%s mu=%-.2f sd=%-.2f" % (tr.stats.n, tr.stats.mu, tr.stats.sd)
      if lvl == 0:
          print("\n{}".format(suffix))
      else:
          # must_be = left( "{}{} = {}".format(pad(), tr.attr or "", tr.val or ""))
          print(left("{}{} = {}".format(pad(), str(tr.attr) or "", str(tr.val) or "")), suffix, sep='\t:\t  ')
      for j in range(len(tr._kids)):
          tprint(tr._kids[j], lvl + 1)

def test(f, y):
    the.tree_min = 10
    y = y or "dom"
    f = f or "auto.csv"

    tb1 = Tbl(f)
    t2 = tb1.discretizeRows(y, tb1)

    # for head in t2.x.cols:
    #     if head.bins:
    #         print(len(head.bins), head.txt)

    tr = grow(t2, y=t2.dom(tb1))
    tprint(tr)
    #show(tr)
# print(t2.spec)
if __name__ == "__main__":
    test("/home/rahulgutal4/auto.csv", "dom")