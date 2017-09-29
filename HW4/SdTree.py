import Num as num
import Lists as lst
import Tbl as tbl
import itertools

class Range:
    def __init__(self, t, yfun, pos, attr, val):
        self._t = all
        self.kids = {}
        self.yfun = yfun
        self.pos = pos
        self.attr = attr
        self.val = val
        self.stats = num.updates(t.rows, yfun)

    def __str__(self):
        return "span=%f, lo=%f, n=%f, hi=%f" % (self.span, self.lo, self.n, self.hi)

def create(t, yfun, pos, attr, val):
    return Range(t, yfun, pos, attr, val)

def xpect(col):
    tmp = 0
    for _,x in enumerate(col.nums):
        tmp = tmp + x.sd*x.n/col.n
    return tmp

def whatif(head, t, y):
    class Detail:
        def __init__(self, head):
            self.pos = head.pos
            self.what = head.txt
            self.nums = {}
            self.n = 0
    col = Detail(head)
    for _,row in enumerate(t.rows):
        x = row.cells[col.pos]
        if x != '?':
            col.n = col.n + 1
            p = col.nums
            if p == None:
                p = num.create()
            col.nums[x] = num.update(p, y(row))
    class KeyVal:
        def __init__(self, key, val):
            self.key = key
            self.val = val
    return KeyVal(xpect(col), col)

def fun(x): return x.val

def order(t, y):
    out = []
    for _,h in enumerate(t.x.calls):
        out.append(whatif(h,y))
    out = sorted(out, key=lambda x,y: x.key < y.key)
    return lst.collect(out, fun)

def grow1(above, yfun, rows, lvl, b4, pos, attr, val):
    def pad(): return ':20'.format(itertools.repeat('| ', lvl))
    def likeAbove(): return tbl.copy(above._t, rows)
    if len(rows) >= 2:
        if lvl <= 10:
            here = above if (lvl == 0) else create(likeAbove(), yfun, pos, attr, val)
            if here.stats.sd < b4:
                if lvl > 0:
                    above._kids.append(here)
                cuts = order(here._t, yfun)
                cut = cuts[1]
                kids = []
                for _, r in enumerate(rows):
                    val = r.cells[cut.pos]
                    if val != '?':
                        rows1 = kids[val] if kids.val != None else []
                        rows1.append(r)
                        kids[val] = rows1
                for val, rows1 in enumerate(kids):
                    if len(rows1) < len(rows):
                        grow1(here, yfun, rows1, lvl + 1, here.stats.sd, cut.pos, cut.what, val)

def grow(t, y):
  yfun = tbl[y](t)
  root = create(t,yfun)
  grow1(root, yfun, t.rows,0,10^32)
  return root

def tprint(tr,    lvl):
  def pad():    return itertools.repeat('| ',lvl-1)
  def left(x):  return ':20'.format(x)
  lvl = lvl if lvl != 0 else 0
  suffix=""
  if len(tr._kids) == 0 or lvl ==0:
      suffix = 'n=%s mu=%-.2f sd=%-.2f'%(tr.stats.n, tr.stats.mu, tr.stats.sd)
  if lvl ==0:
    print("\n" + suffix)
  else:
    print(left(pad() + (tr.attr if tr.attr != None else "") + " = " + (tr.val if tr.val != None else "")) , "\t:",suffix)
  for j in range(1,len(tr._kids)+1):
      tprint(tr._kids[j],lvl+1)

def leaf(tr,cells,  lvl):
  lvl=lvl if lvl != 0 else 0
  for j,kid in enumerate(tr._kids):
    pos,val = kid.pos, kid.val
    if cells[kid.pos] == kid.val:
      return leaf(kid, cells, bins, lvl+1)
  return tr

if __name__ == "__main__":
    print test()

