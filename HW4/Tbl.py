import Lists as LST
import Num as NUM
import Sym as SYM
import Row as ROW

class Val:
    def __init__(self):
        self.nums = []
        self.syms = []
        self.cols = []

class Table:
    def __init__(self):
        self.rows = []
        self.spec = []
        self.goals = []
        self.less = []
        self.more = []
        self.all = Val()
        self.x = Val()
        self.y = Val()

def create():
    return Table()

def categories(i,txt):
    class Detail:
        def __init__(self, when, what, weight, where):
            self.when = when
            self.what = what
            self.weight = weight
            self.where = where
    spec = []
    where = [i.all.cols, i.x.cols, i.all.nums,                  i.x.nums]
    spec.append(Detail("%$", NUM, 1, where))
    where = [i.all.cols, i.y.cols, i.all.nums, i.goals, i.less, i.y.nums]
    spec.append(Detail("<", NUM, -1, where))
    where = [i.all.cols, i.y.cols, i.all.nums, i.goals, i.more, i.y.nums]
    spec.append(">", NUM, 1, where)
    where = [i.all.cols, i.y.syms, i.y.cols, i.all.syms]
    spec.append("!", SYM, 1, where)
    where = [i.all.cols, i.x.cols, i.all.syms, i.x.syms]
    spec.append("", SYM, 1, where)
    for _,want in enumerate(spec):
      if txt.find(want.when) != None:
          return want.what, want.weight, want.where

def data(i,cells,old):
  new = ROW.update(ROW.create(),cells,i)
  i.rows.append(new)
  if old:
    new.id=old.id
  return new

def header(i,cells):
  i.spec = cells
  for col,cell in enumerate(cells):
    what, weight, wheres = categories(i,cell)
    one = what.create()
    one.pos   = col
    one.txt   = cell
    one.what  = what
    one.weight= weight
    i.name[one.txt] = one
    for _,where in enumerate(wheres):
      where.append(one)
  return i

def copy(i, f):
  j=create()
  header(j, LST.copy(i.spec))
  if f=="full":
    for _,r in enumerate(i.rows):
      data(j, LST.copy(r.cells))
  elif type(f)=='number':
    LST.shuffle(i.rows)
    for k in range(1,f+1):
      data(j, LST.copy(i.rows[k].cells))
  elif type(f)=='table':
    for _,r in enumerate(f):
      data(j, LST.copy(r.cells),r)
  return j