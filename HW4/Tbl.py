import copy
import Num
import Sym
import Row
import CSV
import Discretization
import config

the = config

class category_details:
    def __init__(self, when, what, weight, where):
        self.when = when
        self.what = what
        self.weight = weight
        self.where = where

class Tbl:
    def __init__(self, file=False):
        self.rows = []
        self.spec = []
        self.goals = []
        self.less = []
        self.more = []
        self.name = {}
        self.all = {"syms": [], "nums": [], "cols": []}
        self.x = {"syms":[], "nums":[], "cols":[]}
        self.y = {"syms":[], "nums":[], "cols":[]}
        if file:
            self.fromCSV(file)

    def categories(self, txt):
        spec = [
            category_details("$", Num.c(), 1, [self.all.get("cols"), self.x.get("cols"), self.all.get("nums"), self.x.get("nums")]),
            category_details("%", Num.c(), 1, [self.all.get("cols"), self.x.get("cols"), self.all.get("nums"), self.x.get("nums")]),
            category_details("<", Num.c(), -1, [self.all.get("cols"), self.y.get("cols"), self.all.get("nums"), self.goals, self.less, self.y.get("nums")]),
            category_details(">", Num.c(), 1, [self.all.get("cols"), self.y.get("cols"), self.all.get("nums"), self.goals, self.more, self.y.get("nums")]),
            category_details("!", Sym.s(), 1, [self.all.get("cols"), self.y.get("syms"), self.y.get("cols"), self.all.get("syms")]),
            category_details("", Sym.s(), 1, [self.all.get("cols"), self.x.get("cols"), self.all.get("syms"), self.x.get("syms")])
        ]
        for want in spec:
            if txt.find(want.when) > -1:
               return want.what, want.weight, want.where

    def header(self, cells):
        self.spec = cells
        for col, cell in enumerate(cells):
            what, weight, wheres = self.categories(cell)
            one = what
            one.pos = col
            one.txt = cell
            one.what = what
            one.weight = weight
            self.name[one.txt] = one
            for _, where in enumerate(wheres):
                where.append(one)
        return self

    def data(self, cells, old=False):
        new = Row.Row().update(cells, self)
        self.rows.append(new)
        if old:
            new.id = old.id
        return new

    def update(self, cells):
        if len(self.spec) == 0:
            return self.header(cells)
        else:
            return self.data(cells)

    def copy(self, _from):
        j = Tbl()
        j.header(copy.deepcopy(self.spec))
        for r in _from:
            j.data(copy.deepcopy(r.cells), r)
        return j

    def dom(self,t):
        b4 = {}
        def dfun(r):
            if not b4.get(r.id):
                b4[r.id] = r.dominate(t)
            return b4[r.id]
        return dfun

    def discretizeHeaders(self, spec):
        out = []
        if spec:
            for _, val in enumerate(spec):
                out.append(val.replace("$",""))
        return out

    def discretizeRows(self, y, t):
        j = Tbl().header(self.discretizeHeaders(self.spec))
        yfun = j.dom(t)
        for head in self.x.get("nums"):
            cooked = j.all.get("cols")[head.pos]
            def x(r) :
                return r.cells[cooked.pos]
            cooked.bins = Discretization.hw3(self.rows, x, yfun)
        for r in self.rows:
            tmp = copy.deepcopy(r.cells)
            for head in self.x.get("nums"):
                cooked = j.all.get("cols")[head.pos]
                old = tmp[cooked.pos]
                new = cooked.discretize(old)
                tmp[cooked.pos] = new
            j.data(tmp, r)
        return j


    def fromCSV(self, f):
        CSV.CSV(f, lambda cells: self.update(cells))
        return self
