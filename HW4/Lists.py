def copy(t):
    return type(t) != 'table' and t or collect(t, copy)

def collect(t,f):
    out=[]
    if t:
        for i,v in enumerate(t):
            out.append(v)
    return out

def mprint(ts, sep):
    sep = sep or ", "
    fmt, w = {}, {}
    def width(col, x):
        if not w.get(col):
            w[col] = 0
        tmp = len(str(x))
        if tmp > w.get(col):
            w[col] = tmp
            fmt[col] = "{}{}{}".format("%", tmp, "s")
    for t in ts:
        for col, x in enumerate(t):
            width(col, x)
    for i in ts:
        def to_list(dict):
            ls = []
            for key in dict.keys():
                ls.append(dict.get(key))
            return ls