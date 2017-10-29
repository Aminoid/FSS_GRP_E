def collect(t,f):
    out=[]
    if t:
        for i,v in enumerate(t):
            out.append(v)
    return out
def copy(t):
    return type(t) != 'table' and t or collect(t, copy)

def last(x):
    return x[-1]

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

        for _, t in enumerate(ts):
            for col, x in enumerate(t):
                width(col, x)

        for i, t in enumerate(ts):
            # TODO: currently printing, in future write to a file
            def dic_to_list(dic):
                out = []
                for key in dic.keys():
                    out.append(dic.get(key))
                return out
        print(str.format("".format(sep.join(dic_to_list(fmt)),"\n"), *t))