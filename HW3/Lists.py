def collect(t,f):
    out=[]
    if t:
        for i,v in enumerate(t):
            out[i] = f(v)
    return out
def copy(t):
    print "jhjhbvjn"
    return type(t) != 'table' and t or collect(t, copy)
