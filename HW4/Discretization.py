import Range
import Superrange
import numpy as np

np.random.seed(1)

import Random as R
import Num as N
import Range as Ra
import Superrange

def x(z):
    return z[0]

def y(z):
    return z[-1]

def klass(z):
  val = 0.0
  if z < 0.2:
    val = 0.2 + 2*R.r()/100
  elif z < 0.6:
    val = 0.6 + 2*R.r()/100
  else:
    val = 0.9 + 2*R.r()/100
  return val

def nested_val(z):
    v = 2*np.random.rand()/100

    if z < 0.2:
        v += 0.2
    elif z < 0.6:
        v += 0.6
    else:
        v += 0.9
    return v

lst = list(np.random.rand(50))
for i, val in enumerate(lst):
    lst[i] = [val, nested_val(val)]

def hw3(lst, x, y):
    r = Range.main(lst, x)

    print "\nWe have many unsupervised ranges."
    for index, value in enumerate(r):
        print "x, %d, %s" %(index + 1, value)

    print "=" * 60

    print "\nWe have fewer supervised ranges."

    breaks = Superrange.main(r, x, y)

    for index, value in enumerate(breaks):
        print "super, %d, {label=%d, most=%f}" %(index, index, value)
    print "\n"

    return Superrange.labels(breaks)

if __name__ == "__main__":
    t, n = [], N.c()
    for _ in range(1,51):
        w = R.r()
        k = klass(w)
        n.update(k)
        t.append(list({w, k}))
        #print(str(k))
    hw3(t,x,y)