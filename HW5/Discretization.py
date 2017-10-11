import Range
import Superrange
import numpy as np

np.random.seed(1)

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

    # print "\nWe have many unsupervised ranges."
    # for index, value in enumerate(r):
    #     print "x, %d, %s" %(index + 1, value)
    #
    # print "=" * 60
    #
    # print "\nWe have fewer supervised ranges."
    breaks = Superrange.main(r, x, y)
    # for index, value in enumerate(breaks):
    #     print "super, %d, {label=%d, most=%f}" %(index, index, value)
    # print "\n"
    return Superrange.labels(breaks)
