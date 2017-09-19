import Random as R
import Num as N
import Range as Ra

def x(val):
    return val

def klass(z):
  val = 0.0
  if z < 0.2:
    val = 0.2 + 2*R.r()/100
  elif z < 0.6:
    val = 0.6 + 2*R.r()/100
  else:
    val = 0.9 + 2*R.r()/100
  return val

if __name__ == "__main__":
    t, n = [], N.create()
    for _ in range(1,51):
        w = R.r()
        N.update(n, klass(w))
        t.append(w)
    print("\nWe have many unsupervised ranges.")
    for j, one in enumerate(Ra.main(t, x)):
        print("x", j+1, one)
        #print("\nWe have fewer supervised ranges.")
        #for j, one in enumerate(SUPER(t, x, y))
        #print("super", j, one)