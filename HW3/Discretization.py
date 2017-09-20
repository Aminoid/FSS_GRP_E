import Random as R
import Num as N
import Range as Ra

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

if __name__ == "__main__":
    t, n = [], N.create()
    for _ in range(1,51):
        w = R.r()
        k = klass(w)
        N.update(n, k)
        t.append(list({w, k}))
        #print(str(k))
    print("\nWe have many unsupervised ranges.")
    for j, one in enumerate(Ra.main(t, x)):
        print("x", j+1, str(one))
        #print one
        #print("\nWe have fewer supervised ranges.")
        #for j, one in enumerate(SUPER(t, x, y))
        #print("super", j, one)