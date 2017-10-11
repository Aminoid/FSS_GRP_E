
import math

seed0       = 10013
seed        = seed0
multipler   = 16807.0
modulus     = 2147483647.0
randomtable = None


def park_miller_randomizer():
  global seed
  seed = (multipler * seed) % modulus
  return seed / modulus

def rseed(n):
  if n:
    seed = n
  else:
      seed = seed0
  randomtable = None

def system():
  return rseed(math.random()*modulus)

def r ():
  global randomtable
  if randomtable == None:
    randomtable = []
    for i in range(1,98):
      randomtable.append(park_miller_randomizer())
  x = park_miller_randomizer()
  i = math.floor(97*x)
  #print i
  x, randomtable[int(i)] = randomtable[int(i)], x
  return x