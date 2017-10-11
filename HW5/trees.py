#import show
import config
import Tbl
import SdTree as TREE

def create(f,y):
    y = y or 'goal1'
    f = f or "auto.csv"
    tb1 = Tbl.Tbl(f)
    t2 = tb1.discretizeRows(y, tb1)
    return TREE.grow(t2,y = t2.dom(tb1))

def auto(file, y):
    return create(file, y or "dom")

def pom3a(y):
    return create("/data/POM3A_short.csv", y or "dom")

def xomo(y):
    return create("/data/xomo_all_short.csv", y or "dom")

def autogoal1():
    return auto("goal1")

def pom3agoal1():
    return pom3a("goal1")

def xomogoal1(y):
    return xomo("goal1")

#return {auto=auto, pom3a=pom3a, xomo=xomo, autogoal1=autogoal1,
#       pom3agoal1=pom3agoal1, xomogoal1=xomogoal1}