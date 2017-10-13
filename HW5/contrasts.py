#import show
import config as THE
import SdTree as TREE
import trees as TREES
import Lists as LST
import Num as NUM
import copy

def has(branch):
    out = []
    for step in branch:
        out.append({"attr":step["attr"], "val":step["val"]})
    return out

def have(branches):
    for branch in branches:
        branch.append(has(branch))
    return branches

def branches1(tr,out,b):
    if tr.attr:
        b.append({"attr":tr.attr,"val":tr.val,"_stats":tr.stats})
    if len(b) > 0:
        out.append(b)
    for kid in tr._kids:
        branches1(kid,out,copy.deepcopy(b))
    return out

def branches(tr):
    return have(branches1(tr,[],[]))

def member2(twin0,twins):
    for twin1 in twins:
        if twin0['attr'] == twin1['attr'] and twin0['val'] == twin1['val']:
            return True
    return False

def delta(t1,t2):
    out = []
    for twin in t1:
        if not member2(twin,t2):
            out.append([twin['attr'], twin['val']])
    return out

def contrasts(branches,better):
    for i,branch1 in enumerate(branches):
        out = []
        for j,branch2 in enumerate(branches):
            if i != j:
                num1 = branch1[-2]["_stats"]
                num2 = branch2[-2]["_stats"]
                if better(num2.mu, num1.mu):
                    if not NUM.c().same(num1,num2):
                        inc = delta(branch2[-1], branch1[-1])
                        if len(inc) > 0:
                            out.append({'i':i,'j':j,'ninc':len(inc),'muinc':num2.mu-num1.mu,
                                       'inc':inc,'branch1':branch1[-1],'mu1':num1.mu,
                                       'branch2':branch2[-1],'mu2':num2.mu})
        if len(out) > 0:
            print ""
            out = sorted(out, key=lambda x: x['muinc'])
            print i, 'max mu', out[0]
            out = sorted(out, key=lambda x: x['ninc'])
            print i, 'min inc', out[0]

def more(x,y):
    return x > y

def less(x,y):
    return x < y

def plans(branches):
    return contrasts(branches,more)

def monitors(branches):
    return contrasts(branches,less)
