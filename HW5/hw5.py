import contrasts as CON
import trees as TREES
import SdTree as TREE
import sys
import Tbl

x = TREES.auto("dom")
TREE.treePrint(x)
b=CON.branches(x)
CON.plans(b)
CON.monitors(b)

#if __name__ == "__main__":
    #if len(sys.argv) != 2:
    #    print("Usage: python hw2.py <scv_file>")
    #    sys.exit(0)
    #else:
        #file = "/home/rahulgutal4/auto.csv"
        #TREE(file, "dom")