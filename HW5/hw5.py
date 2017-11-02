import contrasts as CON
import trees as TREES
import SdTree as TREE
import sys
import Tbl

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hw2.py <scv_file>")
        sys.exit(0)
    else:
        file = sys.argv[1]
        x = TREES.auto(file, "dom")
        TREE.tprint(x)
        b=CON.branches(x)
        print "Number of branches: ", len(b)
        print "********************Plans********************"
        CON.plans(b)
        print
        print "********************Monitors********************"
        CON.monitors(b)

