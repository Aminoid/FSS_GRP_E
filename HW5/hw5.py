import contrasts as CON
import trees as TREES
import SdTree as TREE
import sys
import Tbl

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hw2.py <csv_file>")
        sys.exit(0)
    else:
        file = sys.argv[1]
        x = TREES.auto(file, "dom")

        #TREE.tprint(x)
        b = CON.branches(x)
        plan = CON.plans(b)
        print "The Plans are given below: \n"
        for i in plan:
            print i

        print "\nThe monitors are given below: \n"
        monitor = CON.monitors(b)
        print monitor
