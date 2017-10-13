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

        # Generate and print the tree
        x = TREES.auto(file, "dom")
        print "The Tree is given below:"
        TREE.tprint(x)

        b = CON.branches(x)

        #Print the Plans
        print "\nThe Plans are given below:"
        CON.plans(b)

        #Print the Monitors
        print "\nThe Monitors are given below:"
        CON.monitors(b)
