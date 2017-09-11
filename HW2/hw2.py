import os
import sys
import time


row_index = 1

with open("auto.csv", encoding='utf-8') as fp:
    lineNo = 0
    while True:
        line = fp.readline()
        if line == b'':
            break
        lineNo += 1
        if (lineNo == 1):
            print(line)
            columnHeaders = line.rstrip().split(",")
            print(columnHeaders)
