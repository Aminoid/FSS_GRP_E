import os
import sys

class Header:
    def __init__(self):
        self.names = []
        self.ignore = []
        self.nums = []
        self.goals = []
        self.syms = []

    def __str__(self):
        return str(self.names)

class Row:
    def __init__(self):
        self.content = []
        self.rank = -1
        self.index = -1

    def __str__(self):
        return "cells=%s, id=%d" %(str(self.content), self.index)

class Goal:
    def __init__(self):
        self.index = -1
        self.weight = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.mean = 0
        self.sd = 0
        self.n = 0

    def get_variance(self):
        return (self.sd / (self.n - 1))

header = Header()
rows = []
contents = []

if len(sys.argv) != 2:
    print "Usage: python hw2.py <scv_file>"
    sys.exit(0)

def convert_to_num(value):
    try:
        val = int(value)
    except ValueError:
        val = float(value)
    return val

def parse_header(index, value):
    global header
    header.names.append(value)
    if "?" in value:
        header.ignore.append(index)
    elif "%$" in value or "<" in value or ">" in value:
        header.nums.append(index)
        if "<" in value or ">" in value:
            temp = Goal()
            temp.index = index
            if "<" in value:
               temp.weight = -1
            else:
               temp.weight = 1
            header.goals.append(temp)
    else:
        header.syms.append(index)


def update_header(content):
    global header

    for i, goal in enumerate(header.goals):
        val = content[goal.index]
        goal.max = max(goal.max, val)
        goal.min = min(goal.min, val)

        goal.n += 1
        if goal.n == 1:
            goal.mean = val
        else:
            goal.mean = goal.mean + (val - goal.mean) / goal.n
            goal.sd = goal.sd + (val - goal.mean) ** 2
    return

def dominate(x, y):
    global header
    global rows

    sum1 = 0
    sum2 = 0
    e = 2.71828

    n = len(header.goals)

    for goal in header.goals:
        weight = goal.weight
        index = goal.index
        mx = goal.max
        mn = goal.min
        norm_x = (x[index] - mn) / (mx - mn)
        norm_y = (y[index] - mn) / (mx - mn)
        sum1 = sum1 - e**(weight * (norm_x - norm_y)/ n)
        sum2 = sum2 - e**(weight * (norm_y - norm_x)/ n)
    return sum1/n < sum2/n

def dom_rank(index, row):
    global rows
    rank = 0
    for i, val in enumerate(rows):
        if i != index:
            if dominate(row.content, val.content):
                rank += 1
    return rank


with open(sys.argv[1], "rb") as fp:
    # Parse Header
    for index, value in enumerate(fp.readline().rstrip().split(',')):
        parse_header(index, value)

    # parse Rows
    for index, value in enumerate(fp.readlines()):
        row = Row()
        row.content = value.rstrip().split(',')
        row.index = index
        for col in header.nums:
            row.content[col] = convert_to_num(row.content[col])

        update_header(row.content)
        rows.append(row)

    for index, row in enumerate(rows):
        row.rank = dom_rank(index, row)

    sort = sorted(rows, key=lambda row: -row.rank)
    print header

    # Top 5
    for i in range(10):
        print sort[i]

    print "\n"

    # Bottom 5
    for i in range(-10, 0):
        print sort[i]
