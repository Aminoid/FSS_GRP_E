import os
import sys
import time

start = time.time()

row_index = 1

def format_value(value):
    try:
        fval = float(value)
        if fval.is_integer():
            return int(value)
        else:
            return fval
    except ValueError:
        return value

def show_error(message):
    print "[Error] Line %d: %s" %(row_index, message)
    sys.exit(0)

def sanitize_line(line, f):
    global row_index
    # Check for comments and commas
    temp = line.split("#")[0].rstrip()
    while (temp[-1] == ','):
        row_index += 1
        temp += f.readline()
        temp = temp.split("#")[0]
    # Removing white spaces before and after column values
    tokens = temp.split(",")
    str = ""
    for token in tokens:
        str += token.strip() + ","
    return str[:-1], f

with open("POM3A.csv", "rb") as fp:
    ignore_cols = []
    num_cols = 0

    header, fp = sanitize_line(fp.readline(), fp)
    cols = header.split(",")
    num_cols = len(cols)
    for index, value in enumerate(cols):
        if "?" in value:
            ignore_cols.append(index)

    while True:
        line = fp.readline()
        if line == '':
            break
        sanitized, fp = sanitize_line(line, fp)
        print sanitized
        cols = sanitized.split(',')
        row = []
        if len(cols) != num_cols:
            show_error("Mismatch in number of cols w.r.t header")
        for index, value in enumerate(cols):
            if index not in ignore_cols:
                row.append(format_value(value))
        print row

end = time.time()

print "=" * 40
print "Time taken: %f" %(end - start)
