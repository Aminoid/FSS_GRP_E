import config
import re

the = config
notsep = "{}{}{}".format("([^", the.sep, "]+)") #-- not cell seperator
dull = "['\"\t\n\r]*"                           #-- white space, quotes
padding = "%s*(.-)%s*"                          #-- space around words
comments = "#.*"                                #-- comments

class CSV:
    def __init__(self, src, function):
        self.withEachLine(src, WME(function))

    files = { "txt": True, "csv": True}

    def incomplete(self, txt):      #-- must join line to next
        if txt == "" or txt[-1] == '\n' or txt[-1] == the.sep:
            return False
        return True

    def ignored(self, txt):         #-- ignore this column
        return txt.find(the.ignore) < 0

    def cellsWeAreNotIgnoring(self, txt, wme):
        out, col = [], 0
        c        = re.compile(notsep)
        for words in c.finditer(txt):
            word = txt[words.span()[0]:words.span()[1]]
            col = col + 1
            if wme.first:
                wme.use[col] = self.ignored(word)
            if wme.use[col]:
                try:
                    out.append(float(word))
                except ValueError:
                    if word == the.ignore:
                        return False
                    out.append(word)
        return out


    def withOneLine(self, txt, wme):
        txt = txt.replace(padding,"%1")
        txt = txt.replace(dull,"")
        txt = txt.replace(comments,"")
        if len(txt) > 0:
            cells = self.cellsWeAreNotIgnoring(txt, wme)
            if cells:
                wme.fn(cells)

    def withEachLine(self, src, wme):
        self.cache = []
        def line1(line):
            self.cache.append(line)
            if not self.incomplete(line):
                self.cache = self.withOneLine("".join(self.cache), wme)
                self.cache = []
                wme.first = False

        if self.files[src[-3:]]:
            for line in open(src):
                line1(line)
        else:
            for line in src.split("[^\r\n]+"):
                line1(line)

class WME:
    def __init__(self, fn):
        self.first = True
        self.fn = fn
        self.use = {}
