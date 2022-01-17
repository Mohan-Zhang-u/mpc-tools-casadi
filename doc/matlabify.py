#!/usr/bin/env python3
import os
import re

def splitfiles(main,directory):
    """
    Splits all functions defined in main and puts them in directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    mainin = open(main, "r")
    mainout = open(directory + "/main.m","w")
    
    startfunc = re.compile(r"^\s*function.*=\s*(\w+)\(")
    endfunc = re.compile(r"^\s*endfunction")
    
    keepgoing = True
    subfun = False
    subfunout = None
    while keepgoing:
        line = transform_source(mainin.readline())
        if len(line) > 0:
            if not subfun:
                m = startfunc.search(line)
                if m:
                    subfunout = open(directory + "/" + m.group(1) + ".m","w")
                    subfunout.write(line)
                    subfun = True
                else:
                    mainout.write(line)
            else:
                m = endfunc.search(line)
                if m:
                    subfunout.write("end\n")
                    subfunout.close()
                    subfun = False
                else:
                    subfunout.write(line)
        else:
            keepgoing = False
    mainout.close()
    if subfunout is not None:
        subfunout.close()

def transform_source(src, matlab=True):
    if matlab:
        src = re.sub(r'end(for|if|switch|while)', r'end', src)
        src = re.sub(r'mysave', r'save', src)
        src = re.sub(r'#.', r'%', src)
        src = re.sub('%%+', '%', src) # Prevents weird things with %% in Matlab editor.
        src = re.sub('!=', '~=', src) # Matlab doesn't know !=.
        src = re.sub(r'lmpc(\s*\()', r'lmpc_matlab\1', src) # call lmpc_matlab instead of lmpc
        src = re.sub(r"^print.*", "", src)
        src = re.sub(r"^.*pkg\(.*\).*$", "", src) # Don't call pkg.
    return src

if __name__ == "__main__":
    splitfiles("../cstr.m","../cstr-matlab")