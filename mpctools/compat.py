"""Compatibility helpers for Python 3.Module cannot be imported by Python 2."""

from functools import reduce

def execfile(file, *args):
    """
    Executes the given Python file.
    
    *args should be (globals, locals), defining the contexts in which the
    code is executed.
    """
    with open(file, "r") as f:
        code = f.read()
    exec(compile(code, file, "exec"), *args)
