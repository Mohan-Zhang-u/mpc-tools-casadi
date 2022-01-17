#!/usr/bin/env python3
"""Adds all distribution files to a zip file for upload to Bitbucket."""

import argparse
import os
import sys
import zipfile
import mpctools.git

# Command-line arguments.
parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument("--help", help="print this help", action="help")
group = parser.add_mutually_exclusive_group()
group.add_argument("--root-folder", help="name for root folder in zip file")
group.add_argument("--no-root-folder", action="store_true",
                   help="don't include root folder in zip file")
parser.add_argument("--name", help="specify name for zip file",
                    default="mpc-tools-casadi.zip")
parser.add_argument("--python2", help="make distribution for Python 2.7",
                    action="store_true")
parser.add_argument("--windows", help="use Windows newlines in text files",
                    action="store_true")
parser.add_argument("files", default=[], nargs="*",
                    help="Files to include")

# Constants.
CHANGESET_ID = mpctools.git.get_changeset_id()
PYTHON_2_HEADER = "from __future__ import division, print_function"
README_MD = "README.md"
README_TXT = "README.txt"

# Helper functions.
def clean_py_file(file, python2=False):
    """Iterator for cleaned file."""
    with open(file, "r") as read:
        read = Peekable(read)
        if python2:
            yield from get_python_header(read)
            yield PYTHON_2_HEADER
        for line in read:
            line = line.rstrip()
            if "#CHANGESET_ID" in line:
                pad = line[:len(line) - len(line.lstrip())]
                line = '{}changeset_id = "{}"'.format(pad, CHANGESET_ID)
            if python2 and "from .compat import" in line:
                continue
            yield line


class Peekable:
    """Iterator with the ability to peek at the next value."""
    def __init__(self, fromiter):
        """Stores the original iterator."""
        self.__fromiter = fromiter
        self.__peek = self.NOPEEK
    
    def __iter__(self):
        """Returns self."""
        return self
    
    def __next__(self):
        """Returns next item in the iterator."""
        if self.__peek is self.NOPEEK:
            val = next(self.__fromiter)
        else:
            val = self.__peek
            self.__peek = self.NOPEEK
        return val
    
    def peek(self):
        """Peeks at the next value in the iterator."""
        if self.__peek is self.NOPEEK:
            self.__peek = next(self.__fromiter)
        return self.__peek
    
    NOPEEK = object() # Sentinel object.


def get_python_header(file):
    """Yields lines for the header of a Python file."""
    indocstring = False
    while True:
        line = file.peek().strip()
        if indocstring:
            if line.endswith('"""'):
                indocstring = False
        else:
            if line.startswith('"""'):
                indocstring = not line.endswith('"""')
            elif not line.startswith("#") and len(line) > 0:
                break
        yield next(file).rstrip()


def clean_txt_file(file):
    """Iterator for cleaned txt file."""
    with open(file, "r") as read:
        for line in read:
            yield line.rstrip("\n")


def clean_setup_file(file):
    """Iterator to replace the docstring in the setup file."""
    with open(README_MD, "r") as read:
        for (i, line) in enumerate(read):
            line = line.rstrip("\n").replace('"', '\\"')
            if i == 0:
                line = '"""' + line
            yield line
    yield '"""'
    with open(file, "r") as read:
        for (i, line) in enumerate(read):
            if i > 0:
                yield line.rstrip("\n")

                
def makefileolderthan(target, relto, delta=1, changeatime=False,
                      mustexist=False):
    """
    Sets modification time of target to be older than relto.
    
    Argument delta (default 1) gives the time difference to use. Argument
    chanteatime decides whether to also change the access time.
    
    If target does not exist and mustexist is False, nothing happens; if
    mustexist is True, then an error is raised.
    
    Returns mtime if set, otherwise None.
    """
    if os.path.isfile(target):
        mtime = os.path.getmtime(relto) - delta
        atime = mtime if changeatime else os.path.getatime(relto)
        os.utime(target, (atime, mtime))
    elif mustexist:
        raise IOError("File {} does not exist!".format(target))
    else:
        mtime = None
    return mtime


# Main function.
def main(files, zipname, root="", python2=False, newline="\n"):
    """Writes the zip file."""
    files = set(files)
    includereadme = (README_MD in files)
    if includereadme:
        files.remove(README_MD)
    files.discard("mpctools/hg.py")
    if python2:
        files.discard("mpctools/compat.py")
    if root is None:
        root = os.path.splitext(os.path.split(zipname)[1])[0]
    
    # Now add files.
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            readfile = f
            writefile = os.path.join(root, f)
            if f == "mpctoolssetup.py":
                z.writestr(writefile, newline.join(clean_setup_file(readfile)))
            elif f.endswith(".py"):
                z.writestr(writefile,
                           newline.join(clean_py_file(readfile,
                                                      python2=python2)))
            elif f.endswith(".pdf"):
                z.write(readfile, writefile)
            else:
                z.writestr(writefile, newline.join(clean_txt_file(readfile)))
        
        # Also add readme with txt extension to play nice with Windows.
        if includereadme:
            z.write(README_MD, os.path.join(root, README_TXT))
        print("Wrote zip file '%s'." % z.filename)


# Script logic.
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    if len(args.files) == 0:
        raise ValueError("Must provide at least 1 file!")
    if args.no_root_folder:
        root = ""
    else:
        root = args.root_folder
    try:
        main(args.files, args.name, root=root, python2=args.python2,
             newline="\r\n" if args.windows else "\n")
    except Exception as exc:
        makefileolderthan(args.name, args.files[0])
        raise RuntimeError("Error writing zip!") from exc
