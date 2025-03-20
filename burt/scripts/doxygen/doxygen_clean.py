#!/usr/bin/env python3

import sys, os, shutil

# change direcotry to rectory with current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def rmFolder(folder, helpMsgIfFailed = None):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        print("Folder '", folder, "' has been cleaned...")
    else:
        print("Folder '", folder, "' does not exist...")
        if helpMsgIfFailed:
            print("INFO:", helpMsgIfFailed)

rmFolder("./generated", "It seems you already cleaned the directory")
