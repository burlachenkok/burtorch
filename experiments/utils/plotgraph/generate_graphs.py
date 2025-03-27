#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, re, shutil, subprocess

def safeCall(cmdline):
    ret = subprocess.call(cmdline, shell=True)
    print("RETURN CODE: ", ret, " -- FROM LAUCNHING ", cmdline)
    if ret != 0: 
        sys.exit(ret)

if __name__ == "__main__":
    # Append Path to installed GraphViz
    if os.name == 'nt' or sys.platform == 'win32':
        os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"
    if len(sys.argv) < 2:
        print("Launch with following cmdline: <script_name> <input_text_file_with_dot>")
        sys.exit(-1)

    inp = sys.argv[1]
    path, ext = os.path.splitext(inp)
    out = path + ".pdf"

    safeCall(f'dot -T pdf {inp} > {out}')
