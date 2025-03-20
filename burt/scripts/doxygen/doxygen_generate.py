#!/usr/bin/env python3

import subprocess, sys, os, shutil

# change direcotry to rectory with current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if os.name == 'posix' or sys.platform.startswith('linux'):
    # Linux and macOS
    doxygen = r'doxygen'
    graphviz_path = r''

    print("Current OS is: Posix")
    os.environ["USE_DOT"] = 'YES'
    os.environ["USE_CHM"] = 'NO'

elif os.name == 'nt' or sys.platform == 'win32':
    # Windows OS
    doxygen       = r'C:/Program Files/doxygen/bin/doxygen.exe'
    graphviz_path = r'C:/Program Files/Graphviz/bin'
    hhc_path = r'C:/Program Files (x86)/HTML Help Workshop/hhc.exe'

    print("Current OS is: Windows")

    os.environ["PATH"] += os.pathsep + '''C:/Program Files (x86)/HTML Help Workshop'''
    os.environ["HHC_APP"] = hhc_path
    os.environ["DOT_APP_PATH"] = graphviz_path
    os.environ["USE_DOT"] = 'YES'
    os.environ["USE_CHM"] = 'YES'
else:
    print("UNDEFINED OS")
    sys.exit(-1)

os.environ["MY_PROJECT_NAME"]      = "BurTorch"
os.environ["MY_INPUT_DIRECTORIES"] = "./../../burt/ ./../../ ./../../../"


try:
    #shutil.rmtree("./generated")
    print(doxygen)
    ret_code = subprocess.call([doxygen, 'doxygen.conf'])
    print("Documentation generation completed [OK] in: ", "./generated/")

except:
    print("Some error happend during documentation generation")
    print("Information about exception: ", sys.exc_info()[0])
