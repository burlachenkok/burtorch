#!/usr/bin/env python3

import sys, os, time, subprocess, shutil

# Prededfined toolchain for build project to Windows OS
used_vs_verions    = "2022"

#======================================================================

sys.path.append(os.path.dirname(sys.argv[0]))

#======================================================================
def rmFolder(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        print("Folder '", folder, "' has been cleaned...")
    else:
        print("Folder '", folder, "' does not exist...")

def safeCall(cmdline, exit_on_error = True):
    print("Run command: ", cmdline)
    print("")

    ret = subprocess.call(cmdline, shell=True)
    print("RETURN CODE: ", ret, " (FROM LAUCNHING ", cmdline, ")")

    if exit_on_error and ret != 0: 
        sys.exit(ret)

def runCmake(binary_tree, source_tree, cmake_cmdline):
    currDir = os.getcwd()

    binary_tree = os.path.abspath(binary_tree)
    source_tree = os.path.abspath(source_tree)
    if not os.path.isdir(binary_tree):
        os.makedirs(binary_tree)
    os.chdir(binary_tree)
    toRun = "%s %s %s" % ('cmake', cmake_cmdline, source_tree)
    print("Run command: ", toRun)
    print("")
    ret = subprocess.call(toRun, shell = True)
    print("Return code from launching 'cmake':", ret)

    os.chdir(currDir)
#======================================================================

if __name__ == "__main__":
    #======================================================================
    # Append standart path to CMake for Windows OS
    if os.name == 'nt' or sys.platform == 'win32':
        os.environ["PATH"] += os.pathsep + "C:/Program Files/CMake/bin"
    #======================================================================

    print("**********************************************************")
    print("Path to python interpretator:", sys.executable)
    print("Version:", sys.version)
    print("Platform name:", sys.platform)
    print("**********************************************************")

    #======================================================================

    buildDebug          = False
    buildRelease        = False

    generateDebug       = False
    generateRelease     = False

    cleanUp             = False
    tests_debug         = False
    tests_release       = False
    use_unity_build     = False

    use_precompiled_headers = False
    use_gnu_code_coverage   = False
    use_llvm_opt_viewer     = False
    use_doxygen_doc         = False
    use_clang_for_vc        = False
    use_ninja               = False

    info_debug         = False
    info_release       = False

    parallel_build_jobs = 1
    #======================================================================

    for i in range(0, len(sys.argv)):
        if sys.argv[i] == "-help" or sys.argv[i] == "--help" or sys.argv[i] == "-h" or len(sys.argv) == 1:
            print("Python version: ", sys.version)
            print(
'''
_____________________________________________________________________________________________________________________________
Use cases:                                                                                                                   |
_____________________________________________________________________________________________________________________________|
project_script.py -help                | -h       -- to print this information text into console                        |
project_script.py -generate_debug      | -gd      -- generate debug version                                             |
project_script.py -generate_release    | -gr      -- generate release version                                           |
project_script.py -build_debug         | -bd      -- build debug version                                                |
project_script.py -build_release       | -br      -- build release version                                              |
project_script.py -clean               | -c       -- cleanup folder with build artifacts                                |
project_script.py -tests_debug         | -td      -- launch debug unit tests                                            |
project_script.py -tests_release       | -tr      -- launch release unit tests                                          |
project_script.py -vc_version          | -vs      -- provide specific version of Visual Studio (2022|2019|...)          |
project_script.py -use_clang_for_vc    | -uc      -- use clang toolset for Visual Studio (2022|2019|...)                |
project_script.py -use_ninja           | -un      -- use Ninja build system (can be faster than GNU Make)               |
project_script.py -parallel_jobs       | -j       -- use a specific number of CPU cores in the building process         |
project_script.py -unity_build         | -ub      -- use unity build to speedup compilation time                        |
project_script.py -use_pch             | -up      -- use precompile headers o speedup compilation time (g++ has issues) |
project_script.py -gnu_code_coverage   | -gc      -- turn on generating code coverage                                   |
project_script.py -llvm_opt_viewer     | -ov      -- turn on LLVM optimization compilers remarks to study them          |
project_script.py -doxygen             | -doc     -- generate doxygen automatic documentation                           |
_____________________________________________________________________________________________________________________________|
'''                                                                                              
                 )
            sys.exit(0)
        elif sys.argv[i] == "-vc_version" or sys.argv[i] == "-vs":
            used_vs_verions = sys.argv[i+1]
            print("+USED VISUAL STUDIO VERSION: ", used_vs_verions)
        elif sys.argv[i] == "-generate_debug" or sys.argv[i] == "-gd":
            generateDebug = True
            print("+BUILD_DEBUG")
        elif sys.argv[i] == "-generate_release" or sys.argv[i] == "-gr":
            generateRelease = True
            print("+BUILD_RELEASE")
        elif sys.argv[i] == "-build_debug" or sys.argv[i] == "-bd":
            buildDebug = True
            print("+BUILD_DEBUG")
        elif sys.argv[i] == "-build_release" or sys.argv[i] == "-br":
            buildRelease = True
            print("+BUILD_RELEASE")
        elif sys.argv[i] == "-tests_debug" or sys.argv[i] == "-td":
            tests_debug = True
            print("+TESTS DEBUG")
        elif sys.argv[i] == "-tests_release" or sys.argv[i] == "-tr":
            tests_release = True
            print("+TESTS RELEASE")
        elif sys.argv[i] == "-clean" or sys.argv[i] == "-c":
            cleanUp = True
            print("+CLEAN")
        elif sys.argv[i] == "-parallel_jobs" or sys.argv[i] == "-j" or sys.argv[i] == "-parallel":
            parallel_build_jobs = int(sys.argv[i+1])
            print(f"+DURING BUILDING ALLOW IN PARALLEL {parallel_build_jobs} JOBS")
        elif sys.argv[i] == "-unity_build" or sys.argv[i] == "-ub":
            use_unity_build = True
            print("+USE UNITY BUILD")
        elif sys.argv[i] == "-use_pch" or sys.argv[i] == "-up":
            use_precompiled_headers = True
            print("+USE PRECOMPILED HEADERS")
        elif sys.argv[i] == "-gnu_code_coverage" or sys.argv[i] == "-gc":
            use_gnu_code_coverage = True
            print("+GENERATING GNU CODE COVERAGE")
        elif sys.argv[i] == "-llvm_opt_viewer" or sys.argv[i] == "-ov":
            use_llvm_opt_viewer = True
            print("+GENERATING LLVM REMARKS FOR OPT-VIEWER2")
        elif sys.argv[i] == "-doxygen" or sys.argv[i] == '-doc':
            use_doxygen_doc = True
            print("+GENERATING DOXYGEN DOCUMENTATION")
        elif sys.argv[i] == "-use_clang_for_vc" or sys.argv[i] == "-uc":
            use_clang_for_vc = True
            print("+USE CLANG TOOLSET FOR VISUAL STUDIO")
        elif sys.argv[i] == "-use_ninja" or sys.argv[i] == "-un":
            use_ninja = True
            print("+USE NINJA BUILD SYSTEM")
    print("----------------------------------------------------------")

    folder_with_binary = "./build"

    if buildRelease or generateRelease:
        folder_with_binary = "./build_release"

    if buildDebug or generateDebug:
        folder_with_binary = "./build_debug"

    #================================================================================================================
    extra_build_args = ""
    extra_cmake_arguments = ""
    project_generator = ""

    t0 = time.time()

    if os.name == 'nt' or sys.platform == 'win32':
        print("Current OS is: ", "Windows")
        extra_build_args = ""
        vs_versions = {"2022" : "17", "2019" : "16", "2017" : "15", "2015" : "14", "2013" : "12", "2012" : "11", "2010" : "10"}       

        if used_vs_verions == "2019" or used_vs_verions == "2022":
            project_generator = "Visual Studio " + vs_versions[used_vs_verions] + " " + used_vs_verions 
        else:
            project_generator = "Visual Studio " + vs_versions[used_vs_verions] + " " + used_vs_verions + " Win64"

        print("Project files are generated for VS: ", used_vs_verions)

        if use_clang_for_vc:
            extra_cmake_arguments += " -T ClangCL"

    else:
        project_generator = "Unix Makefiles"

    if use_ninja:
        project_generator = "Ninja"

    #================================================================================================================
    parallel_build_jobs_cmd = ""

    if os.name == 'nt' or sys.platform == 'win32':
        parallel_build_jobs_cmd = f"--parallel {parallel_build_jobs}"
    else:
        parallel_build_jobs_cmd = f"-- -j {parallel_build_jobs}"

    #================================================================================================================
    # Cleanup folder
    if cleanUp:
        rmFolder("./build_debug")
        rmFolder("./build_release")

    # Run CMake: with makefiles based generators only one configuration can be active and it's specified by "CMAKE_BUILD_TYPE" variable.
    if generateRelease:
        extra_cmake_arguments += " -DCMAKE_BUILD_TYPE:STRING=Release"
    else:
        extra_cmake_arguments += " -DCMAKE_BUILD_TYPE:STRING=Debug"

    # Unity build
    if use_unity_build:
        extra_cmake_arguments += " -DCMAKE_UNITY_BUILD=ON"

    # LLVM Optmization Remarks
    if use_llvm_opt_viewer:
        extra_cmake_arguments += " -DBURT_LLVM_OPT_VIEWER=ON"

    # Precompiled headers
    if use_precompiled_headers:
        extra_cmake_arguments += " -DCOMPILE_TIME_OPTIMIZATION_USE_PCH=ON"

    # GNU Code coverage
    if use_gnu_code_coverage:
        extra_cmake_arguments += " -BURT_CODE_COVERAGE_GCOV_IS_ON=ON"

    # Add custom flags
    if os.getenv('EXTRA_CMAKE_ARGS') != None:
        extra_cmake_arguments += " " + os.getenv('EXTRA_CMAKE_ARGS')

    print("==========================================================")
    print("Folder with build tree: ", folder_with_binary)
    print("CMake arguments: ", extra_cmake_arguments)
    print("==========================================================")


    if generateDebug or generateRelease:
        genCmdLine = '-G ' + '"' + project_generator + '"' + extra_cmake_arguments
        print("CMake project generation cmdline: ", genCmdLine)
        print("==========================================================")
        runCmake(folder_with_binary, "./../", genCmdLine)

    #================================================================================================================
    # Build Release/Debug versions
    # https://cmake.org/cmake/help/latest/manual/cmake.1.html
    #================================================================================================================
    if buildRelease:
        if not os.path.isdir(folder_with_binary): 
            os.makedirs(folder_with_binary)

        buildCmd = 'cmake --build "%s" --clean-first --config Release %s' % (folder_with_binary + extra_build_args, parallel_build_jobs_cmd)
        print("CMake buid cmdline: ", buildCmd)
        print("==========================================================")

        print(safeCall(buildCmd))
        print("Release binaries are available in: ", folder_with_binary)

    if buildDebug:
        if not os.path.isdir(folder_with_binary): 
            os.makedirs(folder_with_binary)
        buildCmd = 'cmake --build "%s" --clean-first --config Debug %s' % (folder_with_binary + extra_build_args, parallel_build_jobs_cmd) 
        print("CMake buid cmdline: ", buildCmd)
        print("==========================================================")

        print(safeCall(buildCmd))
        print("Debug binaries are available in: ", folder_with_binary)

    #================================================================================================================
    gtest_flags = "--gtest_color=yes --gtest_filter=-*PerfTest.*"
    #================================================================================================================

    if tests_release:
        if os.name == 'nt' or sys.platform == 'win32':
            launchCmd = os.path.join(os.getcwd(), r"build_release\bin_tests\Release\bin_tests.exe " + gtest_flags)
        else:
            launchCmd = os.path.join(os.getcwd(), "build_release/bin_tests/bin_tests " + gtest_flags)
        print(safeCall(launchCmd))

    if tests_debug:
        if os.name == 'nt' or sys.platform == 'win32':
            launchCmd = os.path.join(os.getcwd(), r"build_debug\bin_tests\Debug\bin_tests.exe " + gtest_flags)
        else:
            launchCmd = os.path.join(os.getcwd(), "build_debug/bin_tests/bin_tests " + gtest_flags)
        print(safeCall(launchCmd))


    if use_doxygen_doc:
        safeCall(sys.executable + " " + "./doxygen/doxygen_clean.py")
        safeCall(sys.executable + " " + "./doxygen/doxygen_generate.py")

    print("Completed in: ", str(time.time() - t0), " seconds")
    sys.exit(0)
