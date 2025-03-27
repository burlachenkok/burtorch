# BurTorch: Build Locally

----

The goal of this document is to describe how to build a project locally in your Operating System.

----

# Building in IDE
* In [CLion](https://www.jetbrains.com/clion/) or [QtCreator](https://doc.qt.io/qtcreator/), open `burt/CMakeLists.txt` and build within them `Debug` or `Release` configurations.
* For [Visual Studio](https://visualstudio.microsoft.com/) you can open generated project solution. To generate it, use the following:
```
cd ./burt/scripts
./project_scripts.py -c -gr
```

# Building and More via Console

## Reference Table

To assist with various subtle aspects of the building process we have created a helper script `project_scripts.py`. Below are descriptions of the flags and you specify as many flags as you wish.

**Note:** At this moment, it may be a bit overwhelming because you do not need all of them. However, you may wish to return to this table occasionally.

The table itself can be obtained via:
```
cd ./burt/scripts
./project_scripts.py -h
```
                       |
## How to: Build Release Implementation and Run Tests

The first step is to switch to the directory and then invoke the needed commands:
```bash
cd ./burt/scripts
```

Then you can to build a project with the following command line:

`./project_scripts.py -c -gr -br -tr -j 48`
* [-c]  Clean build directory
* [-gr] Generate project files for release build
* [-bd] Start release builds
* [-tr] After the build is finished, launch unit tests for functionality
* [-j 48] During the compilation process, allow the build tool to create a pool of compilation processes equal to 48

**Note:** This command will select some default C and C++ compilers from your system. This choice is made by CMake.


## How to: Build Debug Implementation and Debug Tests

The first step is to switch to the directory and then invoke the needed commands:
```bash
cd ./burt/scripts
```

Then, you can build a project with the following command line:

`./project_scripts.py -c -gd -bd -td -j 8`
* [-c]  Clean build directory
* [-gr] Generate project files for release build
* [-bd] Start release builds
* [-tr] After the build is finished, launch unit tests for functionality
* [-j 8] During the compilation process, allow the build tool to create a pool of compilation processes equal to 8

**Note:** This command will select some default C and C++ compilers from your system. This choice is made by CMake.

##  How to: Clean the Folder from Any Local Artefacts

Clean working directory: `./project_scripts.py -c`

The project can be built with vectorized register support for the two most popular CPU compute architectures:
* `AArch64`
* `x86_64`

**Note:** Once you build a project for `ARM/AArch64` the building with `AVX` and `SSE2` will be automatically turned off
with logic from `./burt/CMakeLists.txt`. In this context, it does not make sense.

## How to: Build Debug and Release Implementations

The following two commands separately build debug and release versions. They will utilize only 1 CPU core.

```bash
./project_scripts.py -gd -bd
./project_scripts.py -gr -br
```

##  How to: Improve Build Time with Ninja

`./project_scripts.py -c -un -gd -bd -j 48`
* [-c]  Clean build directory
* [-un] Utilize Ninja build system
* [-gd] Generate project files for debug build
* [-bd] Start debug builds
* [-j 48] During the compilation process, allow the build tool to create a pool of compilation processes equal to 48

##  How to: Improve Build Time with (Jumbo) Unity Build

`./project_scripts.py -c -ub -gr -br -j 48`
* [-c]  Clean build directory
* [-ub] Use Unity build to speed up compilation time. It is a compile optimization technique where multiple source files are combined into a single large source file before compilation.
* [-gr] Generate project files for release build
* [-br] Start release builds
* [-j 48] During the compilation process, allow the build tool to create a pool of compilation processes equal to 48

##  How to: Build with custom C and C++ Compilers

The environment variables `CXX` and `CC` are commonly used in various build systems, including GNU Make, CMake, Visual Studio, etc.
They specify the C++ and C compilers used, respectively. Typically, it happens in all Operating Systems.

```bash
export CXX="/usr/bin/clang++-17"
export CC="/usr/bin/clang-17"
./project_scripts.py -c -gd -bd -j 48
```

* [-c]  Clean build directory
* [-gd] Generate project files for release build
* [-bd] Start debug build
* [-j 48] During the compilation process allow the build tool to create a pool of compilation processes equal to 48

##  How to: Invoke Generation of Documentation

The following command automatically invokes documentation generation:

```
./project_scripts.py -doxygen
```

##  How to: Pass extra arguments for CMake and more Subtle Control

For more subtle control over the build process, you may wish to pass several options to CMake:
```
cmake -D<OPTION NAME>=1 source-dir
```

To accomplish this, you need to provide all space-separated opinions as lines in:
`EXTRA_CMAKE_ARGS` environment variable before invoking `project_scripts.py`.


# Appendix

##  Build with Optimization Remarks

```bash
export CXX="/usr/bin/clang++-17"
export CC="/usr/bin/clang-17"
./project_scripts.py -c -ov -gd -bd -j 48
```

* [-c]  Clean build directory
* [-ov] LLVM optimization compilers remarks to study them
* [-gd] Generate project files for release build
* [-bd] Start debug build
* [-j 48] During the compilation process, allow the build tool to create a pool of compilation processes equal to 48

If your research includes reporting about passed seconds for your algorithm in your field and you need to report this number,
and you are using C, C++, Java, Fortran, Rust, Scala, Swift, or any other language supported by a set of compiler and toolchain technologies LLVM (https://llvm.org/)
then the following information can be interesting for you because it can assist you with polishing your implementation and gain extra speedup.

The LLVM Clang compiler can produce a special form of feedback information for the author of the code in the form of output files in the format `*.opt.yaml`.

Essentially, LLVM optimization compilers' remarks offer a dialog between:
(1) Person who writes algorithms in high-level languages (Higher than Assembly, but lower than Scripting Languages)
(2) Compiler that converts high-level instruction into instructions for a specific Instruction Set Architecture (ISA) of Computing Device or another machine-dependent representation.

With this tool, it's possible to obtain information that the compiler tried to optimize the source code, but it failed due to the writer violates
some subtle principles of the Language that touch on questions about code optimization.

After building the program, we suggest using one of the available tools to read the remarks. One such tool is [OptViewer](https://github.com/OfekShilon/optview2)
