# BurTorch: Extra Tools for Debugging

----

The goal of this document is to describe how to prepare the extra tools that you may want to use to experiment with this project,
or carry any follow-up work.

----

# Integrated Development Environment (IDE): Alternatives

Project files are presented in [CMake](https://cmake.org/) CMakeLists.txt format.

## IDE for Windows OS

In Windows OS Family we recommend installing [Microsoft Visual Studio 2022](https://visualstudio.microsoft.com/).

Of course, you may install  [CLion](https://www.jetbrains.com/clion/) and/or [QtCreator](https://www.qt.io/product/development-tools).

Most profiling tools and reach debugging under Windows OS are available under [Visual Studio](https://visualstudio.microsoft.com/).


## IDE for macOS and Linux

One way to edit source code is via [QtCreator](https://www.qt.io/product/development-tools) which is distributed with [Qt SDK](https://www.qt.io/).

```bash
brew install qt5
brew install --cask qt-creator
```

Another popular IDE these days for Linux and macOS is [CLion](https://www.jetbrains.com/clion/). This IDE can natively open and work CMake project files similar to [QtCreator](https://www.qt.io/product/development-tools).

# GCC Alternative in macOS/Linux: CLang/LLVM Toolchain

Clang is a front-end tool for several compiled programming languages for the LLVM framework. LLVM framework performs optimizations and code generation for target compute architecture and Operating System.
The goal of the Clang/LLVM is to be a replacement for the GNU Compiler Collection (GCC). Compilers can have a big impact on performance, therefor you may wish to try using LLVM for this project.

## How To Install CLang/LLVM toolchain in Windows

This ecosystem is natively supported by Visual Studio 2019 v16.2, according to https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170.

## How To Install CLang/LLVM toolchain in macOS
```bash
brew install llvm@17
```


## How To Install CLang/LLVM toolchain on a Linux


```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh

# Install specific LLVM version
sudo ./llvm.sh 17

echo "Available Clang Versions:"; ls -1 /usr/bin/clang++-*;
```

Please install the specific version before building the project. Posix OS specifies with CXX and CC environment variables paths to C++ and C compilers:

```bash 
export CXX="/usr/bin/clang++-17"
export CC="/usr/bin/clang-17"
```

Please install the specific version before building the project. Posix OS specifies with CXX and CC environment variables paths to C++ and C compilers:

```bash
export CXX="/opt/homebrew/opt/llvm/bin/clang++"
export CC="/opt/homebrew/opt/llvm/bin/clang-18"
```

----

# Doxygen for Automatic Code Documentation

## Installation for Linux

For Linux, please install graphviz and doxygen  with the following commands:
```bash
sudo apt-get install graphviz doxygen
```

## Installation for Windows OS

For Windows OS, please install:
* Graphviz: https://graphviz.org/_pages/Download/Download_windows.html
* Microsoft HTML Help: https://learn.microsoft.com/en-us/previous-versions/windows/desktop/htmlhelp/microsoft-html-help-downloads
* Doxygen: https://www.doxygen.nl/download.html

## Installation for macOS

For macOS:
```bash
brew install doxygen
brew install graphviz
```

## Automatic Documentation Generation

We provide the Doxygen configuration file, which includes the generation of documentation in 3 formats:
- HTML webpages (*.html)
- Compiled Microsoft HTML Help (.chm)
- Latex (*.tex, *.pdf)

Automatic documentation generation can be achieved via:

```bash
cd ./burt/scripts/doxygen
python3 doxygen_generate.py
```

All automatically generated documentation will be located in `./burt/scripts/doxygen/generated`

To clean the folder with automatically generated documentation, please execute:

```bash
cd ./burt/scripts/doxygen
python3 doxygen_clean.py
```

Commands for Doxygen are available:
* Online in https://www.doxygen.nl/manual/commands.html
* Offline `./burt/scripts/doxygen/doxygen_manual-1.9.7.pdf`

The documentation in formats of [HTML](https://www.ietf.org/rfc/rfc1866.txt) and [Microsoft HTML Help](https://learn.microsoft.com/en-us/previous-versions/windows/desktop/htmlhelp/microsoft-html-help-1-4-sdk) is generated is the output of the process.

However, if you aim only at generating a `pdf` from [LaTeX](https://www.latex-project.org/), it's not the case. First, you need to have TeX distribution installed in your system. There are plenty of systems, but the most popular are:
- [MiKTeX](https://miktex.org/)
- [TeX Live](https://www.tug.org/texlive/)
- [MacTeX](https://www.tug.org/mactex/)
- [proTeXt](https://www.tug.org/protext/)
- [teTeX](https://www.tug.org/tetex/)

In our project, we used MikTeX v21.8.

## Possible Memory Issues with Generated Documentation PDF

If you have the installed `Latex` environment, you may generate documentation in latex format and, after that, build it as a final PDF document.

However, the project is pretty big, and the documentation is pretty involved.  We hope that you will not encounter problems in building documentation. In case you have a problem 95 percent of the time, this problem is due to various memory limits in the Latex/TeX engine. If that's the case, then it's most likely you will need to adjust seriously (maybe by x10) the default limits of various buffers and limits in your latex environment. For [MiKTeX](https://miktex.org/about) it can be accomplished by increasing various limits in the `config/texmfapp.ini` configuration file.
# Ninja Build System as replacement of GNU Make

Ninja is an open-source build system that aims to provide a fast and efficient way to build software projects.
Ninja can be used as a replacement for other build tools like [GNU Make](https://www.gnu.org/software/make/) where build times are not so fast.

## Install Ninja System for Windows OS

For Windows OS you need to install binaries in your system, then add the path to this binary into your `PATH` environment variable.
https://github.com/ninja-build/ninja/releases

## Install Ninja System for Linux

```bash
sudo apt-get install ninja-build
```

## Install Ninja System for macOS

```bash
brew install ninja
```

----

# GNU Coverage Tools

Lcov is a graphical front-end for the coverage testing tool [GNU GCC gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html).
It provides a way to collect coverage data for programs and then generate HTML-based reports that display the coverage information.
To use the underlying tool [GNU GCC gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html) binaries should be built with correct flags.
After this, binaries should be launched. Each compiled source file will have an object file in the filesystem.
After test runs, the coverage information will be stored in files with the extension `.gcda`.

## Install Lcov for Windows

It may be a problem.

## Install Lcov for Linux

```bash
sudo apt-get install lcov
```

## Manually Install Lcov for Linux (in case of having problems)

If your Linux distributive has an old version of lcov potentially you will need to install it manually:

```bash
sudo apt-get remove lcov
sudo apt-get install libjson-perl
wget http://mirrors.edge.kernel.org/ubuntu/pool/universe/l/lcov/lcov_1.16-1_all.deb
sudo dpkg -i lcov_1.16-1_all.deb
```

## Install Lcov for macOS

```bash
sudo apt-get install lcov
```

## Usage of GNU Coverage Tools

```
cd ./burt/scripts
```
To launch unit tests and generate code coverage reports please select paths for C++ and C compilers from the GNU GCC collection.
```bash
export CXX="/opt/homebrew/opt/gcc/bin/g++-14"
export CC="/opt/homebrew/opt/gcc/bin/gcc-14"
```

Next Launch:

```bash
cd ./burt/scripts/gnu_gcc_coverage
bash launch_code_coverage.sh
```

# Appendix

## System Introspection: Information about CPU

* In Windows OS
```bash
systeminfo
```

* In Linux/Posix OS

```bash
arch
nproc
lscpu
lscpu --extended
```

* In macOS
```bash
echo "Number of processors in system:"
sysctl -n hw.physicalcpu
echo "Information about system:"
sysctl -a
```

## Linux/Ubuntu Machine: Installations From Scratch

* Installation of Windows Management System
```bash
sudo apt-get install --no-install-recommends ubuntu-desktop gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal gnome-core gnome-session-flashback ubuntu-settings
```

* Installation of GCC

```bash
sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++

sudo apt-get install gcc-9
sudo apt-get install gcc-11

sudo apt-get install g++-9
sudo apt-get install g++-11

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 60
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 40

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 40

sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

* Installation of Qt IDE

```bash
sudo apt-get install build-essential mc
sudo apt-get install "^libxcb.*"
sudo apt-get install libx11-xcb-dev libglu1-mesa-dev libxrender-dev
sudo apt-get install qt5-default qttools5-dev-tools qtcreator qt4-demos qt4-doc qt4-doc-html qt5-doc qt5-doc-html qtbase5-examples qtbase5-doc-html 
```

* Installation of VNC Serer

```bash
sudo apt-get install tigervnc-standalone-server
sudo apt install net-tools
tigervncserver -localhost no  -geometry 1920x1080

# Kill the VNC server
# vnserver -kill :0
# Check Listen TCP/IP Port for the VNC server
netstat -nap | grep vnc
```

* Installation of VNC Viewer

You can download the original VNC Viewer by RealVNC: [https://www.realvnc.com/en/connect/download/viewer/](https://www.realvnc.com/en/connect/download/viewer/)


* Git Configuration

```bash
git config credential.helper store
git config --global http.sslVerify "false"
git config --global user.name "My Name"
git config --global user.email "my@mailserver.com"
git config --global color.ui auto
```
