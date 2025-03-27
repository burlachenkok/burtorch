# BurTorch: Minimal Environment

----

The goal of this document is to describe how to prepare the main build and runtime environment for building our project and provide needed runtimes for alternative solutions in Windows, Linux, and macOS.

----

# Prepare to Build

## If you are Working under Windows OS

1. Install Visual Studio 2022 (or newer) in your Windows OS. To install Microsoft Visual Studio please visit the Microsoft website [https://visualstudio.microsoft.com/vs/](https://visualstudio.microsoft.com/vs/) and follow Microsoft instructions.

2. Install/Update CMake to version 3.12 or higher from https://cmake.org/download/

3. Download LibTorch from `https://download.pytorch.org/libtorch/cpu/`. 

    - For Windows OS use this verions: `https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.1%2Bcpu.zip`
    - For Linux OS use this version: `https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.1%2Bcpu.zip`

## If you are Working under Linux OS

1. Install GCC-11. See GCC release: https://gcc.gnu.org/releases.html.

    ```bash
    sudo apt-get install gcc-11 g++-11
    ```

    If your Linux distribution does not have Advanced Package Tool (Apt) Package Manager, please use a similar tool distributed with your Operating System.

2. To get recent versions of CMake under at least Ubuntu Linux distributive in 2024 is not so easy. To get it you will need to do some manual work:

    * If your CPU has ARM/AArch64 architecture execute the following:

        ```bash
        sudo apt remove cmake
        # For AArch64 CPUs
        fname=cmake-3.27.0-rc5-linux-aarch64.sh
        wget https://github.com/Kitware/CMake/releases/download/v3.27.0-rc5/${fname}
        sudo cp $fname /opt/
        cd /opt
        sudo bash $fname
        # For AArch64 CPUs
        sudo ln -s /opt/cmake-3.27.0-rc5-linux-aarch64/bin/cmake /usr/local/bin/
        ```

    * If your CPU has x86-64 architecture execute the following:


        ```bash
        sudo apt remove cmake
        # For Intel/AMD CPUs
        fname=cmake-3.27.0-rc5-linux-x86_64.sh
        wget https://github.com/Kitware/CMake/releases/download/v3.27.0-rc5/${fname}
        sudo cp $fname /opt/
        cd /opt
        sudo bash $fname
        # For Intel/AMD CPUs
        sudo ln -s /opt/cmake-3.27.0-rc5-linux-x86_64/bin/cmake /usr/local/bin/
        ```

## If you are Working under macOS

1. Install GCC-11 for example from [brew](https://brew.sh/). See GCC release: https://gcc.gnu.org/releases.html.

    ```bash
    brew install gcc@11
    ```

2. Install the recent version of CMake in macOS. For our project, CMake 3.12 is enough:

    ```bash
    brew install cmake
    ```

## Installations for Benchmarking

### Prepare Python and Conda Environment

BurTorch do not use Python at all, however baselines use it. If you don't have the Conda package and environment manager you can install them via the following steps for Linux OS:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="${PATH}:~/miniconda3/bin"
~/miniconda3/bin/conda init bash && source ~/.bashrc && conda config --set auto_activate_base false
```

For refresh conda commands, you can look into the official [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

Examples which are using uses Python 3.9 and you can install it in conda via:
```
conda create --name my python=3.9
conda activate my
```
