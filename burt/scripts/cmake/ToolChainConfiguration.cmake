# CMake default policy (fix issue with google tests)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html
# Default value for CXX_STANDARD target property if set when a target is created.
set(CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})

# Extra compiler and linker options for specific Toolchains

#=======================================================================================================================================================
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    messageNormal("CMAKE OS: Windows")
    set(BURT_WINDOWS TRUE)
    add_definitions(-DBURT_WINDOWS)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    messageNormal("CMAKE OS: Linux")
    set(BURT_LINUX TRUE)
    add_definitions(-DBURT_LINUX)
elseif(CMAKE_SYSTEM_NAME MATCHES "Macos" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    messageNormal("CMAKE OS: macOS")
    set(BURT_MACOS TRUE)
    add_definitions(-DBURT_MACOS)
else()
    message(FATAL_ERROR "Cannot identify OS")
endif()

#=======================================================================================================================================================

# Special configuration for specific toolchains
# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

    messageNormal("CMAKE COMPILER IS MSVC")

    # No need
    #================================================================
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++${CXX_STANDARD}") # Declare used standart of the language (No need)
    #================================================================

    # List of compiler flags
    #================================================================
    # https://learn.microsoft.com/en-us/cpp/build/reference/compiler-options?view=msvc-170
    #================================================================

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")                  # Parallel compilation (https://msdn.microsoft.com/en-us/library/bb385193.aspx)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4530")              # MS: C++ exception handler used, but unwind semantics are not enabled. Specify /EHsc
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")              # MS: The POSIX name for this item is deprecated from point of view of Microsoft
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244")              # MS: The POSIX name for this item is deprecated from point of view of Microsoft
                                                                   # Regulation of compiler warning behaviour: https://docs.microsoft.com/ru-ru/cpp/build/reference/compiler-option-warning-level?view=vs-2019

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")      # MS: report an updated value for recent C++ language standards support
                                                                   # https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus?view=msvc-170

    # Replace /O2 with more aggressive Ox optimization
    # string(REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

    # Provide compiler information about used target CPU architecture
    #===============================================================================================================================
    if(SUPPORT_CPU_SSE2_128_bits)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:SSE2")
    endif()

    if(SUPPORT_CPU_AVX_256_bits)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /DINSTRSET=8")
    endif()

    if(SUPPORT_CPU_AVX_512_bits)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512 /DINSTRSET=10")
    endif()
    #===============================================================================================================================

    # Turn of RTTI (no dynamic_cast, typeid)
    #================================================================
    if (REMOVE_RTTI_SUPPORT_CPP)
        string(REPLACE "/GR" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
    endif()

    # Turn of exception support from C++
    #================================================================
    if (REMOVE_EXCEPTION_SUPPORT_CPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /D_HAS_EXCEPTIONS=0")
    endif()

    # Omit frame pointer. This option speeds function calls, because no frame pointers need to be set up and removed. It also frees one more register for general usage.
    #================================================================
    if (COMPILE_TIME_OPT_OMIT_FRAME_PTR)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oy")
    endif()

    # Turn on global program optimization (out of dated -- now use INTERPROCEDURAL_OPTIMIZATION, however it does not work properly)
    # Whole program optimization (Link time optimization)
    #================================================================
    if (LINK_TIME_OPTIMIZATION_CPP)
        set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} /GL")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG")
    endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    messageNormal("CMAKE COMPILER IS GNUCXX")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -falign-functions=64")      # Force function aligment to be within cache line size

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")             # Ignore deprecation warnings

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")             # Ignore narrowing. 
                                                                          #   "widening"  -- is when you cast e.g. 'integer' => 'double' and you increase precision
                                                                          #   "narrowing" -- is when you cast e.g. 'double'  => 'integer' and you lose precision
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")                   #  "append pthread" -- as compiler flag

    if (OPT_CODE_COVERAGE_GCOV_IS_ON)
        # https://man7.org/linux/man-pages/man1/gcc.1.html
        #
        #   -fprofile-arcs -- During execution the program records how many times each branch and call is executed and how many times it is taken or returns
        #   -ftest-coverage -- Produce a notes file that the gcov code-coverage utility can use to show program coverage
        #
        # From GCC documentation:
        #   This option (--coverage) is used to compile and link code instrumented for coverage analysis.
        #   The option is a synonym for -fprofile-arcs -ftest-coverage (when compiling) and -lgcov (when linking).
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")

        message(STATUS "Code coverage via gcov have been activated...")
    endif()

    # Provide compiler information about used target CPU architecture
    if (SUPPORT_CPU_SSE2_128_bits OR SUPPORT_CPU_AVX_256_bits OR SUPPORT_CPU_AVX_512_bits OR SUPPORT_CPU_CPP_TS_V2_SIMD)
        #===============================================================================================================================
        # Automatic detection of current processors' features 
        # march=native --- generate code for at compilation time by determining the processor type of the compiling machine.
        # mtune=native --- tune to cpu-type everything applicable about the generated code
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
        #===============================================================================================================================
    else()
        # Use specific target architecture
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a")
    endif()

    # Explicitly mention Fused-Mutiply-Add 
    if (SUPPORT_CPU_FMA_EXT)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    endif()

    # Turn of RTTI (no dynamic_cast, typeid)
    if (REMOVE_RTTI_SUPPORT_CPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
    endif()

    # Turn of exception support from C++
    if (REMOVE_EXCEPTION_SUPPORT_CPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
    endif()

    # Turn on global program optimization. 
    #    INTERPROCEDURAL_OPTIMIZATION unfortunately is buggy: https://gitlab.kitware.com/cmake/cmake/-/issues/23136
    if (LINK_TIME_OPTIMIZATION_CPP)
        set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -flto")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
    endif()

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pthread")

    if (BURT_USE_STATIC_CRT)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
    endif()

    # Omit frame pointer. This option speeds function calls, because no frame pointers need to be set up and removed. It also frees one more register for general usage.
    if (COMPILE_TIME_OPT_OMIT_FRAME_PTR)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer")
    endif()

    # Problems with Unity build for Linux
    #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
    #set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")

    # Turn on all warnings
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

    # Disable ABI warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi")


elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    messageNormal("CMAKE COMPILER IS CLANG")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -falign-functions=64")          # Force function aligment to be within cache line size

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")             # Ignore deprecation warnings

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")             # Ignore narrowing. 
                                                                          # "widening"  -- is when you cast e.g. 'integer' => 'double' and you increase precision
                                                                          # "narrowing" -- is when you cast e.g. 'double'  => 'integer' and you lose precision
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")                   # "append pthread" -- as compiler flag

    if (OPT_CODE_COVERAGE_GCOV_IS_ON)
        # https://man7.org/linux/man-pages/man1/gcc.1.html
        #
        #   -fprofile-arcs  -- During execution the program records how many times each branch and call is executed and how many times it is taken or returns
        #   -ftest-coverage -- Produce a notes file that the gcov code-coverage utility can use to show program coverage
        #
        # From GCC documentation:
        #   This option (--coverage) is used to compile and link code instrumented for coverage analysis.
        #   The option is a synonym for -fprofile-arcs -ftest-coverage (when compiling) and -lgcov (when linking).
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")

        message(STATUS "Code coverage via gcov have been activated...")
    endif()

    # Provide compiler information about used target CPU architecture
    if (SUPPORT_CPU_SSE2_128_bits OR SUPPORT_CPU_AVX_256_bits OR SUPPORT_CPU_AVX_512_bits OR SUPPORT_CPU_CPP_TS_V2_SIMD)
        #===============================================================================================================================
        # Automatic detection of current processors' features 
        # march=native --- generate code for at compilation time by determining the processor type of the compiling machine.
        # mtune=native --- tune to cpu-type everything applicable about the generated code
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native") # for x86_64
        #===============================================================================================================================
    endif()

    if (SUPPORT_CPU_CPP_TS_V2_SIMD)
	#===============================================================================================================================
        # https://en.cppreference.com/w/cpp/experimental
        # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fexperimental-library
        #===============================================================================================================================
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexperimental-library")
        #===============================================================================================================================
    endif()


    # Explicitly mention Fused-Mutiply-Add 
    if (SUPPORT_CPU_FMA_EXT)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    endif()

    # Turn of RTTI (no dynamic_cast, typeid)
    if (REMOVE_RTTI_SUPPORT_CPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
    endif()

    # Turn of exception support from C++
    if (REMOVE_EXCEPTION_SUPPORT_CPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
    endif()

    # Turn on global program optimization.
    # INTERPROCEDURAL_OPTIMIZATION unfortunately is buggy: https://gitlab.kitware.com/cmake/cmake/-/issues/23136
    if (LINK_TIME_OPTIMIZATION_CPP)
        set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -flto")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
    endif()

    if (BURT_USE_STATIC_CRT)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
    endif()

    if(BURT_LLVM_OPT_VIEWER)
        # Enable optimization remarks during compilation and write them to a separate file.
        # Viewer: https://github.com/OfekShilon/optview2
        # Step-1: Add compiler flags
        # Step-2: For safe let's add to linker as well
        # Step-3: Turn off whole program optimization
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsave-optimization-record")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsave-optimization-record")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsave-optimization-record")
    endif()

    # Omit frame pointer. This option speeds function calls, because no frame pointers need to be set up and removed. It also frees one more register for general usage.
    if (COMPILE_TIME_OPT_OMIT_FRAME_PTR)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer")
    endif()
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexperimental-library")
else()
    message(FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
endif()


# If we build libraries or wrappers it's more safer to generate Position-Independet-Code (PIC)
if (BURT_BUILD_SHARED_LIBRARIES OR BURT_SWIG_INTERFACE_GENERATOR)
    set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()


#=======================================================================================================================================================
# FAST REFERENCE FOR CMAKE VARIABLES
#=======================================================================================================================================================
# CMAKE_CXX_FLAGS -- compilation flags for compiling CXX (C++) files.
# CMAKE_C_FLAGS   -- flags for C compiler
# LDFLAGS         -- flags for linker
#=======================================================================================================================================================

#=======================================================================================================================================================
# COMPLETELY FORCE DEBUG
#=======================================================================================================================================================
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fno-inline")

#=======================================================================================================================================================
# Extra Flags for CLion
#=======================================================================================================================================================
#SET(CMAKE_C_FLAGS_DEBUG "-D_DEBUG")
#SET(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG")
