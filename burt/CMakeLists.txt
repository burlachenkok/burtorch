cmake_minimum_required(VERSION 3.12)

#===================================================================================================================
# https://cmake.org/cmake/help/latest/prop_gbl/USE_FOLDERS.html
# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#===================================================================================================================
# START OF CONFIGUTATION
#===================================================================================================================

# Add NN experiments
option(BURT_INCLUDE_NN_EXPERIMENTS        "Turn on NN experiemtns" ON)

# Debug Building Itself
option(BURT_TRACE_DEBUG_PROJECT_BUILDING  "Turn on debug information about project building" OFF)

# Build configutation: Build configuration for C++
option(BURT_USE_STATIC_CRT                "Use static CRT version" OFF)
option(REMOVE_RTTI_SUPPORT_CPP            "Remove RTTI - run time type information from C++" ON)
option(REMOVE_EXCEPTION_SUPPORT_CPP       "Remove exception support from C++" ON)
option(LINK_TIME_OPTIMIZATION_CPP         "Link time Optimization or Whole program Optimization" ON)
option(COMPILE_TIME_OPTIMIZATION_USE_PCH  "Use precompiled headers to improve compile time" OFF)
option(BURT_VERBOSE_BUILD                 "Turn on all verbose messages during build" ON)

# If use Whole Program Optimization the compiler will have more information about the entire program, and it may decide to omit frame pointers based on its analysis.
# Generally don't need to explicitly enable or disable frame pointer omission.
option(COMPILE_TIME_OPT_OMIT_FRAME_PTR    "Compile time Optimization with omitting frame pointer" OFF)

# Build configutation: What to include into build
option(BURT_INCLUDE_UNITTESTS   "Include unittests into build" ON)
option(BURT_INCLUDE_UTILS       "Include utils into build" ON)

option(SUPPORT_CPU_SSE2_128_bits   "Target CPU support SSE2 instruction set with 128 bits registers"   OFF)
option(SUPPORT_CPU_AVX_256_bits    "Target CPU support AVX2 instruction set with 256 bits registers"   ON )
option(SUPPORT_CPU_AVX_512_bits    "Target CPU support AVX512 instruction set with 512 bits registers" OFF)
option(SUPPORT_CPU_CPP_TS_V2_SIMD  "Target compiler support C++ SIMD Extension. Use it." OFF)

option(SUPPORT_CPU_FMA_EXT         "Target CPU support x86/FMA3 instruction" OFF)
option(SUPPORT_CPU_LOAD_STORE_PART "Use store and load partial functionality for SIMD code instead of usual CPU code" OFF)
#===================================================================================================================

# Build configutation: Implementation Variants
#====================================================================================================================================================================
option(BURT_INCLUDE_VECTORIZED_CPU_IMP_VECS        "Include into the build vectorized implementation of dense vectors" ON)
option(BURT_INCLUDE_VECTORIZED_CPU_IMP_MATS        "Include into the build vectorized implementation of dense matrices" ON)
#===================================================================================================================

# Build configutation:  Instrumentation
#===================================================================================================================
option(BURT_EXTRA_DEBUG                            "Turn on debugging and tracking various quantities." OFF)
option(OPT_CODE_COVERAGE_GCOV_IS_ON                "Use GNU gcov code coveraging tool" OFF)
option(BURT_LLVM_OPT_VIEWER                        "Use LLVM optimization remarks. Please turn off link time optimization manually." OFF)

# LLVM Optimization Remarks - Helping the Compiler Generate Better Code - Ofek Shilon - CppCon 2022 06 January 2022: https://www.youtube.com/watch?v=qmEsx4MbKoc&t=2407s
# LLVM Optimizer Viewer project: https://github.com/OfekShilon/optview2
# How to use: 
#  Step-1: Turn option ON and build project with CLANG
#  Step-2: python3 opt-viewer.py -j10 --output-dir ./out --source-dir ./../project_name/build_release/
#
# Remark: The tool provides some way of dialog with the source writer and compiler.
#===================================================================================================================
# END OF CONFIGUTATION
#===================================================================================================================

if(BURT_TRACE_DEBUG_PROJECT_BUILDING)
    # Turn on for debugging purposes
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

project("burt" LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake)
include(HelpFunctions)
include(ToolChainConfiguration)

set(BURT_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

messageNormal("============================================================")
messageNormal("PROJECT_ROOT: " ${BURT_PROJECT_ROOT})
messageNormal("OUTPUT BINARY DIRECTORY: " ${CMAKE_CURRENT_BINARY_DIR})

messageNormal("BUILD TYPE: " ${CMAKE_BUILD_TYPE})
messageNormal("PROJECT NAME: " ${PROJECT_NAME})

messageNormal("CMAKE VERSION: " ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
messageNormal("CMAKE VERSION: " ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
messageNormal("CMAKE DEFAULT MODULE PATH (include, find_package): " ${CMAKE_MODULE_PATH})
messageNormal("CMAKE INSTALLATION DIRECTORY: " ${CMAKE_ROOT})

messageNormal("C++ COMPILER: " ${CMAKE_CXX_COMPILER})
messageNormal("C COMPILER: " ${CMAKE_C_COMPILER})
messageNormal("C++ STANDARD: " ${CXX_STANDARD})
messageNormal("TRACE BUILD COMMANDS: " ${BURT_TRACE_DEBUG_PROJECT_BUILDING})
messageNormal("============================================================")

#==============================================================================================================
# Turn off x86 extension for ARM exlpicitly they are not supported in ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch")
    messageNormal("Target CPU Architecture: ${CMAKE_SYSTEM_PROCESSOR}. Turning off x86_64 extensions")
    set(SUPPORT_CPU_SSE2_128_bits OFF)
    set(SUPPORT_CPU_AVX_256_bits OFF)
    set(SUPPORT_CPU_AVX_512_bits OFF)
else()
    messageNormal("Target Architecture Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Setup variable based on build configuration
#==============================================================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BURT_DEBUG_BUILD 1)
else()
    set(BURT_DEBUG_BUILD 0)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(BURT_RELEASE_BUILD 1)
else()
    set(BURT_RELEASE_BUILD 0)
endif()

# Based on type of C++ runtime selected for project specify option for build unittests and protobuf library
#==============================================================================================================
if(BURT_USE_STATIC_CRT)
    set(gtest_force_shared_crt 0)
else()
    set(gtest_force_shared_crt 1)
endif()  

if (BURT_INCLUDE_UNITTESTS)
    add_subdirectory(burt/3rdparty/gtest-1.12.1)
endif()


# Print extra information
#==============================================================================================================
printInfo()


# Add all subprojects
#==============================================================================================================
add_subdirectory(burt/timers)
add_subdirectory(burt/fs)
add_subdirectory(burt/random)
add_subdirectory(burt/copylocal)
add_subdirectory(burt/linalg_vectors) 
add_subdirectory(burt/mathroutines)
add_subdirectory(burt/system) 

# Add unittests
if (BURT_INCLUDE_UNITTESTS)
    add_subdirectory(bin_tests)
endif()

# Add NN experiments
if (BURT_INCLUDE_NN_EXPERIMENTS)
    add_subdirectory(bin_tiny_example)
    add_subdirectory(bin_small_example)
    add_subdirectory(bin_small_example_loading)
    add_subdirectory(bin_small_example_saving)
    add_subdirectory(bin_makemore_nlp_example)
    add_subdirectory(bin_gpt_example)
endif()

add_subdirectory(burtcore)
#==============================================================================================================

