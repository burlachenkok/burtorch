/** @file
* Compiler-specific, platform-specific macroses
* @remark The most part of this has been taken from http://nadeausoftware.com/articles/2012/01/c_c_tip_how_use_compiler_predefined_macros_detect_operating_system
*/

#pragma once

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

#include <new>

/** Stringizing
* @param text some expression which will not be evaluated
* @return expression text will be presented as a a text
*/
#define BURT_STRINGIZING_NO_EVAL_EXPRESSION(text) #text

/** Stringizing
* @param text which will be checked for macro expansion, and then stringinized
* @return text expression for text with preliminary evaluation of expression
*/
#define BURT_STRINGIZING(text) BURT_STRINGIZING_NO_EVAL_EXPRESSION(text)

/** Info about OS
*/
#if defined(_AIX)
  #define BURT_OS_IBM_AIX 1 ///< Target OS is AIX
#elif defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  #include <sys/param.h>
  #if defined(BSD)
     #define BURT_OS_BSD_BASED_OS 1 ///< Target OS is some from this list: DragonFly BSD, FreeBSD, OpenBSD, NetBSD
  #endif
#endif

#if defined(__hpux)
  #define BURT_OS_HP_UX 1 ///< Target OS is Hewlett-Packard HP-UX
#elif defined(__linux__) && !defined(BSD)
  #include <unistd.h>
  #include <pthread.h>
  #include <sys/sysinfo.h>
  #include <sys/mman.h>

  #define BURT_OS_LINUX 1 ///< Target OS is some from this list: Centos, Debian, Fedora, OpenSUSE, RedHat, Ubuntu
#elif defined(__APPLE__) && defined(__MACH__)
  #include <TargetConditionals.h>
  #if TARGET_IPHONE_SIMULATOR == 1
    #define BURT_OS_IOS 1
    #define BURT_OS_IOS_SIMULATOR 1 ///< Target OS is "iOS in Xcode simulator"
  #elif TARGET_OS_IPHONE == 1
    #define BURT_OS_IOS 1
    #define BURT_OS_IOS_NOT_SIMULATOR 1 ///< Target OS is "iOS in on iPhone, iPad, etc
  #elif TARGET_OS_MAC == 1
    #define BURT_OS_OSX 1 ///< Target OS is Apple OSX
    #include <unistd.h>
    #include <pthread.h>
    #include <sys/mman.h>
  #endif
#elif defined(__CYGWIN__) && !defined(_WIN32)
    #include <winsock2.h>
    #include <WS2tcpip.h> 
    #include <windows.h>
    #include <Psapi.h>

    #define BURT_OS_CYGWIN ///< Cygwin POSIX under Windows

#elif defined(_WIN32) && defined(_WIN64)
    #include <winsock2.h>
    #include <WS2tcpip.h>
    #include <windows.h>
    #include <Psapi.h>

    // Undefine defined max, min macroses by WinAPI headers (conflicts with C++ Standart Library)
    #ifdef max 
        #undef max
    #endif

    #ifdef min 
        #undef min
    #endif

    #define BURT_OS_WINDOWS_64 1   ///< Target OS is windows x86 64 bit
    #define BURT_OS_WINDOWS    1   ///< Target OS is windows
#elif defined(_WIN32) && !defined(_WIN64)
    #include <winsock2.h>
    #include <WS2tcpip.h>
    #include <windows.h>

    // Undefine defined max, min macroses by WinAPI headers (conflicts with C++ Standart Library)

    #ifdef max 
        #undef max
    #endif

    #ifdef min 
        #undef min
    #endif

    #define BURT_OS_WINDOWS_32 1  ///< Target OS is windows x86 32 bit
    #define BURT_OS_WINDOWS    1  ///< Target OS is windows
#endif

/** Compiler-specific macroses*/
#ifdef BURT_OS_WINDOWS
  #define BURT_COMPILER_FUNCTION_NAME    __FUNCTION__

  /** Restrict means that within the scope in which such a pointer is defined, this pointer 
  * is the only way to access the object where the pointer is pointed.
  * For functions parameters in the form of pointers with restrict, it means that within the function scope,
  * pointers aand b always point to different memory locations.
  */
  #define restrict_ext __restrict

  /** The forceinline keyword overrides the cost benefit analysis for the compiler 
  * and relies on the judgment of the programmer instead. Indiscriminate use of forceinline 
  * can result in larger code with only marginal performance gains or in some cases even performance losses*/
  #define forceinline_ext __forceinline

  /** The forceinline keyword overrides the cost benefit analysis for the compiler
  * and relies on the judgment of the programmer instead. Indiscriminate use of forceinline
  * can result in larger code with only marginal performance gains or in some cases even performance losses*/
  #define force_no_inline_ext __declspec(noinline)

  /** Directive in MSVC for exporting functions from a DLL
  */
  #define SHARED_LIBRARY_EXPORT __declspec(dllexport)

  /** Directive in MSVC for importing functions from a DLL
  */
  #define SHARED_LIBRARY_IMPORT __declspec(dllimport)

#else

  #define BURT_COMPILER_FUNCTION_NAME __FUNCTION__

  /** Restrict means that within the scope in which such a pointer is defined, this pointer
  * is the only way to access the object where the pointer is pointed.
  * For functions parameters in the form of pointers with restrict, it means that within the function scope,
  * pointers "a" and "b" always point to different memory locations.
  */
  #define restrict_ext __restrict

  /** The forceinline keyword overrides the cost benefit analysis for the compiler
  * and relies on the judgment of the programmer instead. Indiscriminate use of forceinline
  * can result in larger code with only marginal performance gains or in some cases even performance losses
  * @remark https://gcc.gnu.org/onlinedocs/gcc/Inline.html
  */
  #define forceinline_ext __attribute__((always_inline)) inline

  /** The forceinline keyword overrides the cost benefit analysis for the compiler
  * and relies on the judgment of the programmer instead. Indiscriminate use of forceinline
  * can result in larger code with only marginal performance gains or in some cases even performance losses
  * @remark https://gcc.gnu.org/onlinedocs/gcc/Inline.html
  */
  #define force_no_inline_ext __attribute__((noinline)) inline

  /** Directive in GCC/CLANG for exporting functions from a DLL
  */
  #define SHARED_LIBRARY_EXPORT __attribute__ ((visibility ("default")))

  /** Directive in GCC/CLANG for importing functions from a shared library is empty.
  */
  #define SHARED_LIBRARY_IMPORT

#endif

#if defined(__clang__)
  #define BURT_COMPILER_NAME "Clang/LLVM"
  #define BURT_COMPILER_VERSION_STRING_LONG __VERSION__
  #define BURT_COMPILER_IS_CLANG 1
  #define const_func_ext
  //__attribute__((const))
#elif defined(__ICC) || defined(__INTEL_COMPILER)
  #define BURT_COMPILER_NAME "Intel ICC/ICPC"
  #define BURT_COMPILER_VERSION_STRING_LONG __VERSION__
  #define BURT_COMPILER_IS_INTEL 1
  #define const_func_ext __attribute__((const))
#elif defined(__GNUC__) || defined(__GNUG__)
  #define BURT_COMPILER_NAME "GNU GCC/G++"
  #define BURT_COMPILER_VERSION_STRING_LONG __VERSION__
  #define BURT_COMPILER_IS_GCC 1
  #define BURT_STRUCT_ALIGNMENT(BYTES) __attribute__((aligned(BYTES)))
  #define BURT_FUNCTION_NAME __PRETTY_FUNCTION__

  #if NDEBUG
    #define BURT_LOOK_LIKE_DEBUG_BUILD false
  #else
    #define BURT_LOOK_LIKE_DEBUG_BUILD true
  #endif

  /** If function:
  * (1) Does not examine (read) any values (in global state) except their arguments.
  * (2) Have no effects except the return value.
  * 
  *   __declspec(noalias)   - https://learn.microsoft.com/en-us/cpp/cpp/noalias?view=msvc-170
  * 
  * __attribute__((pure))  - https://gcc.gnu.org/onlinedocs/gcc-4.0.0/gcc/Function-Attributes.html
  *                          https://gcc.gnu.orgonlinedocs/gcc/Function-Attributes.html#Function-Attributes
  * 
  * __attribute__((pure))  - https://clang.llvm.org/docs/AttributeReference.html#function-attributes
  * 
  * https://www.ibm.com/docs/en/i/7.2?topic=attributes-pure-function-attribute
  * 
  * If function that has pointer arguments and examines the data pointed to MUST NOT be declared as const.
  *
  *
  * TODO: Update information -- ((const)) is in fact too restrictive for a correct program.
  */

  #define const_func_ext __attribute__((const))

#elif defined(__HP_cc) || defined(__HP_aCC)
  #define BURT_COMPILER_NAME "Hewlett-Packard C/aC++"
  #define BURT_COMPILER_VERSION_STRING_LONG BURT_STRINGIZING(__HP_aCC)
  #define BURT_COMPILER_IS_HP 1

  /** If function (1) Does not examine (read) any values (in global state) except their arguments; (2) Have no effects except the return value
  * If function that has pointer arguments and examines the data pointed to MUST NOT be declared as const.
  */
  #define const_func_ext __attribute__((const))
#elif defined(__IBMC__) || defined(__IBMCPP__)
  #define BURT_COMPILER_NAME "IBM XL C/C++"
  #define BURT_COMPILER_VERSION_STRING_LONG __xlc__
  #define BURT_COMPILER_IS_IBM 1

  /** If function (1) Does not examine (read) any values (in global state) except their arguments; (2) Have no effects except the return value
  * If function that has pointer arguments and examines the data pointed to MUST NOT be declared as const.
  */
  #define const_func_ext __attribute__((const))
#elif defined(_MSC_VER)

#if _MSC_VER==1933
    #define BURT_COMPILER_NAME "Visual Studio 2022 version 17.3.4"
#elif _MSC_VER==1932
    #define BURT_COMPILER_NAME "Visual Studio 2022 version 17.2.2"
#elif _MSC_VER==1930
    #define BURT_COMPILER_NAME "Visual Studio 2022 version 17.0.2"
#elif _MSC_VER==1929
    #define BURT_COMPILER_NAME "Visual Studio 2019 version 16.11.2"
#elif _MSC_VER==1900
    #define BURT_COMPILER_NAME "Microsoft Visual Studio 2015"
#elif _MSC_VER==1800
    #define BURT_COMPILER_NAME "Microsoft Visual Studio 2013"
#else
    #define BURT_COMPILER_NAME "Microsoft Visual Studio"
#endif


  #define BURT_COMPILER_VERSION_STRING_LONG BURT_STRINGIZING(_MSC_FULL_VER) "." BURT_STRINGIZING(_MSC_BUILD)
  #define BURT_COMPILER_IS_VC 1
  #define BURT_STRUCT_ALIGNMENT(BYTES) __declspec(align(BYTES))
  #define BURT_FUNCTION_NAME __FUNCSIG__

  /** Please us this if function:
  * (1) Does not examine (read) any values (in global state) except their arguments; 
  * (2) Have no effects except the return value
  * (3) Read only memory that directly pointed by pointer parameters (first-level indirections)
  * 
  * If function that has pointer arguments and examines the data pointed to MUST NOT be declared as const.
  */
  #define const_func_ext __declspec(noalias)
#elif defined(__PGI)
  #define BURT_COMPILER_NAME "Portland Group PGCC/PGCPP"
  #define BURT_COMPILER_VERSION_STRING_LONG  BURT_STRINGIZING(__PGIC__) "." BURT_STRINGIZING(__PGIC_MINOR) "." BURT_STRINGIZING(__PGIC_PATCHLEVEL__)
  #define BURT_COMPILER_IS_PGI
  #define const_func_ext __attribute__((const))
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  #define BURT_COMPILER_NAME "Oracle Solaris Studio"
  #define BURT_COMPILER_VERSION_STRING_LONG BURT_STRINGIZING(__SUNPRO_CC)
  #define BURT_COMPILER_IS_ORACLE
  #define const_func_ext __attribute__((const))
#endif

/** Processor-specific macroses*/
#if defined(__ia64) || defined(__itanium__) || defined(_M_IA64)
  #define BURT_ARCH_NAME "Itanium"
  #define BURT_ARCH_ITANIUM 1
  #define BURT_ARCH_LITTLE_ENDIAN 1
  #define BURT_ARCH_BIG_ENDIAN 0

#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
  #if defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__) || defined(__64BIT__) || defined(_LP64) || defined(__LP64__)

    #if  defined(__ALTIVEC__)
        // In AltiVec (also known as VMX) on PowerPC, the vector registers are 128 bits in length.
        #define BURT_ARCH_NAME "Power PC 64 [SIMD: AltiVec-128bits]"
    #elif defined(__VSX__)
        // VSX (Vector-Scalar Extension) on PowerPC, the vector registers are also 128 bits
        #define BURT_ARCH_NAME "Power PC 64 [SIMD: VSX-128bits]"
    #elif defined(__VEC__)
        // __VEC__, which refers to the AltiVec (VMX) vector extension in PowerPC, the vector registers are also 128 bits in length.
        #define BURT_ARCH_NAME "Power PC 64 [SIMD: VMX-128bits]"
    #else
          #define BURT_ARCH_NAME "Power PC 64 [Not Vector Extensions as AltiVec/VMX/VSX]"
    #endif

    #define BURT_ARCH_POWER_PC_64 1

    #if !defined(__BYTE_ORDER__)
       #error "Compile time byter order defined in a CLang and GCC specifiy way. Maybe uncompatible with another compilers"
    #endif

    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        #define BURT_ARCH_LITTLE_ENDIAN 1
        #define BURT_ARCH_BIG_ENDIAN 0
    #else
        #define BURT_ARCH_LITTLE_ENDIAN 0
        #define BURT_ARCH_BIG_ENDIAN 1
    #endif

  #else

    #define BURT_ARCH_NAME "Power PC 32"
    #define BURT_ARCH_POWER_PC_32 1

    #if  defined(__ALTIVEC__)
        // In AltiVec (also known as VMX) on PowerPC, the vector registers are 128 bits in length.
        #define BURT_ARCH_NAME "Power PC 32 [SIMD: AltiVec-128bits]"
    #elif defined(__VSX__)
        // VSX (Vector-Scalar Extension) on PowerPC, the vector registers are also 128 bits
        #define BURT_ARCH_NAME "Power PC 32 [SIMD: VSX-128bits]"
    #elif defined(__VEC__)
        // __VEC__, which refers to the AltiVec (VMX) vector extension in PowerPC, the vector registers are also 128 bits in length.
        #define BURT_ARCH_NAME "Power PC 32 [SIMD: VMX-128bits]"
    #else
          #define BURT_ARCH_NAME "Power PC 32 [Not Vector Extensions as AltiVec/VMX/VSX]"
    #endif

    #if !defined(__BYTE_ORDER__)
       #error "Compile time byter order defined in a CLang and GCC specifiy way. Maybe uncompatible with another compilers"
    #endif

    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        #define BURT_ARCH_LITTLE_ENDIAN 1
        #define BURT_ARCH_BIG_ENDIAN 0
    #else
        #define BURT_ARCH_LITTLE_ENDIAN 0
        #define BURT_ARCH_BIG_ENDIAN 1
    #endif

#endif
#elif defined(__sparc)
  #define BURT_ARCH_NAME "Sparc"
  #define BURT_ARCH_SPARC 1
  #define BURT_ARCH_LITTLE_ENDIAN 0
  #define BURT_ARCH_BIG_ENDIAN 1

#elif defined(__x86_64__) || defined(_M_X64)
  #define BURT_ARCH_NAME "AMD, Intel x86 64 bit"
  #define BURT_ARCH_X86_64BIT 1
  #define BURT_ARCH_LITTLE_ENDIAN 1
  #define BURT_ARCH_BIG_ENDIAN 0

#elif defined(__i386) || defined(_M_IX86)
  #define BURT_ARCH_NAME "AMD, Intel x86 32 bit"
  #define BURT_ARCH_X86_32BIT 1
  #define BURT_ARCH_LITTLE_ENDIAN 1
  #define BURT_ARCH_BIG_ENDIAN 0

#elif defined(__aarch64__)
  #if __ARM_NEON
    #define BURT_ARCH_NAME "AArch 64/ARMv" BURT_STRINGIZING(__ARM_ARCH) " [with ARM Neon]"
  #else
    #define BURT_ARCH_NAME "AArch 64/ARMv" BURT_STRINGIZING(__ARM_ARCH) " [no ARM Neon]"
  #endif

  #define BURT_ARCH_ARM 1

  #if !defined(__BYTE_ORDER__)
    #error "Compile time byter order defined in a CLang and GCC specifiy way. Maybe uncompatible with another compilers"
  #endif

  // ARM CPU CAN OPERATE IN BOTH: LITTLE AND BIG ENDIAN MODES
  #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      #define BURT_ARCH_LITTLE_ENDIAN 1
      #define BURT_ARCH_BIG_ENDIAN 0
  #else
    #define BURT_ARCH_LITTLE_ENDIAN 0
    #define BURT_ARCH_BIG_ENDIAN 1
  #endif

#elif defined(__arm__)
  #if __ARM_NEON
    #define BURT_ARCH_NAME "AArch 32/ARMv" BURT_STRINGIZING(__ARM_ARCH) " [with ARM Neon]"
  #else
    #define BURT_ARCH_NAME "AArch 32/ARMv" BURT_STRINGIZING(__ARM_ARCH) " [no ARM Neon]"
  #endif

  #define BURT_ARCH_ARM 1

  #if !defined(__BYTE_ORDER__)
    #error "Compile time byter order defined in a CLang and GCC specifiy way. Maybe uncompatible with another compilers"
  #endif

  // ARM CPU CAN OPERATE IN BOTH: LITTLE AND BIG ENDIAN MODES
  #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    #define BURT_ARCH_LITTLE_ENDIAN 1
    #define BURT_ARCH_BIG_ENDIAN 0
  #else
    #define BURT_ARCH_LITTLE_ENDIAN 0
    #define BURT_ARCH_BIG_ENDIAN 1
  #endif

#elif defined(__riscv)
  // https://github.com/riscv-non-isa/riscv-c-api-doc/blob/main/src/c-api.adoc
  #if __riscv_vector || __riscv_v
    #define BURT_ARCH_NAME "RISC-V RV(" BURT_STRINGIZING(__riscv_xlen) ")" " [With RISC-V Vector Extension RVV]"
  #else
    #define BURT_ARCH_NAME "RISC-V RV(" BURT_STRINGIZING(__riscv_xlen) ")" " [No Vector Extension RVV]"
  #endif

  #define BURT_ARCH_RISCV 1
  #define BURT_SIMD_LEN_IN_BITS __riscv_v_min_vlen

  #if !defined(__BYTE_ORDER__)
     #error "Compile time byter order defined in a CLang and GCC specifiy way. Maybe uncompatible with another compilers"
  #endif

  #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      #define BURT_ARCH_LITTLE_ENDIAN 1
      #define BURT_ARCH_BIG_ENDIAN 0
  #else
      #define BURT_ARCH_LITTLE_ENDIAN 0
      #define BURT_ARCH_BIG_ENDIAN 1
  #endif

#elif defined(__s390__) || defined(__s390x__) || defined(__370__) || defined(__zarch__)
  #ifdef __s390x__
    #define BURT_ARCH_NAME "IBM SystemZ / 390 (64-bit)"
  #elif defined(__s390__)
    #define BURT_ARCH_NAME "IBM SystemZ / 390 (32-bit)"
  #elif defined(__370__)
    #define BURT_ARCH_NAME "IBM System / 370 (32-bit)"
  #else
    BURT_ARCH_NAME "IBM System (Unknown)"
  #endif

  #define BURT_ARCH_IBM 1

  #if !defined(__BYTE_ORDER__)
     #error "Compile time byter order defined in a CLang and GCC specifiy way. Maybe uncompatible with another compilers"
  #endif

  #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      #define BURT_ARCH_LITTLE_ENDIAN 1
      #define BURT_ARCH_BIG_ENDIAN 0
  #else
      #define BURT_ARCH_LITTLE_ENDIAN 0
      #define BURT_ARCH_BIG_ENDIAN 1
  #endif

#else
  #error Unknown Architecture. Please check PlatformSpecificMacroses.h
#endif

/** Cache line - is size of bytes real fetching by processor from memory when it need even one byte.
* 1. Also for optimization purposes group similar access data to cache line. If write one byte to memory then it will be written full cache line.
* 2. Linear array traversals very cache-friendly.
* 3. Cache Coherency was supported automatically by CPU
* 4. False sharing - Different cores concurrently access same cache line and at least one is a writer
*
* Empty bytes padding for forming cache line.
* @param Name name of the expansion byte array
* @param BytesSoFar size in bytes of all structure fields before current place. Actually dummy array will be placed after this part.
* @param CacheLineSize size of the cache line for compute device
* @return macro expansion with dummy byte array
*/
#define BURT_CACHE_LINE_PAD(Name, BytesSoFar, CacheLineSize) unsigned char Name[(CacheLineSize) - ((BytesSoFar) % (CacheLineSize))]

/** Size of struct element in bytes
* @param structure_name name of the structure. explicit declaration at moment of call T1_SIZEOF_ELEMENT
* @param member_name name of the member
* @return sizeof element in bytes
*/
#define BURT_SIZEOF_ELEMENT(structure_name, member_name) sizeof(((structure_name*)0)->member_name)

/** Element offset in bytes from start of structure. Start of the structure equal to address of the first structure element
* @param structure_name name of the structure. explicit declaration at moment of call T1_SIZEOF_ELEMENT
* @param member_name name of the member
* @return offset of element in bytes from begin of the structure.
*/
#define BURT_OFFSET_OF_ELEMENT(structure_name, member_name) ( (size_t)(& (((structure_name*)0)->member_name)) )

// Requires C++20
#if __has_include(<experimental/source_location>)
    #include <experimental/source_location>
#elif __has_include(<source_location>)
    #include <source_location>
#endif

/** Trace calls to stdout
* @filename current compile unit file name (from __FILE__)
* @line current compile unit line in file (from __LINE__)
* @expression some expression which will not be evaluated
*/
#define BURT_TRACE(expression) do { printf("FILE: %s\nLINE:%i\nFUNCTION:%s\nEXPR: %s\n", \
                                    std::source_location::current().file_name(),\
                                    std::source_location::current().line(), \
                                    std::source_location::current().function_name(), \
                                    #expression);\
                                    expression; } while(0)

/** Assert used for checking after various C++ optimization tricks
* @expression expression which should be non-zero for purpose of success check.
*/
// #define optimization_assert(expression) assert(expression)
#define optimization_assert(expression)

#if 1
    #if defined(_MSC_VER)
        #define burt_unreahable() __assume(false)
    #elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
        #define burt_unreahable() __builtin_unreachable()
    #else
        #define burt_unreahable()
    #endif
#else
    #define burt_unreahable()
#endif

#if 1
    #define burt_assert(x) do { assert(x); } while(0)
#else
    #define burt_assert(x) ;
#endif
