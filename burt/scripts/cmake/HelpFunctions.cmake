# Message wrapper for verbose message
#======================================================================
function(messageVerbose)
  if(BURT_VERBOSE_BUILD)
    message(STATUS "project-" ${PROJECT_NAME} ": " ${ARGV})
  endif()
endfunction()
#======================================================================

# Message wrapper for normal message
#======================================================================
function(messageNormal)
  message(STATUS "project-" ${PROJECT_NAME} ": " ${ARGV})
endfunction()
#======================================================================

# Message wrapper for error message
#======================================================================
function(messageFatal)
  message(FATAL_ERROR "project-" ${PROJECT_NAME} ": " ${ARGV})
endfunction()
#======================================================================


# Print info about system
#======================================================================
function(printInfo)
    messageNormal("System: " ${CMAKE_SYSTEM_NAME} " " ${CMAKE_SYSTEM_VERSION})
    messageNormal("Processor of host system: " ${CMAKE_HOST_SYSTEM_PROCESSOR})
    messageNormal("Processor of target system: " ${CMAKE_SYSTEM_PROCESSOR})
    messageNormal("CMake generator: " ${CMAKE_GENERATOR})
    messageNormal("Binary tree: " ${PROJECT_BINARY_DIR})
    messageNormal("Source tree: " ${PROJECT_SOURCE_DIR})
    messageNormal("Build type: " ${CMAKE_BUILD_TYPE})
    messageNormal("Current project name: " ${PROJECT_NAME})
endfunction()
#======================================================================

# Add extra private definition for C preprocessor for current project
#======================================================================
function(addDefinition option_name)
    if (${option_name})
        target_compile_definitions(${PROJECT_NAME} PRIVATE ${option_name}=1)
    else(${option_name})
        target_compile_definitions(${PROJECT_NAME} PRIVATE ${option_name}=0)
    endif()
endfunction()
#======================================================================

# Rather specific compiler flags, depend on specific Toolchain
#======================================================================
macro(configureCompileFlags)
    if(MSVC)
         # Select crt version for windows builds based on generator expression syntax: 
         # https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html
         # http://stackoverflow.com/questions/34490294/what-does-configdebugrelease-mean-in-cmake
         # http://stackoverflow.com/questions/10199904/how-can-i-set-specific-compiler-flags-for-for-a-specific-target-in-a-specific-bu
         if (BURT_USE_STATIC_CRT)
             target_compile_options(${PROJECT_NAME} PRIVATE "/MT$<$<CONFIG:Debug>:d>")
         else()
             target_compile_options(${PROJECT_NAME} PRIVATE "/MD$<$<CONFIG:Debug>:d>")
         endif()
    endif()

    addDefinition(BURT_USE_STATIC_CRT)
    addDefinition(BURT_VERBOSE_BUILD)

    addDefinition(BURT_INCLUDE_UNITTESTS)
    addDefinition(BURT_INCLUDE_UTILS)

    addDefinition(SUPPORT_CPU_SSE2_128_bits)
    addDefinition(SUPPORT_CPU_AVX_256_bits)
    addDefinition(SUPPORT_CPU_AVX_512_bits)
    addDefinition(SUPPORT_CPU_CPP_TS_V2_SIMD)

    addDefinition(SUPPORT_CPU_FMA_EXT)
    addDefinition(SUPPORT_CPU_LOAD_STORE_PART)

    addDefinition(BURT_INCLUDE_VECTORIZED_CPU_IMP_VECS)
    addDefinition(BURT_INCLUDE_VECTORIZED_CPU_IMP_MATS)

    addDefinition(BURT_EXTRA_DEBUG)

    addDefinition(BURT_DEBUG_BUILD)
    addDefinition(BURT_RELEASE_BUILD)

    # Add employed TCP/IP MTU SIZE
    target_compile_definitions(${PROJECT_NAME} PRIVATE D_OPT_NETWORK_MTU_SIZE=1500)

    # Target OS is MS Windows
    if(BURT_WINDOWS)
        target_link_libraries(${PROJECT_NAME} Ws2_32.lib)
    endif()

    # Add root path
    target_include_directories(${PROJECT_NAME} PUBLIC ${BURT_PROJECT_ROOT})

    # Use precompiled headers
    if(COMPILE_TIME_OPTIMIZATION_USE_PCH)
        if (original_headers)
            target_precompile_headers(${PROJECT_NAME} PRIVATE ${original_headers})
        endif()
    endif()

endmacro()
#======================================================================


# Create Source grouping in Visual Studio
#======================================================================
macro(createSourceGrouping)
    if(MSVC)
        foreach(f ${ARGV})
            file(RELATIVE_PATH SRCGR ${CMAKE_CURRENT_SOURCE_DIR} ${f})
            set(SRCGR "Sources/${SRCGR}")
            string(REGEX REPLACE "(.*)(/[^/]*)$" "\\1" SRCGR ${SRCGR})
            string(REPLACE / \\ SRCGR ${SRCGR})
            source_group("${SRCGR}" FILES ${f})
        endforeach()
    endif()
endmacro()
#======================================================================

# Create Headers grouping in Visual Studio (taken somewhere from stackoverflow)
#======================================================================
macro(createHeadersGrouping)
    if(MSVC)
        foreach(f ${ARGV})
            file(RELATIVE_PATH SRCGR ${CMAKE_CURRENT_SOURCE_DIR} ${f})
            set(SRCGR "Headers/${SRCGR}")
            string(REGEX REPLACE "(.*)(/[^/]*)$" "\\1" SRCGR ${SRCGR})
            string(REPLACE / \\ SRCGR ${SRCGR})
            source_group("${SRCGR}" FILES ${f})
        endforeach()
    endif()
endmacro()
#======================================================================

macro(configureCompileFlagsForSharedLibraryBuild)
    if(MSVC)
        # All functions are not exported by default -- nothing to do
    else()
        # This option effectively hides all symbols by default
        # useful for creating shared libraries that do not expose their internal implementation details.
        target_compile_options(${PROJECT_NAME} PRIVATE -fvisibility=hidden)
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_SHARED_LIBRARY=1)
endmacro()
#======================================================================
