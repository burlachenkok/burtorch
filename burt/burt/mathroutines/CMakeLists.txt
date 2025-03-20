cmake_minimum_required(VERSION 3.12)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId} LANGUAGES CXX C)

file(GLOB_RECURSE original_src "src/*.cpp" "src/*.c" "src/*.cxx")
file(GLOB_RECURSE original_headers "include/*.h" "include/*.hpp" "include_internal/*.h" "include_internal/*.hpp")

# Optionally add vector instructions
if(original_src)
    createSourceGrouping(${original_src})
endif()

if (original_headers)
    createHeadersGrouping(${original_headers})
endif()

#============= BUILD TARGETS =================================================================
add_library(${PROJECT_NAME} STATIC ${original_src} ${original_headers})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

configureCompileFlags()
#============= BUILD TARGETS =================================================================
