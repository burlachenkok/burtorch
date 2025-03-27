cmake_minimum_required(VERSION 3.12)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId} LANGUAGES CXX C)

file(GLOB_RECURSE original_src "src/*.cpp" "src/*.c" "src/*.cxx")
set(original_headers "")
#file(GLOB_RECURSE original_headers "include/*.h" "include/*.hpp")

if(original_src)
    createSourceGrouping(${original_src})
endif()

if (original_headers)
    createHeadersGrouping(${original_headers})
endif()

add_executable(${PROJECT_NAME} ${original_src} ${original_headers})

target_link_libraries(${PROJECT_NAME} system)
target_link_libraries(${PROJECT_NAME} copylocal)
target_link_libraries(${PROJECT_NAME} fs)
target_link_libraries(${PROJECT_NAME} linalg_vectors)
target_link_libraries(${PROJECT_NAME} random)
target_link_libraries(${PROJECT_NAME} timers)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
configureCompileFlags()
