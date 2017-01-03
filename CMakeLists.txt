cmake_minimum_required (VERSION 3.0.2)
project (FFT)

set (CMAKE_CXX_STANDARD 11)

# Base project
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

set (INCLUDE_DIRS "")
foreach (headerFile ${HEADERS})
    get_filename_component(dir ${headerFile} PATH)
    list (APPEND INCLUDE_DIRS ${dir})
endforeach()
list(REMOVE_DUPLICATES INCLUDE_DIRS)

include_directories(${INCLUDE_DIRS})

add_executable (fft ${SOURCES})
