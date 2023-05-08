cmake_minimum_required(VERSION 3.4...3.22)

project(whisper_cpp_python)

option(FORCE_CMAKE "Force CMake build of Python bindings" OFF)

add_subdirectory(vendor/whisper.cpp)

option(${BUILD_SHARED_LIBS} "ON")

install(DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib/ DESTINATION ${CMAKE_INSTALL_PREFIX}/whisper_cpp_python/
        FILES_MATCHING PATTERN "*")

