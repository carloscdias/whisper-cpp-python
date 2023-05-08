cmake_minimum_required(VERSION 3.4...3.22)

project(whisper_cpp_python)

option(FORCE_CMAKE "Force CMake build of Python bindings" OFF)

add_subdirectory(vendor/whisper.cpp)

