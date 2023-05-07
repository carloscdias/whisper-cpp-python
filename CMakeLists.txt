cmake_minimum_required(VERSION 3.4...3.22)

project(whisper_cpp_python)

option(FORCE_CMAKE "Force CMake build of Python bindings" OFF)

set(FORCE_CMAKE $ENV{FORCE_CMAKE})

if (UNIX AND NOT FORCE_CMAKE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/vendor/whisper.cpp/libllama.so
        COMMAND make libwhisper.so
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/whisper.cpp
    )
    add_custom_target(
        run ALL
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/whisper.cpp/libwhisper.so
    )
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vendor/whisper.cpp/libwhisper.so
        DESTINATION whisper_cpp_python
    )
else()
    set(BUILD_SHARED_LIBS "On")
    add_subdirectory(vendor/whisper.cpp)
    install(
        TARGETS whisper
        LIBRARY DESTINATION whisper_cpp_python
        RUNTIME DESTINATION whisper_cpp_python
    )
endif()
