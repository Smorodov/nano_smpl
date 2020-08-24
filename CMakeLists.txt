cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(nano_smpl)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package(OpenMP)
# enable OpenMP if compiler supports it
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_C_FLAGS "/EHsc")
set(CMAKE_CXX_FLAGS "/EHsc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
add_subdirectory(zlib)
add_subdirectory(freeglut-3.0.0)

SET(CMAKE_CXX_FLAGS_DEBUG  "-MDd -Zi -Ob0 -Od -RTC1  -_Debug") 

FIND_PACKAGE(OpenGL)
INCLUDE_DIRECTORIES("glm")
INCLUDE_DIRECTORIES("freeglut-3.0.0/include")
IF ( MSVC )
  SET(M_LIB)
ELSE ( MSVC )
  find_library(M_LIB m)
ENDIF ( MSVC )
INCLUDE_DIRECTORIES("glew/include")
INCLUDE_DIRECTORIES("glm")
include_directories("${CMAKE_SOURCE_DIR}/zlib")
INCLUDE_DIRECTORIES("numpyLib")
include_directories("${CMAKE_SOURCE_DIR}")
set(SOURCES
    main.cpp
    smpl.cpp
    glew/src/glew.c
    numpyLib/cnpy.cpp
    )

set(HEADERS
    smpl.h
    )
SOURCE_GROUP("Source Files" FILES ${SOURCES})
SOURCE_GROUP("Header Files" FILES ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set_target_properties( ${PROJECT_NAME} PROPERTIES  VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} zlib freeglut )

file(COPY ${PROJECT_SOURCE_DIR}/scripts/model DESTINATION ${CMAKE_SOURCE_DIR}/build/Debug)
file(COPY ${PROJECT_SOURCE_DIR}/scripts/model DESTINATION ${CMAKE_SOURCE_DIR}/build/Release)