cmake_minimum_required(VERSION 3.16)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(Itera VERSION 0.1)

# Include the header files
include_directories(include)

# Source file
set(MAIN_SRC_FILE src/main.cpp)

# Add the executable
add_executable(Itera ${MAIN_SRC_FILE})

# Find and link raylib
if (WIN32)
    # Specify the path to raylib if it's not in a standard location
    find_package(raylib CONFIG REQUIRED)
    target_link_libraries(Itera raylib opengl32 gdi32 winmm)
elseif(UNIX)
    find_package(raylib REQUIRED)
    target_link_libraries(Itera raylib pthread dl m X11)
endif()
