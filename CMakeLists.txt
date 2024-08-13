cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_FLAGS -std=c++17)

project(Itera VERSION 0.1)

include_directories(include)

set(MAIN_SRC_FILE src/main.cpp)

add_executable(Itera ${MAIN_SRC_FILE})

target_link_libraries(Itera -lraylib -lm -lpthread -ldl -lOpenGL -lX11)