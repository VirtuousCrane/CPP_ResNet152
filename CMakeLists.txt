cmake_minimum_required(VERSION 2.8.12)
project(mnist)

include(~/dlib-19.6/dlib/cmake)

add_executable(main main.cpp)
target_link_libraries(main dlib::dlib)

# To Compile:
# mkdir build
# cd build
# cmake ..
# cmake --build . --config Release

