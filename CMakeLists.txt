project(posenet)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INSTALL_PREFIX /bin)
set(CMAKE_BUILD_TYPE Debug)
set("OpenCV_DIR" "C:\\opencv\\build\\x64\\vc14\\lib\\")
find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRECTORIES}
    include/
)
add_executable(main main.cpp src/pose_net.cpp)
target_link_libraries(main ${OpenCV_LIBS})
