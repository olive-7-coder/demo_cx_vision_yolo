cmake_minimum_required(VERSION 3.16)
project(RoboMasterDetect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_RUNTIME vc16)
set(OpenCV_ARCH x64)

find_package(OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs highgui videoio dnn
)

file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(detect ${SOURCES})

target_include_directories(detect PRIVATE include)

target_link_libraries(detect PRIVATE ${OpenCV_LIBS})

file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/results")
