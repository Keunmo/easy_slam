cmake_minimum_required(VERSION 3.10)
project(EASY_SLAM)

set(CMAKE_CXX_STANDARD 14)

# pybind11 경로 설정
# add_subdirectory(${CMAKE_SOURCE_DIR}/Thirdparty/pybind11)
find_package(PythonLibs REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# # pybind11 및 python 인터프리터 링크
link_directories(${PYTHON_LIBRARY_DIRS})
link_libraries(${PYTHON_LIBRARIES} ${OpenCV_LIBS})

add_library(EASY_SLAM SHARED
    src/FeatureExtractor.cc
    src/FeatureMatcher.cc
    src/System.cc
)

# # pybind11과 링크
target_link_libraries(EASY_SLAM
    pybind11::module
    ${OpenCV_LIBS}
    ${PYTHON_LIBRARIES}
)

add_executable(stereo_kitti Examples/stereo_kitti.cc)
target_link_libraries(stereo_kitti EASY_SLAM)
