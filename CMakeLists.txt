# 设置cmake的最低版本
cmake_minimum_required(VERSION 3.5)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

set(ONNXRUNTIME_DIR "/opt/dtk/onnxruntime/")
# 设置项目名
project(YOLOV5)

# 设置编译器
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++17) # 2.2版本以上需要c++17
set(CMAKE_BUILD_TYPE release)

# 添加头文件路径
set(INCLUDE_PATH    ${CMAKE_CURRENT_SOURCE_DIR}/include/
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/Utility/ 
                    ${ONNXRUNTIME_DIR}/include/
                    ${ROCM_PATH/include/}
                    /usr/local/include/opencv4/)
include_directories(${INCLUDE_PATH})

# 添加依赖库路径
set(LIBRARY_PATH ${ONNXRUNTIME_DIR}/lib/
                 /usr/local/lib64/ 
                 ${ROCM_PATH}/lib/)
link_directories(${LIBRARY_PATH})

# 添加依赖库
set(LIBRARY opencv_core
            opencv_imgproc
            opencv_imgcodecs
            opencv_dnn
            onnxruntime)
link_libraries(${LIBRARY})

# 添加源文件
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/yolov5.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/Utility/Filesystem.cpp
                )

# 添加可执行目标
add_executable(YOLOV5 ${SOURCE_FILES})
