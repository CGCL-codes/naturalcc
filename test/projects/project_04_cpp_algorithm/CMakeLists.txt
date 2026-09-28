# 项目: project_04_cpp_algorithm
# 测试主题: C++ 算法实现与数据结构
# 功能: CMake 构建脚本

cmake_minimum_required(VERSION 3.10)
project(cpp_algorithm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 主程序
add_executable(algorithm_demo
    src/main.cpp
    src/sorting.cpp
    src/linked_list.cpp
)

target_include_directories(algorithm_demo PRIVATE include)

# 单元测试
enable_testing()
add_executable(test_algorithm tests/test_main.cpp
    src/sorting.cpp
    src/linked_list.cpp
)
target_include_directories(test_algorithm PRIVATE include)
# TODO: 若需移除重复 main，可把算法实现抽成 algorithm_core 静态库再分别链接
add_test(NAME test_algorithm COMMAND test_algorithm)
