// 项目: project_04_cpp_algorithm
// 测试主题: C++ 算法实现与数据结构
// 功能: 排序算法接口

#ifndef SORTING_H
#define SORTING_H

#include <vector>

namespace algo {

// 冒泡排序 - O(n^2)
void bubble_sort(std::vector<int>& arr);

// 快速排序 - 平均 O(n log n)
void quick_sort(std::vector<int>& arr);

// 归并排序 - 稳定 O(n log n)
void merge_sort(std::vector<int>& arr);

// 堆排序 - O(n log n)
void heap_sort(std::vector<int>& arr);

// 希尔排序 - O(n^1.5)
void shell_sort(std::vector<int>& arr);

// 工具函数：打印数组
void print_array(const std::vector<int>& arr);

// 工具函数：判断是否已排序
bool is_sorted(const std::vector<int>& arr);

}  // namespace algo

#endif  // SORTING_H
