// 项目: project_04_cpp_algorithm
// 测试主题: C++ 算法实现与数据结构
// 功能: 排序算法实现

#include "sorting.h"
#include <iostream>
#include <algorithm>

namespace algo {

void bubble_sort(std::vector<int>& arr) {
    size_t n = arr.size();
    for (size_t i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (size_t j = 0; j < n - 1 - i; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

void quick_sort_impl(std::vector<int>& arr, int low, int high) {
    if (low >= high) return;
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    int pi = i + 1;
    quick_sort_impl(arr, low, pi - 1);
    quick_sort_impl(arr, pi + 1, high);
}

void quick_sort(std::vector<int>& arr) {
    if (!arr.empty()) {
        quick_sort_impl(arr, 0, static_cast<int>(arr.size()) - 1);
    }
}

void merge(std::vector<int>& arr, int l, int m, int r) {
    std::vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    std::vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);
    size_t i = 0, j = 0;
    int k = l;
    while (i < left.size() && j < right.size()) {
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    }
    while (i < left.size()) arr[k++] = left[i++];
    while (j < right.size()) arr[k++] = right[j++];
}

void merge_sort_impl(std::vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int m = l + (r - l) / 2;
    merge_sort_impl(arr, l, m);
    merge_sort_impl(arr, m + 1, r);
    merge(arr, l, m, r);
}

void merge_sort(std::vector<int>& arr) {
    if (!arr.empty()) {
        merge_sort_impl(arr, 0, static_cast<int>(arr.size()) - 1);
    }
}

void heap_sort(std::vector<int>& arr) {
    std::make_heap(arr.begin(), arr.end());
    for (auto it = arr.end(); it != arr.begin(); --it) {
        std::pop_heap(arr.begin(), it);
    }
}

void print_array(const std::vector<int>& arr) {
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i];
        if (i + 1 < arr.size()) std::cout << " ";
    }
    std::cout << std::endl;
}

bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < arr[i - 1]) return false;
    }
    return true;
}

}  // namespace algo
