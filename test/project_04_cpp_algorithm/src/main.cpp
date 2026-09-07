// 项目: project_04_cpp_algorithm
// 测试主题: C++ 算法实现与数据结构
// 功能: 主程序，演示排序和链表操作

#include <iostream>
#include <vector>
#include "sorting.h"
#include "linked_list.h"

int main() {
    using namespace algo;

    // 测试排序
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    std::cout << "原始数组: ";
    print_array(arr);

    std::vector<int> arr2 = arr;
    bubble_sort(arr2);
    std::cout << "冒泡排序: ";
    print_array(arr2);

    std::vector<int> arr3 = arr;
    quick_sort(arr3);
    std::cout << "快速排序: ";
    print_array(arr3);

    std::vector<int> arr4 = arr;
    shell_sort(arr4);
    std::cout << "希尔排序: ";
    print_array(arr4);

    // 测试链表
    LinkedList list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.push_front(0);
    std::cout << "链表: ";
    list.print();

    list.reverse();
    std::cout << "反转后: ";
    list.print();

    list.remove(2);
    std::cout << "删除 2 后: ";
    list.print();

    return 0;
}
