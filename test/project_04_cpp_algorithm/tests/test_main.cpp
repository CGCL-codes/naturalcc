// 项目: project_04_cpp_algorithm
// 测试主题: C++ 算法实现与数据结构
// 功能: 单元测试主程序（无外部框架，基于断言，全部通过返回 0）

#include "sorting.h"
#include "linked_list.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void report_failure(const std::string& name) {
    std::cout << "[FAIL] " << name << std::endl;
    ++g_failures;
}

void check_sorted(const std::string& name, const std::vector<int>& expected,
                  const std::vector<int>& actual) {
    if (expected != actual) {
        std::cout << "[FAIL] " << name << " -> unexpected result\n";
        ++g_failures;
    } else if (!algo::is_sorted(actual)) {
        std::cout << "[FAIL] " << name << " -> result not sorted\n";
        ++g_failures;
    } else {
        std::cout << "[PASS] " << name << std::endl;
    }
}

// 对给定排序函数运行一组用例：分别用已知的带符号序列与包含重复元素的序列。
void run_sort_tests(const std::string& fn, void (*sort)(std::vector<int>&)) {
    const std::vector<int> input = {64, 34, 25, 12, 22, 11, -90, 7, 12};
    std::vector<int> expected = input;

    // 参考排序结果（不依赖被测算法）
    for (size_t i = 0; i < expected.size(); ++i) {
        for (size_t j = i + 1; j < expected.size(); ++j) {
            if (expected[j] < expected[i]) {
                int tmp = expected[i];
                expected[i] = expected[j];
                expected[j] = tmp;
            }
        }
    }

    std::vector<int> out = input;
    sort(out);
    check_sorted(fn + " (multi-case)", expected, out);

    // 空数组与单元素
    std::vector<int> empty;
    sort(empty);
    if (empty.empty()) {
        std::cout << "[PASS] " << fn << " (empty)" << std::endl;
    } else {
        report_failure(fn + " (empty)");
    }

    std::vector<int> single = {42};
    sort(single);
    check_sorted(fn + " (single)", {42}, single);

    // 已排序的数组（应保持不变）
    std::vector<int> pre = {1, 2, 3, 4, 5};
    sort(pre);
    check_sorted(fn + " (pre-sorted)", {1, 2, 3, 4, 5}, pre);
}

void test_merge_sort() { run_sort_tests("merge_sort", algo::merge_sort); }
void test_heap_sort() { run_sort_tests("heap_sort", algo::heap_sort); }

void test_linked_list() {
    using algo::LinkedList;

    // 基础插入与 size
    {
        LinkedList list;
        if (list.size() != 0) report_failure("LinkedList initial size");
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.push_front(0);
        if (list.size() != 4) report_failure("LinkedList size after 4 pushes");
        list.print();
        std::cout << "[PASS] LinkedList construct/push/size/print" << std::endl;
    }

    // reverse
    {
        LinkedList list;
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.reverse();
        if (list.size() != 3) report_failure("LinkedList reverse size");
        list.print();
        std::cout << "[PASS] LinkedList reverse" << std::endl;
    }

    // remove（命中头部 / 中部 / 缺失）
    {
        LinkedList list;
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        if (!list.remove(20)) report_failure("LinkedList remove middle returned false");
        if (!list.remove(10)) report_failure("LinkedList remove head returned false");
        if (list.remove(99)) report_failure("LinkedList remove absent returned true");
        if (list.size() != 1) report_failure("LinkedList size after removals");
        list.print();
        std::cout << "[PASS] LinkedList remove" << std::endl;
    }

    // 在空链表上 remove 与 reverse
    {
        LinkedList list;
        if (list.remove(5)) report_failure("LinkedList remove on empty returned true");
        list.reverse();  // 不应崩溃
        std::cout << "[PASS] LinkedList edge cases (empty)" << std::endl;
    }
}

// 捕获 LinkedList::print() 输出的文本（形如 "1 -> 2 -> 3\n"），
// 用于验证链表的真实节点内容与顺序（仅依赖对外接口 print()）。
std::string capture_print_output(const algo::LinkedList& list) {
    std::stringstream ss;
    std::streambuf* old = std::cout.rdbuf(ss.rdbuf());
    list.print();
    std::cout.rdbuf(old);
    return ss.str();
}

// 基于 print 输出的文本对链表内容做功能断言：
// push_back 保序、push_front 插最前、reverse 反转、remove 精准删除。
void test_linked_list_content() {
    using algo::LinkedList;

    // push_back + push_front 的最终顺序
    {
        LinkedList list;
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.push_front(0);
        if (capture_print_output(list) != "0 -> 1 -> 2 -> 3\n")
            report_failure("LinkedList content after push_back/push_front");
        else
            std::cout << "[PASS] LinkedList content order 0->1->2->3" << std::endl;
    }

    // reverse 后顺序反转
    {
        LinkedList list;
        list.push_back(3);
        list.push_back(2);
        list.push_back(1);
        list.reverse();
        if (capture_print_output(list) != "1 -> 2 -> 3\n")
            report_failure("LinkedList reverse content");
        else
            std::cout << "[PASS] LinkedList reverse content 1->2->3" << std::endl;
    }

    // remove 掉头部与中部后的剩余内容
    {
        LinkedList list;
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
        list.remove(20);
        list.remove(10);
        if (capture_print_output(list) != "30\n")
            report_failure("LinkedList remove content remains");
        else
            std::cout << "[PASS] LinkedList remove leftover 30" << std::endl;
    }
}

}  // namespace

int main() {
    test_merge_sort();
    test_heap_sort();
    // 其余排序算法也做一致性检查
    run_sort_tests("bubble_sort", algo::bubble_sort);
    run_sort_tests("quick_sort", algo::quick_sort);
    run_sort_tests("shell_sort", algo::shell_sort);

    // is_sorted 正向用例（额外覆盖已排序/未排序判定经由 check_sorted 完成）
    if (algo::is_sorted({1, 2, 3})) {
        std::cout << "[PASS] is_sorted (positive)" << std::endl;
    } else {
        report_failure("is_sorted (positive)");
    }
    // is_sorted 负向用例：乱序数组应判定为非有序
    if (!algo::is_sorted({3, 1, 2})) {
        std::cout << "[PASS] is_sorted (negative)" << std::endl;
    } else {
        report_failure("is_sorted (negative)");
    }

    test_linked_list();
    test_linked_list_content();

    if (g_failures == 0) {
        std::cout << "\nALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << "\n" << g_failures << " TEST(S) FAILED" << std::endl;
    return 1;
}
