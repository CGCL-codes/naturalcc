// 项目: project_04_cpp_algorithm
// 测试主题: C++ 算法实现与数据结构
// 功能: 单链表实现

#ifndef LINKED_LIST_H
#define LINKED_LIST_H

namespace algo {

struct ListNode {
    int val;
    ListNode* next;
    explicit ListNode(int v) : val(v), next(nullptr) {}
};

class LinkedList {
public:
    LinkedList();
    ~LinkedList();

    // 在头部插入
    void push_front(int val);
    // 在尾部插入
    void push_back(int val);
    // 删除指定值的第一个节点
    bool remove(int val);
    // 反转链表
    void reverse();
    // 打印链表
    void print() const;
    // 获取长度
    size_t size() const;

private:
    ListNode* head_;
    size_t size_;
};

}  // namespace algo

#endif  // LINKED_LIST_H
