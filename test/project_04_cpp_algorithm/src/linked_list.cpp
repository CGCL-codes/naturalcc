// 项目: project_04_cpp_algorithm
// 测试主题: C++ 算法实现与数据结构
// 功能: 单链表实现

#include "linked_list.h"
#include <iostream>

namespace algo {

LinkedList::LinkedList() : head_(nullptr), size_(0) {}

LinkedList::~LinkedList() {
    while (head_) {
        ListNode* tmp = head_;
        head_ = head_->next;
        delete tmp;
    }
}

void LinkedList::push_front(int val) {
    ListNode* node = new ListNode(val);
    node->next = head_;
    head_ = node;
    ++size_;
}

void LinkedList::push_back(int val) {
    ListNode* node = new ListNode(val);
    if (!head_) {
        head_ = node;
    } else {
        ListNode* cur = head_;
        while (cur->next) cur = cur->next;
        cur->next = node;
    }
    ++size_;
}

bool LinkedList::remove(int val) {
    if (!head_) return false;
    if (head_->val == val) {
        ListNode* tmp = head_;
        head_ = head_->next;
        delete tmp;
        --size_;
        return true;
    }
    ListNode* cur = head_;
    while (cur->next && cur->next->val != val) cur = cur->next;
    if (cur->next) {
        ListNode* tmp = cur->next;
        cur->next = tmp->next;
        delete tmp;
        --size_;
        return true;
    }
    return false;
}

void LinkedList::reverse() {
    ListNode* prev = nullptr;
    ListNode* cur = head_;
    while (cur) {
        ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    head_ = prev;
}

void LinkedList::print() const {
    ListNode* cur = head_;
    while (cur) {
        std::cout << cur->val;
        if (cur->next) std::cout << " -> ";
        cur = cur->next;
    }
    std::cout << std::endl;
}

size_t LinkedList::size() const { return size_; }

}  // namespace algo
