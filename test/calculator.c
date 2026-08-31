#include <stdio.h>
#include <stdlib.h>

/**
 * 简单的计算器，实现不完整。
 * 用于测试 NaturalCC 代码补全功能。
 */

// TODO: 完成此函数 - 应返回 a + b
int add(int a, int b) {
    // 你的实现代码
}

// TODO: 完成此函数 - 应返回 a - b
int subtract(int a, int b) {

}

// TODO: 完成此函数 - 应返回 a * b
int multiply(int a, int b) {
    return 0; // 占位符
}

// BUG: 未处理除零错误
int divide(int a, int b) {
    return a / b;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: %s <num1> <op> <num2>\n", argv[0]);
        return 1;
    }

    int a = atoi(argv[1]);
    int b = atoi(argv[3]);
    char op = argv[2][0];

    int result;
    switch (op) {
        case '+': result = add(a, b); break;
        case '-': result = subtract(a, b); break;
        case '*': result = multiply(a, b); break;
        case '/': result = divide(a, b); break;
        default:
            printf("Unknown operator: %c\n", op);
            return 1;
    }

    printf("%d %c %d = %d\n", a, op, b, result);
    return 0;
}
