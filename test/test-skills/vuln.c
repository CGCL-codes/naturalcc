#include <stdio.h>
#include <string.h>

void unsafe_copy(const char *user_input) {
    char buffer[8];

    // 栈缓冲区溢出：CWE-120
    strcpy(buffer, user_input);

    // 非安全格式化字符串：CWE-134
    printf(user_input);
}

int main(void) {
    unsafe_copy("this input is longer than eight bytes");
    return 0;
}