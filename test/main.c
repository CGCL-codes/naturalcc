#include <stdio.h>
#include <stdlib.h>
#include "db_connect.h"
#include "utils.h"

int main() {
    // 安全地分配内存并检查错误
    char *host = malloc(256 * sizeof(char));
    if (!host) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化主机字符串
    snprintf(host, 256, "localhost");
    if (!host) {
        log_error("Memory allocation failed for host string");
        return EXIT_FAILURE;
    }

    // 连接数据库并包含重试逻辑
    for (int i = 0; i < retry_count; i++) {
        connect_to_db(host);
        // 在此处添加连接验证逻辑
    }

    // Free allocated memory
    free(host);
    host = NULL;

    return EXIT_SUCCESS;
}
