#include <stdio.h>
#include <string.h>

/*
 * Intentional vulnerable sample for vulnerability_detection plugin testing.
 * Do not use in production.
 */
void vulnerable_copy(const char *user_input) {
    char buf[16];
    strcpy(buf, user_input);      // CWE-120: potential buffer overflow
    printf(user_input);           // CWE-134: potential format string issue
    sprintf(buf, "%s", user_input); // CWE-120: unsafe formatted write
}

int main(int argc, char **argv) {
    if (argc > 1) {
        vulnerable_copy(argv[1]);
    }
    return 0;
}
