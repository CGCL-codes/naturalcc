/* Test-only independent Contract-3 fixture. Do not include in production builds. */
#include <stdio.h>
#include <string.h>

struct contract3_box { int *pointer; };

static int contract3_identity(int value) { return value; }
static int contract3_read(int *pointer) { return *pointer; }
static void contract3_copy(char *target, const char *source) { strcpy(target, source); }
static int *contract3_select(int *pointer, int key) { return key > 0 ? pointer : 0; }

static int contract3_independent_string_overflow_case_01(void)
{
    char target[4] = {0};
    snprintf(target, sizeof(target), "%s", "ABCDEFGHIJ");
    return target[0];
}

static int contract3_independent_string_overflow_case_02(void)
{
    char target[8] = "ABC";
    strcat(target, "DEFG");
    return target[0];
}

static int contract3_independent_string_overflow_case_03(void)
{
    char target[4] = {0};
    snprintf(target, sizeof(target), "%s", "ABCDEFGHIJ");
    return target[0];
}

static int contract3_independent_string_overflow_case_04(void)
{
    char target[4];
    strncpy(target, "ABCDEFGH", 8);
    return target[0];
}

static int contract3_independent_string_overflow_case_05(void)
{
    char target[8] = "ABC";
    strcat(target, "DEFGH");
    return target[0];
}

static int contract3_independent_string_overflow_case_06(void)
{
    char target[6] = {0};
    strcpy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_07(void)
{
    char target[5] = {0};
    strcpy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_08(void)
{
    char target[6] = {0};
    strcpy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_09(void)
{
    char target[4];
    memcpy(target, "ABCDE", sizeof(target));
    return target[0];
}

static int contract3_independent_string_overflow_case_10(void)
{
    char target[8] = "A";
    strcat(target, "BCDEFGH");
    return target[0];
}

static int contract3_independent_string_overflow_case_11(void)
{
    char target[4] = {0};
    memcpy(target, "ABCDE", 3);
    return target[0];
}

static int contract3_independent_string_overflow_case_12(void)
{
    char target[8] = "A";
    strcat(target, "BC");
    return target[0];
}

static int contract3_independent_string_overflow_case_13(void)
{
    char target[6] = {0};
    strcpy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_14(void)
{
    char target[4];
    memcpy(target, "ABCDE", sizeof("ABCDE"));
    return target[0];
}

static int contract3_independent_string_overflow_case_15(void)
{
    char target[8] = "ABC";
    strcat(target, "DEFG");
    return target[0];
}

static int contract3_independent_string_overflow_case_16(void)
{
    char target[4] = {0};
    contract3_copy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_17(void)
{
    char target[8] = "ABC";
    strcat(target, "DEFGH");
    return target[0];
}

static int contract3_independent_string_overflow_case_18(void)
{
    char target[5] = {0};
    strcpy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_19(void)
{
    char target[4] = {0};
    snprintf(target, sizeof(target), "%s", "ABCDEFGHIJ");
    return target[0];
}

static int contract3_independent_string_overflow_case_20(void)
{
    char target[4] = {0};
    contract3_copy(target, "ABC");
    return target[0];
}

static int contract3_independent_string_overflow_case_21(void)
{
    char target[6] = {0};
    strcpy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_22(void)
{
    char target[4] = {0};
    snprintf(target, 16, "%s", "ABCDEFGHIJ");
    return target[0];
}

static int contract3_independent_string_overflow_case_23(void)
{
    char target[4] = {0};
    memcpy(target, "ABCDE", 6);
    return target[0];
}

static int contract3_independent_string_overflow_case_24(void)
{
    char target[4] = {0};
    snprintf(target, sizeof(target), "%s", "ABCDEFGHIJ");
    return target[0];
}

static int contract3_independent_string_overflow_case_25(void)
{
    char target[4] = {0};
    sprintf(target, "%s", "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_26(void)
{
    char target[4];
    strncpy(target, "ABCDEFGH", sizeof(target) - 1);
    target[sizeof(target) - 1] = '\0';
    return target[0];
}

static int contract3_independent_string_overflow_case_27(void)
{
    char target[5] = {0};
    strcpy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_28(void)
{
    char target[4];
    strncpy(target, "ABCDEFGH", sizeof(target) - 1);
    target[sizeof(target) - 1] = '\0';
    return target[0];
}

static int contract3_independent_string_overflow_case_29(void)
{
    char target[8] = "ABC";
    strcat(target, "DEFGH");
    return target[0];
}

static int contract3_independent_string_overflow_case_30(void)
{
    char target[4];
    strncpy(target, "ABCDEFGH", 8);
    return target[0];
}

static int contract3_independent_string_overflow_case_31(void)
{
    char target[4] = {0};
    snprintf(target, 16, "%s", "ABCDEFGHIJ");
    return target[0];
}

static int contract3_independent_string_overflow_case_32(void)
{
    char target[8] = "ABC";
    strcat(target, "DEFG");
    return target[0];
}

static int contract3_independent_string_overflow_case_33(void)
{
    char target[4];
    strncpy(target, "ABCDEFGH", 8);
    return target[0];
}

static int contract3_independent_string_overflow_case_34(void)
{
    char target[4] = {0};
    strcpy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_35(void)
{
    char target[5] = {0};
    strcpy(target, "ABCDE");
    return target[0];
}

static int contract3_independent_string_overflow_case_36(void)
{
    char target[4] = {0};
    snprintf(target, 16, "%s", "ABCDEFGHIJ");
    return target[0];
}

static int contract3_independent_string_overflow_case_37(void)
{
    char target[4] = {0};
    sprintf(target, "%s", "ABC");
    return target[0];
}

static int contract3_independent_string_overflow_case_38(void)
{
    char target[4] = {0};
    snprintf(target, 16, "%s", "ABCDEFGHIJ");
    return target[0];
}

static int contract3_independent_string_overflow_case_39(void)
{
    char target[4];
    strncpy(target, "ABCDEFGH", sizeof(target) - 1);
    target[sizeof(target) - 1] = '\0';
    return target[0];
}

static int contract3_independent_string_overflow_case_40(void)
{
    char target[4] = {0};
    strcpy(target, "ABC");
    return target[0];
}

