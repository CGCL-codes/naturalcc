/* Test-only independent Contract-3 fixture. Do not include in production builds. */
#include <stdio.h>
#include <string.h>

struct contract3_box { int *pointer; };

static int contract3_identity(int value) { return value; }
static int contract3_read(int *pointer) { return *pointer; }
static void contract3_copy(char *target, const char *source) { strcpy(target, source); }
static int *contract3_select(int *pointer, int key) { return key > 0 ? pointer : 0; }

static int contract3_independent_array_oob_case_01(void)
{
    int values[3] = {1, 2, 3};
    int total = 0;
    for (int index = 3; index-- > 0;) total += values[index];
    return total;
}

static int contract3_independent_array_oob_case_02(void)
{
    int values[3] = {1, 2, 3};
    return values[sizeof(values) / sizeof(values[0]) - 1];
}

static int contract3_independent_array_oob_case_03(void)
{
    int values[3] = {1, 2, 3};
    int index = contract3_identity(2);
    return values[index];
}

static int contract3_independent_array_oob_case_04(void)
{
    int values[3] = {1, 2, 3};
    int total = 0;
    for (int index = 3; index >= 0; --index) total += values[index];
    return total;
}

static int contract3_independent_array_oob_case_05(void)
{
    int values[3] = {1, 2, 3};
    int total = 0;
    for (int index = 3; index-- > 0;) total += values[index];
    return total;
}

static int contract3_independent_array_oob_case_06(void)
{
    int values[3] = {1, 2, 3};
    int index = contract3_identity(3);
    if (index >= 3) return 0;
    return values[index];
}

static int contract3_independent_array_oob_case_07(void)
{
    int values[3] = {1, 2, 3};
    int index = 2;
    return values[index];
}

static int contract3_independent_array_oob_case_08(void)
{
    int values[3] = {1, 2, 3};
    return values[sizeof(values) / sizeof(values[0])];
}

static int contract3_independent_array_oob_case_09(void)
{
    int values[3] = {1, 2, 3};
    return values[sizeof(values) / sizeof(values[0]) - 1];
}

static int contract3_independent_array_oob_case_10(void)
{
    int values[3] = {1, 2, 3};
    int index = contract3_identity(3);
    if (index > 3) return 0;
    return values[index];
}

static int contract3_independent_array_oob_case_11(void)
{
    int values[3] = {1, 2, 3};
    int index = contract3_identity(3);
    if (index > 3) return 0;
    return values[index];
}

static int contract3_independent_array_oob_case_12(void)
{
    int values[3] = {1, 2, 3};
    return values[sizeof(values) / sizeof(values[0])];
}

static int contract3_independent_array_oob_case_13(void)
{
    int values[3] = {1, 2, 3};
    return values[2];
}

static int contract3_independent_array_oob_case_14(void)
{
    int values[3] = {1, 2, 3};
    int total = 0;
    for (int index = 3; index >= 0; --index) total += values[index];
    return total;
}

static int contract3_independent_array_oob_case_15(void)
{
    int values[3] = {1, 2, 3};
    int *end = values + 3;
    return end[-1];
}

static int contract3_independent_array_oob_case_16(void)
{
    int values[3] = {1, 2, 3};
    int index = contract3_identity(3);
    return values[index];
}

static int contract3_independent_array_oob_case_17(void)
{
    int values[4] = {1, 2, 3, 4};
    return values[4];
}

static int contract3_independent_array_oob_case_18(void)
{
    int values[3] = {1, 2, 3};
    int total = 0;
    for (int index = 0; index <= 3; ++index) total += values[index];
    return total;
}

static int contract3_independent_array_oob_case_19(void)
{
    int values[3] = {1, 2, 3};
    int total = 0;
    for (int index = 0; index < 3; ++index) total += values[index];
    return total;
}

static int contract3_independent_array_oob_case_20(void)
{
    int values[4] = {1, 2, 3, 4};
    return values[3];
}

static int contract3_independent_array_oob_case_21(void)
{
    int values[3] = {1, 2, 3};
    return values[sizeof(values) / sizeof(values[0])];
}

static int contract3_independent_array_oob_case_22(void)
{
    int values[3] = {1, 2, 3};
    int *cursor = values + 2;
    return *cursor;
}

static int contract3_independent_array_oob_case_23(void)
{
    int values[3] = {1, 2, 3};
    int index = contract3_identity(3);
    if (index >= 3) return 0;
    return values[index];
}

static int contract3_independent_array_oob_case_24(void)
{
    int values[4] = {1, 2, 3, 4};
    return values[4];
}

static int contract3_independent_array_oob_case_25(void)
{
    int values[3] = {1, 2, 3};
    int *end = values + 3;
    return end[-1];
}

static int contract3_independent_array_oob_case_26(void)
{
    int values[3] = {1, 2, 3};
    return values[sizeof(values) / sizeof(values[0]) - 1];
}

static int contract3_independent_array_oob_case_27(void)
{
    int values[3] = {1, 2, 3};
    int *end = values + 3;
    return end[0];
}

static int contract3_independent_array_oob_case_28(void)
{
    int values[4] = {1, 2, 3, 4};
    return values[3];
}

static int contract3_independent_array_oob_case_29(void)
{
    int values[3] = {1, 2, 3};
    int index = contract3_identity(3);
    if (index > 3) return 0;
    return values[index];
}

static int contract3_independent_array_oob_case_30(void)
{
    int values[3] = {1, 2, 3};
    int *end = values + 3;
    return end[-1];
}

static int contract3_independent_array_oob_case_31(void)
{
    int values[3] = {1, 2, 3};
    int total = 0;
    for (int index = 3; index-- > 0;) total += values[index];
    return total;
}

static int contract3_independent_array_oob_case_32(void)
{
    int values[3] = {1, 2, 3};
    int index = 3;
    return values[index];
}

static int contract3_independent_array_oob_case_33(void)
{
    int values[3] = {1, 2, 3};
    int index = contract3_identity(3);
    if (index >= 3) return 0;
    return values[index];
}

static int contract3_independent_array_oob_case_34(void)
{
    int values[3] = {1, 2, 3};
    int *end = values + 3;
    return end[0];
}

static int contract3_independent_array_oob_case_35(void)
{
    int values[3] = {1, 2, 3};
    int *end = values + 3;
    return end[0];
}

static int contract3_independent_array_oob_case_36(void)
{
    int values[3] = {1, 2, 3};
    int *cursor = values + 3;
    return *cursor;
}

static int contract3_independent_array_oob_case_37(void)
{
    int values[4] = {1, 2, 3, 4};
    return values[3];
}

static int contract3_independent_array_oob_case_38(void)
{
    int values[3] = {1, 2, 3};
    return values[3];
}

static int contract3_independent_array_oob_case_39(void)
{
    int values[3] = {1, 2, 3};
    int total = 0;
    for (int index = 3; index >= 0; --index) total += values[index];
    return total;
}

static int contract3_independent_array_oob_case_40(void)
{
    int values[4] = {1, 2, 3, 4};
    return values[4];
}

