/* Test-only independent Contract-3 fixture. Do not include in production builds. */
#include <stdio.h>
#include <string.h>

struct contract3_box { int *pointer; };

static int contract3_identity(int value) { return value; }
static int contract3_read(int *pointer) { return *pointer; }
static void contract3_copy(char *target, const char *source) { strcpy(target, source); }
static int *contract3_select(int *pointer, int key) { return key > 0 ? pointer : 0; }

static int contract3_independent_null_pointer_case_01(void)
{
    int *pointer = 0;
    return *pointer;
}

static int contract3_independent_null_pointer_case_02(void)
{
    int value = 7;
    int *pointer = contract3_select(&value, 1);
    return *pointer;
}

static int contract3_independent_null_pointer_case_03(void)
{
    int value = 7;
    int *pointer = 0;
    pointer = &value;
    return *pointer;
}

static int contract3_independent_null_pointer_case_04(void)
{
    int value = 7;
    int *pointer = 0;
    if (pointer == 0) value = -1;
    return *pointer + value;
}

static int contract3_independent_null_pointer_case_05(void)
{
    int value = 7;
    struct contract3_box box = {0};
    if (box.pointer == 0) box.pointer = &value;
    return *box.pointer;
}

static int contract3_independent_null_pointer_case_06(void)
{
    int value = 7;
    int *pointer = 0;
    if (pointer == 0) value = -1;
    return *pointer + value;
}

static int contract3_independent_null_pointer_case_07(void)
{
    int value = 7;
    struct contract3_box box = {0};
    if (box.pointer == 0) box.pointer = &value;
    return *box.pointer;
}

static int contract3_independent_null_pointer_case_08(void)
{
    int value = 7;
    int *pointer = 0;
    pointer = &value;
    return *pointer;
}

static int contract3_independent_null_pointer_case_09(void)
{
    int value = 7;
    int *pointer = &value;
    pointer = 0;
    return *pointer;
}

static int contract3_independent_null_pointer_case_10(void)
{
    int value = 7;
    int *pointer = value < 0 ? &value : 0;
    return *pointer;
}

static int contract3_independent_null_pointer_case_11(void)
{
    int value = 7;
    int *pointer = value < 0 ? &value : 0;
    return pointer ? *pointer : 0;
}

static int contract3_independent_null_pointer_case_12(void)
{
    int value = 7;
    int *pointer = &value;
    pointer = 0;
    return *pointer;
}

static int contract3_independent_null_pointer_case_13(void)
{
    int value = 7;
    int *pointer = value < 0 ? &value : 0;
    return *pointer;
}

static int contract3_independent_null_pointer_case_14(void)
{
    int value = 7;
    int *pointer = contract3_select(&value, 0);
    return *pointer;
}

static int contract3_independent_null_pointer_case_15(void)
{
    int value = 7;
    int *pointer = &value;
    pointer = 0;
    return *pointer;
}

static int contract3_independent_null_pointer_case_16(void)
{
    int value = 7;
    int *pointer = 0;
    if (pointer == 0) return -1;
    return *pointer + value;
}

static int contract3_independent_null_pointer_case_17(void)
{
    int value = 7;
    int *pointer = 0;
    if (value > 0) pointer = &value;
    return *pointer;
}

static int contract3_independent_null_pointer_case_18(void)
{
    int value = 7;
    int *pointer = value < 0 ? &value : 0;
    return *pointer;
}

static int contract3_independent_null_pointer_case_19(void)
{
    int value = 7;
    return contract3_read(&value);
}

static int contract3_independent_null_pointer_case_20(void)
{
    int value = 7;
    int *pointer = 0;
    if (value < 0) pointer = &value;
    return *pointer;
}

static int contract3_independent_null_pointer_case_21(void)
{
    int value = 7;
    return contract3_read(0);
}

static int contract3_independent_null_pointer_case_22(void)
{
    int value = 7;
    struct contract3_box box = {0};
    if (box.pointer != 0) box.pointer = &value;
    return *box.pointer;
}

static int contract3_independent_null_pointer_case_23(void)
{
    int value = 7;
    int *pointer = 0;
    if (pointer == 0) return -1;
    return *pointer + value;
}

static int contract3_independent_null_pointer_case_24(void)
{
    int value = 7;
    struct contract3_box box = {0};
    if (box.pointer != 0) box.pointer = &value;
    return *box.pointer;
}

static int contract3_independent_null_pointer_case_25(void)
{
    int value = 7;
    int *pointer = contract3_select(&value, 0);
    return *pointer;
}

static int contract3_independent_null_pointer_case_26(void)
{
    int value = 7;
    int *pointer = 0;
    if (pointer == 0) return -1;
    return *pointer + value;
}

static int contract3_independent_null_pointer_case_27(void)
{
    int value = 7;
    int *pointer = value < 0 ? &value : 0;
    return pointer ? *pointer : 0;
}

static int contract3_independent_null_pointer_case_28(void)
{
    int value = 7;
    int *pointer = contract3_select(&value, 0);
    return *pointer;
}

static int contract3_independent_null_pointer_case_29(void)
{
    int value = 7;
    int *pointer = 0;
    if (pointer == 0) value = -1;
    return *pointer + value;
}

static int contract3_independent_null_pointer_case_30(void)
{
    int value = 7;
    int *pointer = value < 0 ? &value : 0;
    return pointer ? *pointer : 0;
}

static int contract3_independent_null_pointer_case_31(void)
{
    int value = 7;
    return contract3_read(0);
}

static int contract3_independent_null_pointer_case_32(void)
{
    int value = 7;
    int *pointer = contract3_select(&value, 1);
    return *pointer;
}

static int contract3_independent_null_pointer_case_33(void)
{
    int value = 7;
    int *pointer = 0;
    pointer = &value;
    return *pointer;
}

static int contract3_independent_null_pointer_case_34(void)
{
    int value = 7;
    int *pointer = 0;
    if (pointer == 0) value = -1;
    return *pointer + value;
}

static int contract3_independent_null_pointer_case_35(void)
{
    int value = 7;
    return value;
}

static int contract3_independent_null_pointer_case_36(void)
{
    int value = 7;
    int *pointer = 0;
    return *pointer;
}

static int contract3_independent_null_pointer_case_37(void)
{
    int value = 7;
    int *pointer = 0;
    if (pointer == 0) return -1;
    return *pointer + value;
}

static int contract3_independent_null_pointer_case_38(void)
{
    int value = 7;
    int *pointer = &value;
    return *pointer;
}

static int contract3_independent_null_pointer_case_39(void)
{
    int value = 7;
    int *pointer = contract3_select(&value, 1);
    return *pointer;
}

static int contract3_independent_null_pointer_case_40(void)
{
    int value = 7;
    return contract3_read(&value);
}

