import json
from pathlib import Path

import pytest

from code_agent.security_c_macros import harvest_attribute_macros, neutralize
from code_agent.security_c_rules import _error_nodes, parse_source, scan_c_source

DATASET = Path(__file__).resolve().parents[1] / "artifacts" / "contract5" / "rtthread" / "dataset"


def errors(source: str, macros=None) -> list[str]:
    root = parse_source(source, ".c", macros)
    lines = source.splitlines()
    return [lines[node.start_point.row].strip() for node in _error_nodes(root)]


def assert_same_shape(original: str, rewritten: str) -> None:
    assert len(rewritten) == len(original)
    assert [i for i, c in enumerate(rewritten) if c == "\n"] == [i for i, c in enumerate(original) if c == "\n"]


# --- offsets never move -------------------------------------------------------------

def test_rewrite_keeps_every_offset_and_line():
    source = '''rt_weak int f(void)
{
    struct item *it = rt_list_entry(node, struct item, list);
    rt_list_for_each(node, &head)
    {
        use(node);
    }
    return 0;
}
MSH_CMD_EXPORT(f, list the FILEs.);
'''
    rewritten = neutralize(source)
    assert_same_shape(source, rewritten)
    assert rewritten != source


def test_findings_keep_original_line_numbers():
    source = '''#include <rtthread.h>
rt_weak void run(const char *name)
{
    char cmd[64];
    sprintf(cmd, "echo %s", name);
    system(cmd);
}
MSH_CMD_EXPORT(run, run a command);
'''
    findings, failed = scan_c_source(source, "main.c", 0)
    assert not failed
    assert [f["line"] for f in findings if f["category"] == "command_execution"] == [6]


# --- the constructs RT-Thread uses ------------------------------------------------

@pytest.mark.parametrize("source", [
    "rt_weak int handler(void)\n{\n    return 0;\n}\n",
    "rt_inline int fast(int x)\n{\n    return x + 1;\n}\n",
    "rt_section(\".text.fast\") void boot(void)\n{\n}\n",
    "ALIGN(RT_ALIGN_SIZE) static char stack[512];\nint ok;\n",
])
def test_attribute_macros(source):
    assert errors(source) == []


def test_harvested_attribute_macros_from_project_headers():
    header = """
#define rt_weak                     __attribute__((weak))
#define rt_inline                   static __inline
#define rt_always_inline            rt_inline
#define rt_noreturn
#define rt_section(x)               __attribute__((section(x)))
#define DLOG(...)
#define __CONTRACT_H__
#define MY_QUALIFIER                volatile const
"""
    object_like, function_like = harvest_attribute_macros([header])
    assert {"rt_weak", "rt_inline", "rt_always_inline", "rt_noreturn", "MY_QUALIFIER"} <= object_like
    assert "rt_section" in function_like
    # Logging statements parse as calls already; include guards never appear in code.
    assert "DLOG" not in function_like and "__CONTRACT_H__" not in object_like
    source = "MY_QUALIFIER int reg;\nint f(void)\n{\n    return reg;\n}\n"
    assert errors(source) and not errors(source, (object_like, function_like))


def test_label_sharing_an_attribute_name_is_untouched():
    source = "int f(int x)\n{\n    if (x) goto __used;\n    return 0;\n__used:\n    return 1;\n}\n"
    assert neutralize(source) == source


def test_expression_macro_sharing_an_attribute_name_is_untouched():
    source = "int f(int len)\n{\n    int size = ALIGN(len, 4);\n    return size;\n}\n"
    assert neutralize(source) == source


def test_registrations_with_free_text_at_file_scope():
    source = '''int cmd_ls(int argc, char **argv)
{
    return 0;
}
MSH_CMD_EXPORT_ALIAS(cmd_ls, ls, List information about the FILEs.);
CMD_OPTIONS_NODE_START(cmd_list)
CMD_OPTIONS_NODE(1, thread, list threads)
CMD_OPTIONS_NODE(2, timer, list timers)
CMD_OPTIONS_NODE_END
FINSH_FUNCTION_EXPORT(copy, copy file or dir)
int after(void)
{
    return 1;
}
'''
    assert errors(source) == []
    assert "int after(void)" in neutralize(source)


def test_uppercase_call_inside_a_function_is_kept():
    source = "int f(void)\n{\n    RT_ASSERT(f != 0);\n    return 0;\n}\n"
    assert neutralize(source) == source


@pytest.mark.parametrize("source", [
    "void f(rt_list_t *node)\n{\n    struct rt_object *obj = rt_list_entry(node, struct rt_object, list);\n}\n",
    "void f(va_list ap)\n{\n    void *arg = va_arg(ap, void *);\n}\n",
    "void f(void)\n{\n    RT_OBJECT_HOOK_CALL(hook, ());\n}\n",
])
def test_type_arguments(source):
    assert errors(source) == []


def test_sizeof_keeps_its_type_argument():
    source = "int f(void)\n{\n    return sizeof(struct rt_object);\n}\n"
    assert neutralize(source) == source


def test_iterator_macro_becomes_a_loop():
    source = "void f(void)\n{\n    rt_list_for_each_entry(item, &head, list)\n    {\n        use(item);\n    }\n}\n"
    assert errors(source) == []
    assert "while" in neutralize(source)


# --- #if arms that split a statement ---------------------------------------------

@pytest.mark.parametrize("source", [
    # each arm opens the brace the shared body closes
    "void f(int a, int b)\n{\n#ifdef X\n    if (a) {\n#else\n    if (b) {\n#endif\n        go();\n    }\n}\n",
    # an arm that continues the if before it
    "void f(int a)\n{\n    if (a) { one(); }\n#ifdef DMA\n    else if (a > 1) { two(); }\n#endif\n}\n",
    # the code after #endif continues the arms
    "void f(int a)\n{\n#ifdef S\n    if (a) { return; }\n#else\n    if (!a) { return; }\n#endif\n    else if (a > 2) { go(); }\n}\n",
    # an expression split by #ifdef
    "int f(void)\n{\n    int mask =\n#ifdef O_ASYNC\n        O_ASYNC |\n#endif\n        O_RDONLY;\n    return mask;\n}\n",
])
def test_split_conditionals(source):
    assert errors(source) == []


def test_parser_picks_the_arm_that_is_plain_c():
    """A newlib-only declarator in the first arm: the second arm is kept instead."""
    source = '''#if defined(RT_USING_NEWLIB) && defined(_EXFUN)
_READ_WRITE_RETURN_TYPE _EXFUN(read, (int fd, void *buf, size_t len))
#else
ssize_t read(int fd, void *buf, size_t len)
#endif
{
    return 0;
}
'''
    assert errors(source) == []


def test_balanced_conditionals_are_left_alone():
    source = "int f(void)\n{\n#ifdef A\n    one();\n#else\n    two();\n#endif\n    return 0;\n}\n"
    assert neutralize(source) == source


# --- the dataset that motivated this --------------------------------------------

@pytest.mark.skipif(not (DATASET / "ground_truth.json").exists(), reason="dataset not built")
def test_every_contract5_detection_file_parses_cleanly():
    from code_agent.security_c_macros import harvest_project_macros
    ground_truth = json.loads((DATASET / "ground_truth.json").read_text(encoding="utf-8"))
    macros = harvest_project_macros(DATASET / "project")
    for row in ground_truth["detection_files"]:
        source = (DATASET / "project" / row["path"]).read_text(encoding="utf-8")
        assert_same_shape(source, neutralize(source, *macros))
        assert errors(source, macros) == [], row["path"]
