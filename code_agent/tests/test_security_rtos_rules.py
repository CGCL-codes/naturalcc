import pytest

from code_agent.security_rtos_rules import scan_rtos_source


def scan(source: str, suffix: str = ".c"):
    return scan_rtos_source(source, "module" + suffix, 1)[0]


def categories(source: str) -> list[str]:
    return [f["category"] for f in scan(source)]


# --- the gap these rules exist to close -----------------------------------

def test_rtos_api_spellings_are_recognised():
    """The libc-shaped rules miss rt_-prefixed wrappers entirely."""
    source = """
int build(const char *name)
{
    char cmd[64];
    rt_snprintf(cmd, sizeof(cmd), "echo %s", name);
    return msh_exec(cmd, rt_strlen(cmd));
}
"""
    assert categories(source) == ["command_execution"]


def test_macro_parse_error_elsewhere_does_not_hide_a_clean_construct():
    """RT-Thread's rt_weak/varargs make tree-sitter flag whole functions."""
    source = """
rt_weak int rt_kprintf(const char *fmt, ...)
{
    static char buf[128];
    va_list args;
    va_start(args, fmt);
    rt_vsnprintf(buf, sizeof(buf) + 64, fmt, args);
    va_end(args);
    return 0;
}
"""
    assert "buffer_overflow" in categories(source)


def test_cast_around_allocator_is_still_an_allocation():
    source = """
struct queue *make(void)
{
    struct queue *q = (struct queue *)rt_malloc(sizeof(struct queue));
    q->worker = rt_thread_create("w", entry, 0, 1024, 10, 10);
    if (q->worker == RT_NULL)
    {
        return RT_NULL;
    }
    return q;
}
"""
    assert "memory_leak" in categories(source)


# --- CWE-78 ----------------------------------------------------------------

def test_literal_only_command_is_not_reported():
    source = """
void run(void)
{
    char cmd[32];
    rt_snprintf(cmd, sizeof(cmd), "echo %s", "fixed");
    msh_exec(cmd, rt_strlen(cmd));
}
"""
    assert "command_execution" not in categories(source)


def test_shell_dispatcher_given_its_own_parameter_is_not_a_build():
    """system()/msh_exec() themselves take the caller's whole command by design."""
    source = """
int system(const char *command)
{
    return msh_exec((char *)command, rt_strlen(command));
}
"""
    assert "command_execution" not in categories(source)


# --- CWE-401 ---------------------------------------------------------------

def test_error_return_before_release_leaks():
    source = """
int load(const char *path)
{
    char *buf = rt_malloc(4096);
    if (buf == RT_NULL)
        return -RT_ENOMEM;
    if (open_target(path) < 0)
        return -RT_ERROR;
    rt_free(buf);
    return 0;
}
"""
    findings = [f for f in scan(source) if f["category"] == "memory_leak"]
    assert len(findings) == 1
    assert "error return" in findings[0]["evidence"]


def test_failed_allocation_is_not_a_leak():
    for guard in ("buf == RT_NULL", "!buf"):
        source = f"""
int load(void)
{{
    char *buf = rt_malloc(4096);
    if ({guard})
        return -RT_ENOMEM;
    rt_free(buf);
    return 0;
}}
"""
        assert "memory_leak" not in categories(source), guard


def test_success_return_hands_ownership_to_the_caller():
    source = """
struct seq *open_seq(struct file *file)
{
    struct seq *seq = rt_calloc(1, sizeof(struct seq));
    if (!seq)
        return RT_NULL;
    file->data = seq;
    return seq;
}
"""
    assert "memory_leak" not in categories(source)


def test_passing_a_buffer_to_a_reader_is_not_a_hand_over():
    source = """
int copy(const char *dst)
{
    char *block = rt_malloc(4096);
    dfs_file_read(dst, block, 4096);
    if (open_target(dst) < 0)
        return -RT_ERROR;
    rt_free(block);
    return 0;
}
"""
    assert "memory_leak" in categories(source)


def test_clearing_an_owning_field_without_releasing_it():
    source = """
void create(struct mnt *m, const char *p)
{
    m->fullpath = rt_strdup(p);
}

void drop(struct mnt *m)
{
    m->fullpath = RT_NULL;
}
"""
    findings = [f for f in scan(source) if f["category"] == "memory_leak"]
    assert any("cleared without releasing" in f["evidence"] for f in findings)


def test_allocation_in_each_arm_of_a_branch_is_not_an_overwrite():
    source = """
void open_rx(struct serial *s, int dma)
{
    void *rx;
    if (dma)
    {
        rx = rt_malloc(32);
        s->rx = rx;
    }
    else
    {
        rx = rt_malloc(64);
        s->rx = rx;
    }
}
"""
    assert [f for f in scan(source) if "overwritten" in f["evidence"]] == []


# --- CWE-362 ---------------------------------------------------------------

def test_write_outside_the_critical_section():
    source = """
void release(struct sem *s)
{
    rt_base_t level = rt_spin_lock_irqsave(&(s->spinlock));
    s->waiters = 0;
    rt_spin_unlock_irqrestore(&(s->spinlock), level);
    s->value++;
}
"""
    findings = [f for f in scan(source) if f["category"] == "data_race"]
    assert any("s->value" in f["evidence"] for f in findings)


def test_function_without_a_lock_is_not_reported():
    """Initialising an object before it is published races with nothing."""
    source = """
void init(struct work *w)
{
    w->queue = RT_NULL;
    w->type = 0;
    w->flags = 0;
}
"""
    assert "data_race" not in categories(source)


def test_lock_initialisation_is_not_holding_a_lock():
    source = """
void setup(struct pool *p, void *blocks)
{
    rt_spin_lock_init(&(p->spinlock));
    p->block_list = blocks;
}
"""
    assert "data_race" not in categories(source)


def test_list_surgery_outside_the_lock():
    source = """
void urgent(struct queue *q, struct work *w)
{
    rt_list_remove(&(w->list));
    rt_list_insert_after(&q->work_list, &(w->list));
    rt_base_t level = rt_spin_lock_irqsave(&(q->spinlock));
    rt_spin_unlock_irqrestore(&(q->spinlock), level);
}
"""
    findings = [f for f in scan(source) if f["category"] == "data_race"]
    assert any("rewires shared links" in f["evidence"] for f in findings)


# --- CWE-787 ---------------------------------------------------------------

def test_bound_larger_than_the_destination():
    source = """
void log_it(const char *fmt)
{
    static char buf[64];
    rt_snprintf(buf, sizeof(buf) + 16, "%s", fmt);
}
"""
    findings = [f for f in scan(source) if f["category"] == "buffer_overflow"]
    assert any("larger than" in f["evidence"] for f in findings)


def test_length_measured_on_the_source():
    source = """
void put(char *dst, const char *src)
{
    rt_memcpy(dst, src, rt_strlen(src) + 1);
}
"""
    findings = [f for f in scan(source) if f["category"] == "buffer_overflow"]
    assert any("measured on the source" in f["evidence"] for f in findings)


def test_allocation_smaller_than_the_write():
    source = """
char *join(const char *a, const char *b)
{
    int n = strlen(a) + strlen(b) + 1;
    char *out = (char *)rt_malloc(n);
    rt_snprintf(out, strlen(a) + strlen(b) + 2, "%s/%s", a, b);
    return out;
}
"""
    findings = [f for f in scan(source) if f["category"] == "buffer_overflow"]
    assert any("may write" in f["evidence"] for f in findings)


def test_strict_greater_wrap_check_is_off_by_one():
    source = """
char take(struct fifo *f, int cap)
{
    char c = f->buffer[f->get_index];
    f->get_index += 1;
    if (f->get_index > cap) f->get_index = 0;
    return c;
}
"""
    findings = [f for f in scan(source) if f["category"] == "buffer_overflow"]
    assert any("one past the end" in f["evidence"] for f in findings)


def test_inclusive_wrap_check_is_accepted():
    source = """
char take(struct fifo *f, int cap)
{
    char c = f->buffer[f->get_index];
    f->get_index += 1;
    if (f->get_index >= cap) f->get_index = 0;
    return c;
}
"""
    assert "buffer_overflow" not in categories(source)


def test_bounded_copy_with_a_literal_source_is_quiet():
    source = """
void set(void)
{
    char name[16];
    rt_strcpy(name, "fixed");
}
"""
    assert "buffer_overflow" not in categories(source)


# --- shape of the output ---------------------------------------------------

def test_findings_carry_reviewable_evidence():
    source = """
void run(const char *name)
{
    char cmd[64];
    rt_snprintf(cmd, sizeof(cmd), "echo %s", name);
    msh_exec(cmd, rt_strlen(cmd));
}
"""
    finding = scan(source)[0]
    assert finding["rule_id"] == "cwe-78"
    assert finding["analyzer"] == "c-source"
    assert finding["line"] > 0 and finding["snippet"]
    assert finding["evidence"] and finding["recommendation"]
    assert finding["references"] == ["https://cwe.mitre.org/data/definitions/78.html"]


def test_unparsable_source_does_not_raise():
    findings, failed = scan_rtos_source("int main(void) { ", "broken.c", 1)
    assert findings == [] and failed is True
