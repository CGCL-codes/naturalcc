from code_agent.scripts import build_contract3_juliet as juliet

SOURCE = '''#include "std_testcase.h"
#ifndef OMITBAD
void CWE476_x_01_bad() { /* FLAW */ int *p = NULL; printIntLine(*p); }
#endif
#ifndef OMITGOOD
static void goodG2B() { int v = 1; int *p = &v; printIntLine(*p); } // FIX
#ifdef _WIN32
#define SEP "\\\\"
#endif
void CWE476_x_01_good() { goodG2B(); }
#endif
#ifdef INCLUDEMAIN
int main() { CWE476_x_01_good(); }
#endif
'''


def test_resolve_keeps_only_requested_path_and_unrelated_directives():
    bad = juliet.resolve(SOURCE, {"OMITGOOD"})
    good = juliet.resolve(SOURCE, {"OMITBAD"})
    assert "_bad" in bad and "goodG2B" not in bad and "main" not in bad
    assert "_bad" not in good and "goodG2B" in good and "#ifdef _WIN32" in good and "main" not in good


def test_strip_comments_respects_string_literals():
    text = juliet.strip_comments('char *s = "/* keep */"; // drop\n/* drop */ int x;\n')
    assert '"/* keep */"' in text and "drop" not in text


def test_build_sample_removes_label_words(tmp_path):
    path = tmp_path / "CWE476_x_01.c"
    path.write_text(SOURCE)
    case = {"testcase": "CWE476_x_01", "files": [path]}
    for vulnerable in (True, False):
        files = juliet.build_sample(case, vulnerable)
        assert list(files) == ["src_a.c"]
        assert not juliet.LABEL_WORD.search(files["src_a.c"])
        assert "FLAW" not in files["src_a.c"] and "FIX" not in files["src_a.c"]
