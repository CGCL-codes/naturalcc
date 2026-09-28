import pytest

from code_agent.security_c_rules import scan_c_source


def findings(source, suffix=".c"):
    return scan_c_source(source, "module" + suffix, 1)[0]


@pytest.mark.parametrize("suffix", [".c", ".cpp"])
@pytest.mark.parametrize("body", [
    "system(input);", "popen(input, \"r\");",
    'char cmd[128]; snprintf(cmd,sizeof(cmd),"echo %s",input); system(cmd);',
    'char cmd[128]; strcpy(cmd,input); system(cmd);',
    'char *alias=input; system(alias);',
    'char *cmd=getenv("COMMAND"); system(cmd);',
    'char cmd[128]; fgets(cmd,sizeof(cmd),stdin); system(cmd);',
    'char cmd[128]; if(!fgets(cmd,sizeof(cmd),stdin))return; system(cmd);',
    'char cmd[128]; snprintf(cmd,sizeof(cmd),"%s",input); if(flag) strcpy(cmd,"date"); system(cmd);',
])
def test_command_flow_records_evidence(suffix, body):
    result = findings(f"void invoke(char *input, int flag) {{ {body} }}", suffix)
    assert len(result) == 1
    assert result[0]["analyzer_id"] == "shellStringFlow"
    assert "->" in result[0]["evidence"]


@pytest.mark.parametrize("body", [
    'system("date");', 'char *cmd="date"; system(cmd);',
    'char cmd[128]; snprintf(cmd,sizeof(cmd),"echo %d",atoi(input)); system(cmd);',
    'execl("/bin/echo","echo",input,(char *)0);',
    'char *cmd=input; cmd="date"; system(cmd);',
    '/* system(input); */ const char *s="system(input)";',
    'char cmd[128]; scanf("%d",&n); snprintf(cmd,sizeof(cmd),"echo %d",n); system(cmd);',
])
def test_command_controls_do_not_claim_shell_injection(body):
    assert findings(f"void invoke(char *input) {{ {body} }}") == []


def threaded(access="++shared;", between="", before="", after="", declaration="int shared;"):
    return f"""{declaration}
pthread_mutex_t mutex, other;
void *worker(void *arg) {{ {before} {access} {after} return arg; }}
void launch(void) {{ pthread_t a,b;
pthread_create(&a,0,worker,0); {between}
pthread_create(&b,0,worker,0);
pthread_join(a,0); pthread_join(b,0);
}}
"""


@pytest.mark.parametrize("access", ["++shared;", "shared += 1;", "shared = 42;"])
def test_overlapping_writers(access):
    result = findings(threaded(access))
    assert len(result) == 1
    assert result[0]["category"] == "data_race"
    assert "global shared" in result[0]["evidence"]


@pytest.mark.parametrize("source", [
    threaded(between="pthread_join(a,0);"),
    threaded(before="pthread_mutex_lock(&mutex);", after="pthread_mutex_unlock(&mutex);"),
    threaded(before="{pthread_mutex_lock(&mutex);}", after="pthread_mutex_unlock(&mutex);"),
    threaded(before="if(pthread_mutex_lock(&mutex)==0){", after="pthread_mutex_unlock(&mutex);}"),
    threaded(access="int local=shared;"),
    threaded(access="int shared=0; ++shared;"),
    threaded(declaration="_Atomic int shared;"),
    threaded(declaration="_Thread_local int shared;"),
    "int shared; void *worker(void *x){++shared;return x;}",
])
def test_race_controls(source):
    assert findings(source) == []


def test_main_access_after_join_is_safe_but_before_join_conflicts():
    source = "int shared; void *worker(void *x){++shared;return x;} void f(){pthread_t t; pthread_create(&t,0,worker,0); %s}"
    assert findings(source % "pthread_join(t,0); ++shared;") == []
    assert findings(source % "++shared; pthread_join(t,0);")


def test_different_mutexes_do_not_protect_shared_object():
    source = """int shared; pthread_mutex_t x,y;
void *a(void *p){pthread_mutex_lock(&x);++shared;pthread_mutex_unlock(&x);return p;}
void *b(void *p){pthread_mutex_lock(&y);++shared;pthread_mutex_unlock(&y);return p;}
void run(){pthread_t t,u;pthread_create(&t,0,a,0);pthread_create(&u,0,b,0);pthread_join(t,0);pthread_join(u,0);}
"""
    assert findings(source)


def test_cpp_threads_and_lock_guard():
    source = """int shared; std::mutex mutex;
void worker(){ %s ++shared; }
void launch(){std::thread a(worker);std::thread b(worker);a.join();b.join();}
"""
    assert findings(source % "", ".cpp")
    assert findings(source % "std::lock_guard<std::mutex> guard(mutex);", ".cpp") == []
