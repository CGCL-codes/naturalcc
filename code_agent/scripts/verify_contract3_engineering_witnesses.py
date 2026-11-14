"""Execute selected shipped function bodies in minimal ASan/UBSan witness harnesses.

This validates six representative mutations and their controls, not a booted OS,
not every site, and not the static detector's performance. Bodies are extracted
from the frozen engineering snapshots; harnesses remain outside Agent workspaces.
"""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts.build_contract3_engineering import DATA, REVISION, SUITES, digest, verify, write_json

COMMON = '#include <stddef.h>\n#include <stdint.h>\n#include <string.h>\n#include <assert.h>\n#include <errno.h>\n'
WITNESSES = [
    ('freertos_kernel','array_oob-02', '''
typedef unsigned UBaseType_t;
typedef void *QueueHandle_t;
#define configQUEUE_REGISTRY_SIZE 2
#define configASSERT(x) assert(x)
#define traceENTER_pcQueueGetName(...)
#define traceRETURN_pcQueueGetName(...)
#define mtCOVERAGE_TEST_MARKER()
static struct {const char *pcQueueName; QueueHandle_t xHandle;} xQueueRegistry[2];
''', 'int main(void) { int queue; (void)pcQueueGetName(&queue); return 0; }'),
    ('freertos_kernel','string_overflow-03', '#define configMAX_TASK_NAME_LEN 8\n',
     'int main(void) {char buffer[8]; (void)prvWriteNameToBuffer(buffer, "A"); return 0;}'),
    ('zephyr','string_overflow-04', 'static char *utf8_trunc(char *s) {return s;}\n',
     'int main(void) {char buffer[4]; utf8_lcpy(buffer, "ABCDE", sizeof(buffer)); return 0;}'),
    ('zephyr','null_pointer-05', '''
typedef int k_spinlock_key_t;
struct k_spinlock {int unused;};
struct k_mem_slab {struct k_spinlock lock; struct {size_t num_used, block_size, num_blocks, max_used;} info;};
struct sys_memory_stats {size_t allocated_bytes, free_bytes, max_allocated_bytes;};
static int k_spin_lock(struct k_spinlock *lock) {(void)lock; return 0;}
static void k_spin_unlock(struct k_spinlock *lock, int key) {(void)lock; (void)key;}
''', 'int main(void) {struct k_mem_slab slab = {0}; (void)k_mem_slab_runtime_stats_get(&slab, NULL); return 0;}'),
    ('rt_thread','array_oob-03', '''
#define rt_inline static inline
#define RT_TIMER_SKIP_LIST_LEVEL 2
typedef struct list {struct list *next, *prev;} rt_list_t;
struct rt_timer {rt_list_t row[RT_TIMER_SKIP_LIST_LEVEL];};
typedef struct rt_timer *rt_timer_t;
static void rt_list_remove(rt_list_t *n) {n->next->prev = n->prev; n->prev->next = n->next; n->next = n->prev = n;}
''', 'int main(void) {struct rt_timer timer; for(int i=0;i<2;i++) timer.row[i].next=timer.row[i].prev=&timer.row[i]; _timer_remove(&timer); return 0;}'),
    ('rt_thread','string_overflow-03', '''
#define RT_NULL NULL
#define RT_EINVAL EINVAL
#define RT_EOK 0
#define rt_strncpy strncpy
#define rt_strcpy strcpy
typedef int rt_err_t;
typedef uint8_t rt_uint8_t;
typedef size_t rt_size_t;
struct object {char name[8];};
typedef struct object *rt_object_t;
''', 'int main(void) {struct object o = {"ABCDEFG"}; char buffer[3]; (void)rt_object_get_name(&o,buffer,sizeof(buffer)); return 0;}'),
]


def main():
    records = []
    with tempfile.TemporaryDirectory(prefix='c3-engineering-witness-') as temporary:
        directory = Path(temporary)
        for project, site_id, support, entry in WITNESSES:
            for expected in ('control','defect'):
                for suite in SUITES:
                    destination = DATA / suite / project
                    gt = verify(destination)
                    case = next(c for c in gt['cases'] if c['site_id']==site_id)
                    if case['expected']==expected:
                        break
                source_path = destination/'project'/case['file']
                lines = source_path.read_text().splitlines()
                start,end = case['upstream_function_range']
                body = '\n'.join(lines[start-1:end])
                source = COMMON + support + '\n' + body + '\n' + entry + '\n'
                cfile, binary = directory/'witness.c', directory/'witness'
                cfile.write_text(source)
                build = subprocess.run(['gcc','-O0','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer',str(cfile),'-o',str(binary)],capture_output=True,text=True,timeout=30)
                if build.returncode:
                    raise RuntimeError(build.stderr)
                run = subprocess.run([str(binary)],capture_output=True,text=True,timeout=10,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'})
                diagnosed = 'ERROR: AddressSanitizer' in run.stderr or 'runtime error:' in run.stderr
                valid = run.returncode == 0 if expected=='control' else run.returncode != 0 and diagnosed
                records.append({'project':project,'suite':suite,'site_id':site_id,'expected':expected,
                                'source_sha256':digest(source_path.read_bytes()), 'function_sha256':digest(body.encode()),
                                'harness_sha256':digest(source.encode()), 'exit_code':run.returncode,
                                'sanitizer_diagnostic':diagnosed,'valid':valid, 'diagnostic':run.stderr[:2000]})
                print(project,site_id,expected,'OK' if valid else 'FAIL',flush=True)
    write_json(DATA/'label-witnesses.json',{'dataset_revision':REVISION,'method':__doc__,
        'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
        'passed':sum(r['valid'] for r in records),'total':len(records),'records':records})
    if not all(r['valid'] for r in records):
        raise SystemExit('Witness validation failed; inspect label-witnesses.json')


if __name__=='__main__':
    main()
