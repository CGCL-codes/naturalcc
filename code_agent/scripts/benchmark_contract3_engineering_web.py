"""Optional Web evaluation: one real source file per Run, whole snapshot workspace."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from code_agent.scripts.build_contract3_engineering_v5 import DATA, REVISION, SUITES, CATEGORIES, digest, verify, write_json
from code_agent.scripts import benchmark_contract5_web as transport


def goal(file):
    return f'''审计工程中的 `{file}`，只报告数组越界 array_oob、字符串溢出 string_overflow、空指针 null_pointer。
先读取整个文件，可读取工作区内其他业务源码和头文件追踪调用关系。调用只读 vulnerability_detection，
参数 target_files:["{file}"]、scan_scope:"targets"、scan_type:"frequent_defects"、max_findings:1000。
不要修改、编译或执行源码，不申请执行/写入审批，不搜索外部资料。工具失败或解析不完整必须明确说明。
结论以源码和触发条件为依据；无告警不代表安全。不得搜索 GT、标签或注入清单。
只输出 JSON：{{"findings":[{{"category":"array_oob","line":1,"reason":"具体源码证据和触发条件"}}],"summary":"审计结论"}}。
line 必须是当前文件的问题位置；多个问题分别列出。'''


def parse_answer(answer, lines):
    value = json.loads(transport.extract_json(answer))
    if not isinstance(value, dict) or not isinstance(value.get('findings'), list) or not isinstance(value.get('summary'), str):
        raise ValueError('Missing findings or summary')
    for f in value['findings']:
        if not isinstance(f, dict) or f.get('category') not in CATEGORIES or type(f.get('line')) is not int or not 1 <= f['line'] <= lines or not isinstance(f.get('reason'), str) or not f['reason'].strip():
            raise ValueError('Invalid finding/category/line/evidence')
    return value


def classify(cases, prediction):
    # No function-name fallback: multiple operations in a real function can
    # have different labels, so a name-only match would contaminate the score.
    rows, used = [], set()
    for c in cases:
        hits = [i for i,f in enumerate(prediction['findings']) if f['category']==c['category'] and c['line_start'] <= f['line'] <= c['line_end']]
        used.update(hits)
        rows.append({**c, 'detected': bool(hits), 'matched_findings': [prediction['findings'][i] for i in hits]})
    return rows, [f for i,f in enumerate(prediction['findings']) if i not in used]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, default='balanced')
    parser.add_argument('--project', choices=['freertos_kernel','zephyr','rt_thread'], default='freertos_kernel')
    parser.add_argument('--base-url', default='http://127.0.0.1:7860')
    parser.add_argument('--model', help='Must match the Web backend runtime model exactly; no fallback')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--print-prompts', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    destination = DATA / args.suite / args.project
    gt = verify(destination)
    targets = gt['detection_files'][:args.limit] if args.limit else gt['detection_files']
    if args.print_prompts:
        for target in targets:
            print(goal(target['path']))
        print(f"{len(targets)} target files; full snapshot workspace; GT excluded")
        return
    if not args.model:
        parser.error('--model is required for an actual run or saved model report')
    out = DATA / 'web' / transport.model_slug(args.model) / args.suite / args.project
    out.mkdir(parents=True, exist_ok=True)
    with (out/'benchmark.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out/'checkpoint.json'
        checkpoint = json.loads(path.read_text()) if path.exists() else {'samples': {}}
        signature = {'revision': REVISION, 'model': args.model, 'manifest': digest((destination/'manifest.json').read_bytes()),
                     'protocol': digest(Path(__file__).read_bytes()), 'transport': digest(Path(transport.__file__).read_bytes()),
                     'product': {f: digest((ROOT/f).read_bytes()) for f in (*transport.PRODUCT_FILES, 'security_exam.py')},
                     'base_url': args.base_url, 'budget': transport.BUDGET}
        if checkpoint.get('signature', signature) != signature:
            raise ValueError('Dataset, model or protocol changed; preserve old results in a separate directory')
        checkpoint['signature'] = signature
        def persist():
            rows = [r for s in checkpoint['samples'].values() for r in s.get('rows', [])]
            checkpoint['statistics'] = {c: transport.metric([r for r in rows if r['category']==c]) for c in CATEGORIES}
            checkpoint['attempted_files'] = len(checkpoint['samples'])
            checkpoint['total_files'] = len(gt['detection_files'])
            checkpoint['completed_files'] = sum(s.get('status')=='scored' for s in checkpoint['samples'].values())
            checkpoint['scope'] = 'Only completed files and annotated operation ranges scored; unmatched findings are unlabelled, not automatic false positives.'
            transport.save_json(path, checkpoint)
        if args.report_only:
            persist()
            print(json.dumps(checkpoint.get('statistics'), ensure_ascii=False, indent=2))
            return
        if not os.environ.get('OPENROUTER_API_KEY'):
            raise ValueError('OPENROUTER_API_KEY is required')
        client = transport.WebClient(args.base_url)
        config = client.call('GET','/api/bootstrap')['runtime_default_model_config']
        if config.get('provider') != 'openrouter' or config.get('model') != args.model or config.get('fallback_models') or config.get('base_url','').rstrip('/') != 'https://openrouter.ai/api/v1':
            raise ValueError('Backend model/provider does not match requested OpenRouter model without fallback')
        for target in targets:
            file = target['path']
            record = checkpoint['samples'].setdefault(file, {'attempts': []})
            if record.get('status') == 'scored' or (record.get('status') in ('failed','unscored') and not args.retry_failed):
                continue
            thread = client.call('POST','/api/agent/threads',json={'title':f'合同3工程审计 {args.project}/{file}', 'workspace': str((destination/'project').resolve()), 'runtime_mode':'agent','runtime_model_config':config,'budget':transport.BUDGET})
            attempt = {'thread_id':thread['thread_id']}
            record['attempts'].append(attempt)
            persist()
            run = client.call('POST',f"/api/agent/threads/{thread['thread_id']}/messages",json={'content':goal(file),'target_files':[file],'budget':transport.BUDGET,'api_key':os.environ['OPENROUTER_API_KEY']})
            attempt['run_id'] = run['run_id']
            run_path = f"/api/agent/runs/{run['run_id']}"
            while True:
                state = client.call('GET',run_path)
                if state.get('runtime_model_config') != config:
                    raise ValueError('Run model changed')
                events = client.call('GET',run_path+'/events')['events']
                transport.save_json(out/'runs'/f"{run['run_id']}.json", {'state':state,'events':events})
                record['status'] = state['status']
                persist()
                if state['status'] in transport.TERMINAL:
                    break
                if state['status']=='paused':
                    raise RuntimeError('Run paused; resume explicitly through the Web backend')
                client.call('POST',run_path+('/reject' if state['status']=='waiting_approval' else '/step'))
            if state['status']=='completed':
                try:
                    prediction = parse_answer(state['final_answer'], target['lines'])
                    rows, unmatched = classify([c for c in gt['cases'] if c['file']==file], prediction)
                    record.update(status='scored',prediction=prediction,rows=rows,unlabelled_findings=unmatched)
                except ValueError as error:
                    record.update(status='unscored',error=str(error))
            else:
                record['status']='failed'
            persist()
            print(file,record['status'],flush=True)
            if transport.credit_failure(events):
                break


if __name__=='__main__':
    main()
