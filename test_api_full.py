"""
NaturalCC API 完整功能测试脚本
启动方式: python test_api_full.py
前提: API 服务已启动在 http://127.0.0.1:7860
"""
import json
import os
import urllib.request

BASE_URL = "http://127.0.0.1:7860"
PROJ_DIR = os.path.abspath(".")

def api_call(endpoint, data=None):
    url = f"{BASE_URL}{endpoint}"
    body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"} if body else {})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def test_all():
    results = []

    # ── 1. Health ──
    print("=" * 60)
    print("NaturalCC API 功能测试")
    print("=" * 60)
    print("\n[1/6] Health Check")
    r = api_call("/api/health")
    print(f"  Status: {r['status']} ✓")
    results.append(("Health", r['status'] == 'ok'))

    # ── 2. Bootstrap ──
    print("\n[2/6] Bootstrap (插件 & 模型)")
    r = api_call("/api/bootstrap")
    print(f"  模型: {len(r['models'])} 个")
    print(f"  插件: {len(r['features'])} 个")
    for f in r["features"]:
        print(f"    • {f['name']} [{f['execution_mode']}]")
    results.append(("Bootstrap", len(r['features']) >= 6))

    # ── 3. Workspace Scan ──
    print("\n[3/6] Workspace Scan")
    r = api_call("/api/workspace/scan", {
        "project_dir": PROJ_DIR,
        "target_files": ["test/SampleTest.java"]
    })
    print(f"  项目目录: {r['exists']}")
    print(f"  可见文件: {r['counts']['visible_files']}")
    print(f"  主目标: {r['primary_target']}")
    results.append(("Workspace", r['exists']))

    # ── 4. Prompt Preview (Java - 不需要 clang) ──
    print("\n[4/6] Prompt Preview (Java 代码补全)")
    r = api_call("/api/prompt/preview", {
        "project_dir": PROJ_DIR,
        "target_files": ["test/SampleTest.java"],
        "instruction": "补全 add 方法的功能",
        "feature": "code_completion"
    })
    ok = r['status'] == 'success'
    print(f"  状态: {'✓ success' if ok else '✗ ' + r['log'][:100]}")
    results.append(("Prompt Preview (Java)", ok))

    # ── 5. Code Repair Preview ──
    print("\n[5/6] Code Repair Preview")
    r = api_call("/api/prompt/preview", {
        "project_dir": PROJ_DIR,
        "target_files": ["test/SampleTest.java"],
        "instruction": "修复 add 方法的溢出问题",
        "feature": "code_repair",
        "feature_config": {"repair_type": "bug_fix"}
    })
    ok = r['status'] == 'success'
    print(f"  状态: {'✓ success' if ok else '✗ ' + r['log'][:100]}")
    results.append(("Code Repair", ok))

    # ── 6. Vulnerability Detection ──
    print("\n[6/6] Vulnerability Detection Preview")
    r = api_call("/api/prompt/preview", {
        "project_dir": PROJ_DIR,
        "target_files": ["test/vuln_demo.c"],
        "instruction": "扫描安全漏洞",
        "feature": "vulnerability_detection",
        "feature_config": {"scan_scope": "targets"}
    })
    ok = r['status'] == 'success'
    print(f"  状态: {'✓ success' if ok else '✗ ' + r['log'][:100]}")
    results.append(("Vulnerability Detection", ok))

    # ── Summary ──
    print("\n" + "=" * 60)
    passed = sum(1 for _, p in results if p)
    print(f"测试结果: {passed}/{len(results)} 通过")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n浏览器访问: {BASE_URL}")
    print("=" * 60)
    return passed == len(results)

if __name__ == "__main__":
    try:
        test_all()
    except urllib.error.URLError as e:
        print(f"\n❌ 无法连接 API 服务: {e}")
        print("请先启动服务: python -m code_agent.agent_web_api --host 127.0.0.1 --port 7860")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
