"""Quick smoke test — no API key needed"""
import json, urllib.request, sys

BASE = "http://127.0.0.1:7860"
OK = 0

def test(name, endpoint, body=None):
    global OK
    try:
        b = json.dumps(body).encode() if body else None
        h = {"Content-Type": "application/json"} if b else {}
        with urllib.request.urlopen(urllib.request.Request(f"{BASE}{endpoint}", data=b, headers=h)) as r:
            data = json.loads(r.read())
        status = data.get("status", "?")
        if status == "success" or status == "ok":
            print(f"  PASS  {name}")
            OK += 1
        else:
            print(f"  FAIL  {name}: {status}")
            if "log" in data:
                print(f"        {data['log'][:100]}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")

print("NaturalCC Smoke Tests")
print("=" * 50)

test("Health", "/api/health")

r = json.loads(urllib.request.urlopen(
    urllib.request.Request(f"{BASE}/api/bootstrap")).read())
print(f"  INFO  {len(r['features'])} features, {len(r['models'])} models")

# Workspace scan uses different response format (exists/counts, no status)
try:
    b = json.dumps({"project_dir": ".", "target_files": ["test/StudentManager.java"]}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"{BASE}/api/workspace/scan", data=b, headers={"Content-Type": "application/json"})) as r:
        data = json.loads(r.read())
    if data.get("exists"):
        print(f"  PASS  Workspace Scan ({data['counts']['visible_files']} files)")
        OK += 1
    else:
        print(f"  FAIL  Workspace Scan: dir not found")
except Exception as e:
    print(f"  FAIL  Workspace Scan: {e}")

test("Prompt Preview - Code Completion (Java)", "/api/prompt/preview", {
    "project_dir": ".", "target_files": ["test/StudentManager.java"],
    "instruction": "补全 addStudent 方法", "feature": "code_completion"
})

test("Prompt Preview - Code Repair", "/api/prompt/preview", {
    "project_dir": ".", "target_files": ["test/StudentManager.java"],
    "instruction": "修复除零问题", "feature": "code_repair",
    "feature_config": {"repair_type": "bug_fix"}
})

test("Prompt Preview - Code Summary", "/api/prompt/preview", {
    "project_dir": ".", "target_files": ["test/StudentManager.java"],
    "instruction": "分析代码设计缺陷", "feature": "code_summary",
    "feature_config": {"summary_scope": "targets", "detail_level": "detailed"}
})

# Command preview uses provider/api_key_label format
try:
    b = json.dumps({"project_dir": ".", "target_files": ["test/StudentManager.java"], "instruction": "补全方法", "feature": "code_completion"}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"{BASE}/api/command-preview", data=b, headers={"Content-Type": "application/json"})) as r:
        data = json.loads(r.read())
    if data.get("provider"):
        print(f"  PASS  Command Preview (provider={data['provider']})")
        OK += 1
    else:
        print(f"  FAIL  Command Preview: unexpected format")
except Exception as e:
    print(f"  FAIL  Command Preview: {e}")

print("=" * 50)
print(f"Result: {OK} tests passed")
print("Detailed guide: 测试实操指南.md")
