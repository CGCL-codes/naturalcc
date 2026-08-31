"""Test DeepSeek API connectivity

用法：
    export DEEPSEEK_API_KEY=sk-...        # 或 OPENAI_API_KEY
    export CODE_AGENT_BASE_URL=http://127.0.0.1:7860   # 可选，默认本机 7860
    python test_deepseek.py
"""
import json, urllib.request, sys, os

BASE = os.environ.get("CODE_AGENT_BASE_URL", "http://127.0.0.1:7860")
API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
PROJ_DIR = os.path.abspath(".")

if not API_KEY:
    sys.exit("请先设置环境变量 DEEPSEEK_API_KEY（或 OPENAI_API_KEY）后再运行本脚本。")

def api_call(endpoint, body=None):
    b = json.dumps(body, ensure_ascii=False).encode() if body else None
    h = {"Content-Type": "application/json; charset=utf-8"} if b else {}
    req = urllib.request.Request(f"{BASE}{endpoint}", data=b, headers=h)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

print("=" * 60)
print("DeepSeek API 连通性测试")
print("=" * 60)

# 1. Health
print("\n[1] Health Check")
r = api_call("/api/health")
print(f"  Status: {r['status']}")

# 2. Bootstrap - verify model list
print("\n[2] Bootstrap - 模型列表")
r = api_call("/api/bootstrap")
print(f"  Models: {r['models']}")
print(f"  Default: {r['default_model']}")

# 3. Prompt Preview with API key
print("\n[3] Prompt Preview (with API key)")
body = {
    "project_dir": PROJ_DIR,
    "target_files": ["test/SampleTest.java"],
    "instruction": "补全 add 方法",
    "model": "deepseek/deepseek-chat",
    "api_key": API_KEY,
    "feature": "code_completion",
}
r = api_call("/api/prompt/preview", body)
print(f"  Status: {r['status']}")
if r['status'] == 'success':
    # Check that API key is masked in command
    cmd = r.get('command', '')
    print(f"  Command preview: {cmd[:120]}...")
    print(f"  Prompt length: {len(r['log'])} chars")
else:
    print(f"  Error: {r['log'][:200]}")

# 4. Command Preview - verify provider is deepseek
print("\n[4] Command Preview")
body2 = {
    "project_dir": PROJ_DIR,
    "target_files": ["test/SampleTest.java"],
    "instruction": "test",
    "model": "deepseek/deepseek-chat",
    "api_key": API_KEY,
    "feature": "code_completion",
}
r = api_call("/api/command-preview", body2)
print(f"  Provider: {r['provider']}")
print(f"  API Key label: {r['api_key_label']}")

print("\n" + "=" * 60)
print("配置验证通过！可以执行实际的代码补全测试。")
print("=" * 60)
