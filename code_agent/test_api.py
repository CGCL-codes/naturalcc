import os
import requests

API_KEY = os.getenv("OPENROUTER_API_KEY", "你的_openrouter_key")
print(f"正在测试 OpenRouter API: {API_KEY} 可用性...")

MODEL = "openai/gpt-4o-mini"

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",

    # 可选，不写也通常能测通
    "HTTP-Referer": "https://example.com",
    "X-Title": "openrouter-test",
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "请只回复：ok"}
    ]
}

try:
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    print("status_code:", resp.status_code)
    print("response:")
    print(resp.text)

    if resp.ok:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print("\nAPI 可用，模型返回：", content)
    else:
        print("\nAPI 调用失败，请检查 key、模型名、额度或返回报错信息。")

except requests.exceptions.RequestException as e:
    print("请求异常：", e)