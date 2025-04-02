import requests
import base64
import time
import io
import base64
from PIL import Image


GPT4V_CONFIGS = [
    [
        "bdc4a955791541fd850a0be390e6e91a",
        "https://gpt4v-test-rp.openai.azure.com/",
        "IMG-TXT",
    ],
    [
        "180951b6849245b2b9068e6f9123e20b",
        "https://gpt4v-test-rp4.openai.azure.com/",
        "4v1",
    ],
    [
        "a81a70bd551746ae9ca49b8be61847e3",
        "https://gpt4v-test-rp2.openai.azure.com/",
        "4v2",
    ],
    [
        "f50c0d72545c46f8b97b07caa70f45fd",
        "https://gpt4v-test-rp3.openai.azure.com/",
        "4V3",
    ],
]

ERROR_KEY = "GPT4VError"  # 可以根据此字符串是否在结果中判断是否有错误发生


# gpt4v图像理解
def request(
    imgpath,
    prompt_sys,
    prompt_user,
    timeout=30,
    max_tokens=512,
    apikey=0,
    compress_rate=95,
):
    try:
        img = Image.open(imgpath)

        """ 缩略会导致输出结构差别较大，暂时不使用
        width, height = img.size                
        if width > 1024 or height > 1024:
            max_size = (1024,1024)
            img.thumbnail(max_size, Image.ANTIALIAS)   
        """

        buffered = io.BytesIO()
        # img.save("/home/starmage/projects/uicoder/output/F5FFC91C-2C64-4DB8-9008-F040CF1F2146.png", format=img.format, quality=compress_rate)
        img.save(
            buffered, format=img.format, quality=compress_rate
        )  # quality 参数的范围是 1（最差质量）到 95（最佳质量）
        encoded_image = base64.b64encode(buffered.getvalue()).decode("ascii")
    except Exception as e:
        return f"Failed to make the request. {ERROR_KEY}: {e}"
    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_CONFIGS[apikey][0],
    }

    # Payload for the request
    payload = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": prompt_sys}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                    {
                        "type": "text",
                        "text": prompt_user,
                    },
                ],
            },
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }

    # GPT4V_ENDPOINT = "https://gpt4v-test-rp.openai.azure.com/openai/deployments/IMG-TXT/chat/completions?api-version=2023-07-01-preview"
    GPT4V_ENDPOINT = f"{GPT4V_CONFIGS[apikey][1]}/openai/deployments/{GPT4V_CONFIGS[apikey][2]}/chat/completions?api-version=2023-07-01-preview"

    # Send request
    try:
        response = requests.post(
            GPT4V_ENDPOINT, headers=headers, json=payload, timeout=timeout
        )
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        # Handle the response as needed (e.g., print or process)
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return content
    except requests.RequestException as e:
        return f"Failed to make the request. {ERROR_KEY}: {e}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, required=True)
    parser.add_argument(
        "-p1",
        "--prompt_system",
        type=str,
        default="You are an ai assitant to help people find information.",
    )
    parser.add_argument(
        "-p2",
        "--prompt_user",
        type=str,
        default="Give some description about the input image.",
    )
    parser.add_argument("-t", "--timeout", type=int, default=30)
    parser.add_argument("-c", "--compress_rate", type=int, default=95)
    parser.add_argument("-m", "--max_tokens", type=int, default=512)
    parser.add_argument("-k", "--gpt4v_key", type=int, default=0, choices=[0, 1, 2, 3])

    args = parser.parse_args()

    content = request(
        args.image_path,
        args.prompt_system,
        args.prompt_user,
        args.timeout,
        args.max_tokens,
        args.gpt4v_key,
        args.compress_rate,
    )
    print(content)
