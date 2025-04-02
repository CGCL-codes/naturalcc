import sys
sys.path.append("/data02/users/lz/code/UICoder")
import cv2
import io
import base64
import requests
import anthropic
import os
from PIL import Image
# 设置代理环境变量
os.environ["HTTP_PROXY"] = "http://127.0.0.1:27890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:27890"
API_KEY_CLAUDE = os.getenv("API_KEY_CLAUDE")
ENDPOINT_CLAUDE = os.getenv("ENDPOINT_CLAUDE")

def claude(prompt, image, text):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = buffered.getvalue()
    image_base64 = base64.b64encode(img_str).decode('utf-8')
    image_media_type="image/png"
    # Prepare the content list for the API
    user_message=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_base64,
                    },
                },
                {
                    "type": "text",
                    "text": text
                }
            ],
        },
    ]
    client = anthropic.Anthropic(api_key=API_KEY_CLAUDE)

    # Send the request using the anthropic library
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        system=prompt,  
        messages=user_message
    )
    # 提取并返回 HTML 内容
    if message and isinstance(message.content, list) and len(message.content) > 0:
        html_content = message.content[0].text
        return html_content.strip()  # 返回去掉首尾空白的 HTML 内容
    return "<html><body><p>Error occurred</p></body></html>"  # 出错时返回默认的 HTML

# # 运行 claude 测试函数
# # 使用任意图片和文本进行测试

prompt = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps 
using Tailwind, HTML and JS.

- Make sure the app looks exactly like the screenshot.
- Make sure the app has the same page layout like the screenshot, i.e., the gereated html elements should be at the same place with the correspondingpart in the screenshot and the generated  html containers should have the same hierachy structure as the screenshot.
- Pay close attention to background color, text color, font size, font family, 
padding, margin, border, etc. Match the colors and sizes exactly.
- Use the exact text from the screenshot.
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writingthe full code. WRITE THE FULL CODE.
- Repeat elements as needed to match the screenshot. For example, if there are 15 items, the code should have 15 items. DO NOT LEAVE comments like"<!-- Repeat for each news item -->" or bad things will happen.
- For images, use placeholder images from https://placehold.co and include a detailed description of the image in the alt text so that an imagegeneration AI can generate the image later.

In terms of libraries,

- Use this script to include Tailwind: <script src="https://cdn.tailwindcss.com"></script>
- You can use Google Fonts
- Font Awesome for icons: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

Return only the full code in <html></html> tags.
Do not include markdown "```" or "```html" at the start or end.
"""
test_image = Image.new("RGB", (100, 100), color="blue")  # 创建一个蓝色测试图像
# text = "Turn this into a single html file using tailwind."
text = "Turn the follow image into a single html file using tailwind."
response = claude(prompt, test_image, text)
print(response)

