import sys
sys.path.append("/data02/users/lz/code/UICoder")
import io
import base64
from PIL import Image
import google.generativeai as genai
import os
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")
# 提取 HTML 内容的示例代码
def extract_html_from_response(response):
    try:
        html_content = response.candidates[0].content.parts[0].text
        # 移除markdown代码块标记
        if html_content.startswith("```html"):
            html_content = html_content[7:]
        if html_content.endswith("```"):
            html_content = html_content[:-3]
        return html_content.strip()
    except Exception as e:
        print("Error extracting HTML:", e)
        return None

genai.configure(api_key=API_KEY_GEMINI)
def gemini(prompt, image, text):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = buffered.getvalue()
    image_base64 = base64.b64encode(img_str).decode('utf-8')
    # 创建生成模型
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    # 创建请求并获取响应
    try:
        response = model.generate_content([prompt,image_base64,text])
        html_content = extract_html_from_response(response)
        # return html_content
        return response
    except Exception as e:
        print("Error during content generation:", e)

# 运行测试函数
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
text = "Turn this into a single html file using tailwind."

response = gemini(prompt, test_image, text)
print(response)
