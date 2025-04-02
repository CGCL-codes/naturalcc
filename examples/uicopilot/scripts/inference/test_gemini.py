from datasets import Dataset, load_dataset
from tqdm import tqdm
from utils import image2md5, save_result
import io
import base64
import os
import time
import sys
import google.generativeai as genai
from PIL import Image,ImageDraw
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")

# 提取 HTML 内容的示例代码
def extract_html_from_response(response):
    try:
        if not response.candidates:
            tqdm.write("No candidates found in response.")
            return None

        # 检查 content 是否存在
        candidate = response.candidates[0]
        if 'content' not in candidate:
            tqdm.write("No content found in candidate.")
            return None

        html_content = candidate.content.parts[0].text
        if html_content.startswith("```html"):
            html_content = html_content[7:]
        if html_content.endswith("```"):
            html_content = html_content[:-3]
        return html_content.strip()
    except Exception as e:
        tqdm.write("Error extracting HTML:", e)
        return None


genai.configure(api_key=API_KEY_GEMINI)
def gemini(prompt, image, text):
    # 创建生成模型
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    # 创建请求并获取响应
    try:
        response = model.generate_content([prompt,image,text])
        html_content = extract_html_from_response(response)
        return html_content
    except Exception as e:
        tqdm.write("Error during content generation:", e)



model_name = 'gemini'
test_data_name = 'vision2ui'

result_path = f'/data02/projects/vision2ui/results/{model_name}-{test_data_name}'
data_path = f'/data02/projects/vision2ui/datasets/{test_data_name}_benchmark'

prompt_system = """
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
prompt_user = "Turn this into a single html file using tailwind."


def predict(image):
    html = gemini(prompt_system, image, prompt_user)
    return html


if test_data_name == 'vision2ui':
    ds = load_dataset('parquet', data_dir=data_path)['train']
else:
    ds = Dataset.load_from_disk(data_path)



start_index =0
end_index = 767

for index, item in tqdm(enumerate(ds), total=len(ds)):
    if index < start_index:
        continue 
    if index > end_index:
        break  
    image = item['image']
    md5 = image2md5(image)

    save_path = result_path
    if test_data_name == 'vision2ui':
        tokens = sum(item['tokens'])
        size = 'short' if tokens < 2048 else ('mid' if tokens < 4096 else 'long')
        save_path = os.path.join(result_path, size)

    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        tqdm.write(f"Created directory: {save_path}") 


    t_start = time.time()
    html = predict(image) 
    duration = time.time() - t_start
    if html is not None:
        save_result(save_path, image, item['text'] if 'text' in item else item['html'], html, duration)
        tqdm.write(f"{index}")
    else:
        tqdm.write(f"Failed for index {index}: {md5}")
    sys.stdout.flush()