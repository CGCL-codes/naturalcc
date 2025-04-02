from datasets import Dataset, load_dataset
from tqdm import tqdm
from agents.utils.gpt4o import gpt4o
from utils import image2md5, save_result
import os
import time
import sys
import time
import math
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from multiprocessing import Process, Pool

model_name = 'llava-onevision-72b'
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

model_id = "/data02/models/llava-onevision-qwen2-72b-ov-hf"
torch_type = torch.bfloat16
def load_model(device):
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_type, 
        device_map=device
    )
    return model, processor

def llava(model, processor, system, image, prompt):
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system}
            ]
        },
        {

          "role": "user",
          "content": [
              {"type": "image"},
              {"type": "text", "text": prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors='pt').to(next(model.parameters()).device, torch_type)

    output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    output = processor.decode(output[0][2:], skip_special_tokens=True)
    output = output.split('assistant')[1].strip()

    if '```html' in output:
        output = output.split('```html')[1]
        output = output.split('```')[0]

    return output

def predict(model, processor, image, retries=5, retry_delay=2, index=None):
    for attempt in range(retries):
        try:
            html = llava(model, processor, prompt_system, image, prompt_user)
            if isinstance(html, dict):  # 检查返回值是否为字典
                # 假设字典中有一个 'html_content' 键
                html = html.get('html_content', "<html><body><p>Empty HTML returned</p></body></html>")
            return html
        except Exception as e:
            tqdm.write(f"Attempt {attempt + 1} failed for index {index}: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                tqdm.write(f"Failed after multiple attempts for index {index}")
                return "<html><body><p>Failed after multiple attempts</p></body></html>"

def run_a_batch(ds, device='auto'):
    model, processor = load_model(device)
    for index, item in tqdm(enumerate(ds), total=len(ds)):
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
        
        if os.path.exists(f'{save_path}/{md5}'):
            continue

        t_start = time.time()
        html = predict(model, processor, image, index=index)  # 传递 index 以便记录失败的索引

        # 检查生成的 HTML 是否有效
        max_attempts = 5  # 设置最大尝试次数
        attempt = 0
        while attempt < max_attempts:
            if 'Failed' not in html and isinstance(html, str):
                duration = time.time() - t_start
                tqdm.write(f"Processed index {index}: {md5}, Duration: {duration:.2f} seconds")
                save_result(save_path, image, item['text'] if 'text' in item else item['html'], html, duration)
                break  # 成功后退出循环
            else:
                attempt += 1
                tqdm.write(f"Attempt {attempt} failed to generate HTML for index {index}: {md5}")

                if attempt < max_attempts:  # 只有在未达到最大尝试次数时才重新请求
                    html = predict(model, processor, image, index=index)  # 重新请求
                else:
                    tqdm.write(f"Failed after {max_attempts} attempts for index {index}: {md5}")
        sys.stdout.flush()

if __name__ == '__main__':
    if test_data_name == 'vision2ui':
        ds = load_dataset('parquet', data_dir=data_path)['train']
    else:
        ds = Dataset.load_from_disk(data_path)

    # devices = [2,3,4,5,6,7]

    # with Pool(len(devices)) as p:
    #     params = []
    #     batch_size = math.ceil(len(ds)/len(devices))
    #     for idx, device in enumerate(devices):
    #         params.append((ds.select(range(idx*batch_size,(idx+1)*batch_size)), f'cuda:{device}'))
    #     p.starmap(run_a_batch, params)
    # run_a_batch(ds, 'cuda:7')
    run_a_batch(ds, 'auto')