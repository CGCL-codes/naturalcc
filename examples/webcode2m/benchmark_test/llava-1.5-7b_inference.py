from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import Dataset, load_dataset
from tqdm import tqdm
from utils import *
import time

model_name = 'llava-1.5-7b'
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

model_id = "/data02/users/lz/code/tmp/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    # torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to('cuda')

processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": prompt_system},
        ],
    },
    {
      "role": "user",
      "content": [
          {"type": "text", "text": prompt_user},
          {"type": "image"},
        ],
    },
]

def predict(image):
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors='pt').to('cuda')

    output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    text = processor.decode(output[0], skip_special_tokens=True)
    html = text.split('<!DOCTYPE html>')
    if len(html) > 1:
        html = html[1].split('```')[0]
    else:
        html = html[0].split('```')[0]

    return html

if test_data_name == 'vision2ui':
    ds = load_dataset('parquet',data_dir=data_path)['train']
else:
    ds = Dataset.load_from_disk(data_path)

for item in tqdm(ds):
    image = item['image']
    md5 = image2md5(image)
    save_path = result_path
    if test_data_name == 'vision2ui':
        tokens = sum(item['tokens'])
        size = 'short' if tokens<2048 else ('mid' if tokens<4096 else 'long')
        save_path = os.path.join(result_path,size)

    if os.path.exists(os.path.join(save_path,md5)):
        continue

    t_start = time.time()
    html = predict(image)
    
    if 'Failed' not in html:
        duration = time.time()-t_start
        save_result(save_path, image, item['text'] if 'text' in item else item['html'], html, duration)



