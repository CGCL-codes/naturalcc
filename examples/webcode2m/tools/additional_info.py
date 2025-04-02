from scripts.data_cc_pipeline.post_process import detect_lang
from PIL import  Image
from io import BytesIO
from datasets import load_from_disk,concatenate_datasets,Image

import re

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")


def split_html_css(text):
    html_content = re.sub(f'\s+', ' ', text)

    # 正则表达式匹配<style>标签及其内容
    style_tag_pattern = r'<style[^>]*>([\s\S]*?)<\/style>'

    # 使用re.DOTALL使得.可以匹配包括换行符在内的任意字符
    # re.IGNORECASE忽略大小写
    style_content = re.search(style_tag_pattern, html_content, re.DOTALL | re.IGNORECASE)

    # 提取<style>标签内的CSS内容
    if style_content:
        css_content = style_content.group(1).strip()
    else:
        css_content = ''

    # 移除HTML中的<style>标签及其内容
    clean_html = re.sub(style_tag_pattern, '', html_content, flags=re.DOTALL | re.IGNORECASE)

    return  [css_content,clean_html]


def func(items): 
    try:
        items["scale"]=[item.size for item in items['image']]    
        items["lang"]=[i[0]["label"] for i in detect_lang(items["text"], device="cuda:7")]
        contents = [  split_html_css(text) for text in items["text"]]
        items["tokens"] = [ list(map(len,tokenizer(con,  max_length=10240, truncation=True)["input_ids"]))  for con in contents]        
        return items    
    except Exception as e:
        print(e)
        return None
    
ds_path="/data02/starmage/datasets/cc/arrows_15_processed"
ds = load_from_disk(ds_path)
ds=ds.cast_column("image", Image(decode=True))
ds = ds.map(func, num_proc=32, batched=True, batch_size=128)
ds.save_to_disk("/data02/starmage/datasets/cc/tmp", num_proc=16)