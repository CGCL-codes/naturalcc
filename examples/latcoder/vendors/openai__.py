from openai import OpenAI, AzureOpenAI
from utils.utils import encode_image
import json
import io
from PIL import Image
import torch

def get_client(model=''):
    client = AzureOpenAI(
        azure_endpoint = "xx", 
        api_key="xxx", 
        api_version="2024-02-01"
    )
    
    return client

def message_formator(prompt, texts_imgs):
    """
     prompt  texts_imgs  JSON 。

    ：
    - prompt: ，。
    - texts_imgs: ， [, ...]， PIL.Image 。

    ：
    -  JSON ，。
    """
    return [
        {
            "role": "system",
            "content": [
                {
                    'type': 'text',
                    'text': prompt
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    'type': 'image_url',
                    'image_url': {"url": f"data:image/png;base64,{encode_image(c)}"}
                } if isinstance(c, Image.Image) else
                {
                    'type': 'text',
                    'text': c
                }
                for c in texts_imgs
            ],
        },
    ]

def gpt4o(prompt, texts_imgs=[],temperature=0.0, seed=0, n=1):
    client = get_client()
    prompt_messages = message_formator(prompt, texts_imgs)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=prompt_messages,
            temperature=temperature,
            seed=seed
        )
        message_content = response.choices[0].message.content.strip()
    except Exception as e:
        print("Error during content generation:", e)
        message_content = f"Error: {e}."
        
    return message_content


def batch_item(id, prompt, text, imgs=[], model='gpt-4o'):
    prompt_messages = message_formator(prompt, text, imgs)
    
    item = {"custom_id": str(id), "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {"model": model, 
                    "messages": prompt_messages,"max_tokens": 8000}
            }
    
    return json.dumps(item)


# https://platform.openai.com/docs/guides/batch
def gtp4o_batch(item_generator): # item_generator: should return (id, prompt, text, imgs)
    client = get_client()
    
    jsonl_file = io.BytesIO()
    for id, prompt, text, imgs in item_generator():
        item:str = batch_item(id, prompt, text, imgs)
        jsonl_file.write(item.encode('utf-8')+b'\n')
    jsonl_file.seek(0)
    
    batch_input_file = client.files.create(
        file=jsonl_file,
        purpose="batch"
    )
    print(batch_input_file)
    
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            
        }
    )
    
    batch = client.batches.retrieve("batch_abc123")
    print(batch)
    
    file_response = client.files.content("file-xyz123")
    print(file_response.text)