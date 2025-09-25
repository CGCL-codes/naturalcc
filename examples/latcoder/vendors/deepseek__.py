
import os,sys
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
    sys.path.append(os.path.abspath('.'))
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from PIL import Image
import vllm
from vllm import LLM, SamplingParams, EngineArgs
from vllm.inputs import TextPrompt
from vllm.sampling_params import (BeamSearchParams, GuidedDecodingParams,
                                  RequestOutputKind, SamplingParams)
import torch

from utils.utils import encode_image

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

CHAT_TEMPLATE="""
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {% set system_message = '' -%}
{%- endif -%}

{{ bos_token + system_message }}
{%- for message in messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif -%}

    {%- if message['role'] == 'user' -%}
        {{ '<|User|>: ' + message['content'] + '\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|Assistant|>: ' + message['content'] + eos_token + '\n' }}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {{ '<|Assistant|>: ' }}
{% endif %}
"""
__LLM = None
def init_llm(tensor_parallel_size,seed, limit_mm_per_prompt=1):
    model_path="model/deepseek-vl2"
    global __LLM
    __LLM = LLM(model=model_path,
                max_model_len=4096,
                max_num_seqs=2,
                tensor_parallel_size=tensor_parallel_size,
                hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
                limit_mm_per_prompt={"image": limit_mm_per_prompt},
                seed=seed
                )

def deepseek_vl2(prompt, texts_imgs=[],temperature=0.0, seed=0, 
                 n=1):
    assert not (n>1 and temperature < 0.0001), "If you want to sample, then set the temperature more than 0.0!"
    
    im_count =0
    for c in texts_imgs:
        if isinstance(c, Image.Image):
            im_count += 1
            

    sampling_params = SamplingParams(n=n, temperature=temperature, max_tokens=4096, repetition_penalty=1.2)
    
    conversation = message_formator(prompt, texts_imgs)
    #__LLM.llm_engine.engine_args.limit_mm_per_prompt = {"image": im_count}
    outputs = __LLM.chat(
        messages = conversation,
        sampling_params=sampling_params,
        chat_template=CHAT_TEMPLATE,
    )

    res = [outputs[0].outputs[i].text for i in range(len(outputs[0].outputs)) ]
    if n==1:
        return res[0]
    else:
        return res
    
   
def deepseek_vl2_beamsearch(system_prompt, texts_imgs=[],temperature=0.0, seed=0, 
                 n=1):
    text, image = texts_imgs

    image_placeholder = "<image>\n" * 1   # adpated to n_images
    user_prompt = f"<|User|>: {image_placeholder}{text}"
    assistant_prompt = "<|Assistant|>: "
    prompt = "\n\n".join([system_prompt, user_prompt, assistant_prompt])

    print(prompt)

    params = BeamSearchParams(beam_width= n, max_tokens= 128)
    inputs = [TextPrompt(prompt=prompt,
                         multi_modal_data={"image": image})]

    outputs = __LLM.beam_search(inputs, params)
    return outputs
    
if __name__ == "__main__":
    img1 = Image.open('tmp/11.jpg')
    img2 = Image.open('tmp/22.jpg')
    init_llm(2, 2026, 2)
    
    res = deepseek_vl2_beamsearch("You are an expert.", ["Tell what is the content in the image", img1], n=2)
    print(res)