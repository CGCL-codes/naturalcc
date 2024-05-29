import os
import openai
import transformers
from transformers import GPT2TokenizerFast, AutoTokenizer, BloomTokenizerFast
import torch
import time
import numpy as np
import os
from huggingface_hub import InferenceClient

hf_access_token = "XXX"
_MAX_TOKENS = 144
_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
_TOKENIZER_GPTJ = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
_TOKENIZER_BLOOM = BloomTokenizerFast.from_pretrained("bigscience/bloom")
_TOKENIZER_STARCODER = transformers.AutoTokenizer.from_pretrained('bigcode/starcoderbase-1b', token=hf_access_token)
_TOKENIZER_CODEGEN = transformers.AutoTokenizer.from_pretrained('Salesforce/codegen-350M-mono', token=hf_access_token)
_TOKENIZER_CODEGEN2 = transformers.AutoTokenizer.from_pretrained('Salesforce/codegen2-1B', token=hf_access_token)

GPT3_LENGTH_LIMIT = 4097
LLAMA_LENGTH_LIMIT= 4097
openai.api_key = "XXX"


def bloom_style_tokenize(x):
    return _TOKENIZER_BLOOM.decode(_TOKENIZER_BLOOM.encode(x))


def gpt_style_tokenize(x, model='gpt3'):
    if model == 'gpt3':
        return _TOKENIZER.tokenize(x)
    elif model == 'gptj':
        return torch.from_numpy(np.array(_TOKENIZER_GPTJ.encode(x))).unsqueeze(0)
    else:
        raise NotImplementedError(f'model {model} unimplemented')


def gptj_decode(x, length, stop):
    text = _TOKENIZER_GPTJ.decode(x[length:])
    text = text.split(stop)[0].replace('<|endoftext|>', '')
    return text


def starcoder_decode(x, length, stop):
    text = _TOKENIZER_STARCODER.decode(x[length:])
    text = text.split(stop)[0].replace('<|endoftext|>', '')
    return text


def codegen_decode(x, length, stop):
    text = _TOKENIZER_CODEGEN.decode(x[length:])
    text = text.split(stop)[0].replace('<|endoftext|>', '')
    return text

def codegen2_decode(x, length, stop):
    text = _TOKENIZER_CODEGEN.decode(x[length:])
    text = text.split(stop)[0].replace('<|endoftext|>', '')
    return text

def get_gptj_text_offset(token_ids, text, length):
    offset = 0
    offsets = []
    tokens = []
    for t in token_ids[length:]:
        token = _TOKENIZER_GPTJ.decode([t])
        tokens.append(token)
        offsets.append(offset)
        offset += len(token)
    return offsets, tokens


def get_starcoder_text_offset(token_ids, text, length):
    offset = 0
    offsets = []
    tokens = []
    for t in token_ids[length:]:
        token = _TOKENIZER_STARCODER.decode([t])
        tokens.append(token)
        offsets.append(offset)
        offset += len(token)
    return offsets, tokens


def get_codegen_text_offset(token_ids, text, length):
    offset = 0
    offsets = []
    tokens = []
    for t in token_ids[length:]:
        token = _TOKENIZER_CODEGEN.decode([t])
        tokens.append(token)
        offsets.append(offset)
        offset += len(token)
    return offsets, tokens

def get_codegen2_text_offset(token_ids, text, length):
    offset = 0
    offsets = []
    tokens = []
    for t in token_ids[length:]:
        token = _TOKENIZER_CODEGEN2.decode([t])
        tokens.append(token)
        offsets.append(offset)
        offset += len(token)
    return offsets, tokens

def length_of_prompt(prompt, model='gpt3'):
    if model == 'gpt3':
        return len(_TOKENIZER.tokenize(prompt)) + _MAX_TOKENS
    elif model == 'gptj':
        return len(_TOKENIZER_GPTJ.tokenize(prompt)) + _MAX_TOKENS
    else:
        raise NotImplementedError(f'model {model} unimplemented')


chat_clinet = openai.OpenAI(api_key=openai.api_key)
system_prompt = "You are an experienced software engineer tasked with a project. To guide your work, you will be presented with several demonstration examples. Please ensure that your output closely follows the patterns demonstrated in these examples."

local_model = None
local_tokenizer = None



def llama_completion(prompt, stop, temp=0.7, n=1, MAX_TOKENS=4096, model="codellama/CodeLlama-13b-hf"):
    client = InferenceClient(model=model, token=hf_access_token)
    client.headers["x-use-cache"] = "0"

    def generate_text(prompt):
        previous_err = None
        for attempt in range(5):
            try:
                output = client.text_generation(prompt, temperature=temp, max_new_tokens=MAX_TOKENS,
                                                stop_sequences=[stop])
                return output
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                previous_err = e
        raise AssertionError(f"Error: {previous_err}")

    from concurrent.futures import ThreadPoolExecutor
    import concurrent
    choices = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        results = [executor.submit(generate_text, prompt) for _ in range(n)]
        for result in concurrent.futures.as_completed(results):
            try:
                choices.append({'text': result.result().replace(stop, "")})
            except Exception as exc:
                #choices.append({'text': ""})
                print(f"Generated an exception: {exc}")
    return {'choices': choices, 'usage': {'prompt_tokens': 0}}


def safe_completion(engine, prompt, MAX_TOKENS, stop, temp=0.7, logprobs=5, n=1, num_tries=0):
    print(f"PROMPT = {prompt}")
    # time.sleep(1)

    temp = 0.7
    try:
        if (engine == 'gpt-3.5-turbo'):
            len_prompt_token = len(_TOKENIZER.tokenize(prompt))
            if MAX_TOKENS + len_prompt_token >= GPT3_LENGTH_LIMIT:
                print("OVERFLOW", MAX_TOKENS + len_prompt_token)
                return {'choices': n*[{'text': 'Input Overflow'}], 'usage': {''}}
            result = chat_clinet.chat.completions.create(model=engine,
                                                         messages=[
                                                             {"role": "system", "content": system_prompt},
                                                             {"role": "user", "content": prompt}],
                                                         max_tokens=MAX_TOKENS,
                                                         stop=stop,
                                                         temperature=temp,
                                                         logprobs=True,
                                                         top_logprobs=logprobs,
                                                         # echo=True,
                                                         n=n)
            resp = {}
            choices = []
            # process it
            for item in result.choices:
                text = item.message.content
                tokens = [i.token for i in item.logprobs.content]
                token_logprobs = [i.logprob for i in item.logprobs.content]
                current_offset = 0
                token_offsets = []
                for token in tokens:
                    token_offsets.append(current_offset)
                    current_offset += len(token)
                choices.append({'text': text, 'logprobs': {'token_logprobs': token_logprobs,
                                                           'tokens': tokens,
                                                           'text_offset': token_offsets}})

            resp = {'choices': choices, 'usage': {}}

        # online inference
        elif(engine == 'code-llama' or engine == 'code-llama-7b'):
            model = "codellama/CodeLlama-13b-hf"
            if(engine == 'code-llama-7b'):
                model = "codellama/CodeLlama-7b-hf"
            resp = llama_completion(prompt=prompt,
                                      stop=stop,
                                      temp=temp,
                                      n=n,
                                      MAX_TOKENS=MAX_TOKENS,
                                    model=model)

        else:
            raise AssertionError(f"Engine {engine} not implemented.")

    except Exception as e:
        print(f'Encountered Error {e}, trying for the {num_tries} time.')
        time.sleep(10)
        if num_tries >= 10:
            return None
        else:
            return safe_completion(engine, prompt, MAX_TOKENS, stop, temp, logprobs, \
                                   n, num_tries=num_tries + 1)
    return resp



