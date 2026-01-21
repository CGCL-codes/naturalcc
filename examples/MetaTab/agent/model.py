import openai
import os
import time
import tiktoken
import timeout_decorator
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Union
from typing import List

import fire

from llama import Llama
import sys
sys.path.append("/home/ligen/lg")
import torch
import torch.distributed as dist
def setup_distributed():
    # 设置分布式环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"  # 默认端口
    os.environ["WORLD_SIZE"] = "1"  # 总进程数
    os.environ["RANK"] = "0"  # 当前进程的 rank
    os.environ["LOCAL_RANK"] = "0"  # 当前节点的 rank

    # 初始化分布式后端
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

def cleanup_distributed():
    # 清理分布式环境
    dist.destroy_process_group()

class Model:
    def __init__(self, model_name: str, provider: str = 'openai'):
        self.model_name = model_name
        self.provider = provider  # 'openai' or 'huggingface'
        if provider == 'huggingface':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif provider == 'openai':
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            
            API_KEY = os.getenv("OPENAI_API_KEY", None)
            
            if API_KEY is None:
                raise ValueError("OPENAI_API_KEY not set, please run `export OPENAI_API_KEY=<your key>` to ser it")
            else:
                openai.api_key = API_KEY
                
        elif provider == "vllm":
            from vllm import LLM
            self.model = LLM(model_name, gpu_memory_utilization=0.9)
            self.tokenizer = self.model.get_tokenizer()
        elif provider == "llama3":
            #local_path = '/home/ligen/lg/tablellm/Meta-Llama-3-8B-Instruct/tokenizer.model'
            #local_path = '/home/ligen/lg/tablellm/Meta-Llama-3-8B-Instruct/'
            #self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.tokenizer = tiktoken.encoding_for_model(model_name)


    def query(self, prompt: str, **kwargs) -> Union[str, list]:
        if self.provider == 'openai':
            return self.query_openai(prompt, **kwargs)
        elif self.provider == 'huggingface':
            return self.query_huggingface(prompt, **kwargs)
        elif self.provider == "vllm":
            return self.query_vllm(prompt, **kwargs)
        elif self.provider == "llama3":
            return self.query_llama3(prompt,"./Meta-Llama-3-8B-Instruct", "./Meta-Llama-3-8B-Instruct/tokenizer.model")
        else:
            raise ValueError("Unsupported provider")

    @timeout_decorator.timeout(60, timeout_exception=StopIteration)
    def query_with_timeout(self, messages, **kwargs):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )


    def query_openai(self, 
                     prompt: str, 
                     system: Optional[str] = None, 
                     rate_limit_per_minute: Optional[int] = None, **kwargs) -> Union[str, list]:
        # Set default system message
        if system is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        for i in range(64):
            try:
                response = self.query_with_timeout(messages, **kwargs)

                # Sleep to avoid rate limit if rate limit is set
                if rate_limit_per_minute:
                    time.sleep(60 / rate_limit_per_minute - 0.5)  # Buffer of 0.5 seconds

                if kwargs.get('n', 1) == 1:
                    return response.choices[0].message['content'], response
                else:
                    return [choice.message['content'] for choice in response.choices], response
                
            except StopIteration:
                print("Query timed out, retrying...")
                continue # Retry
            except Exception as e:
                print(e)
                time.sleep(10)

        raise RuntimeError("Failed to query the OpenAI API after 64 retries.")

    def query_huggingface(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, **kwargs)

        # Decode the generated text
        decoded_outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the start of the sequence
        prompt_length = len(self.tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        return decoded_outputs[prompt_length:], {"prompt": prompt, "prompt_length": len(inputs[0])}
    
    def query_vllm(self, prompt: str, **kwargs) -> str:
        from vllm import SamplingParams
        
        n = kwargs.get("n", 1)
    
        
        sampling_params = SamplingParams(
            max_tokens=256,
            temperature=kwargs.get("temperature", 0.8),
            stop=kwargs.get("stop", []),
            top_p=kwargs.get("top_p", 1.0) if kwargs.get("temperature", 0.8) != 0 else 1.0
        )
        
        prompts = [
            f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
        ]*n
        
        try:      
            outputs = self.model.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            outputs = [output.outputs[0].text for output in outputs]
        except ValueError as e:
            print(e)
            outputs = ["Sorry, I don't know the answer to that question."]
        
        if n == 1:
            return outputs[0], {"prompt": prompts[0]}
        else:
            return outputs, {"prompt": prompts[0]}

    def query_llama3(self,
    prompts,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    #max_seq_len: int = 128,
    #max_gen_len: int = 64,
    max_seq_len: int = 1280,
    max_gen_len: int = 640,
    max_batch_size: int = 4,) -> str:

        # Set default system message

        generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        prompts = [prompts]

        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        #for prompt, result in zip(prompts, results):
            #print(prompt)
            #print(f"> {result['generation']}")
         #   print("\n==================================\n")

        return [results[0]['generation']], results[0]

        """
        if system is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        for i in range(64):
            try:
                response = self.query_with_timeout(messages, **kwargs)

                # Sleep to avoid rate limit if rate limit is set
                if rate_limit_per_minute:
                    time.sleep(60 / rate_limit_per_minute - 0.5)  # Buffer of 0.5 seconds

                if kwargs.get('n', 1) == 1:
                    return response.choices[0].message['content'], response
                else:
                    return [choice.message['content'] for choice in response.choices], response

            except StopIteration:
                print("Query timed out, retrying...")
                continue  # Retry
            except Exception as e:
                print(e)
                time.sleep(10)

        raise RuntimeError("Failed to query the OpenAI API after 64 retries.")
        """