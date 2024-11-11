import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from codetf.models import load_model_pipeline
from codetf.data_utility.util import EOF_STRINGS, EndOfFunctionCriteria, remove_last_block
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteriaList
import torch
import os
from accelerate import Accelerator
import torch
from collections import defaultdict
from tqdm import tqdm
import torch
from evaluate import load
import numpy as np

class ModelEvaluator:
    def __init__(self, model_class, num_workers=5):
        self.model_class = model_class
        self.code_eval = load("code_eval")
        self.accelerator = Accelerator()

   
    def evaluate_pass_k(self, problems, unit_tests, batch_size=1, max_length=600, 
                        top_p=0.95, k=[1,10,100], temperature=1.2,
                        num_return_sequences=200, sequences_per_chunk=10, num_workers=1):
        # Load dataset
        data_loader = DataLoader(problems, batch_size=batch_size)
        data_loader = self.accelerator.prepare(data_loader)
        model_name = type(self.model_class).__name__
        # Initialize stopping criteria
        gen_kwargs = {
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "stopping_criteria": StoppingCriteriaList([EndOfFunctionCriteria(0, EOF_STRINGS, self.model_class.get_tokenizer())]),
        }
        
        # Store generated tokens
        gen_token_dict = defaultdict(list)
        solutions = []
        chunks = num_return_sequences // sequences_per_chunk
        # Generate and evaluate solutions
        
        dataloader_pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for step, batch in dataloader_pbar:
            prompt_ids, attention_masks = batch
            
            solutions_per_chunk = []
            for i in range(chunks):
                with torch.no_grad():
                    gen_kwargs["stopping_criteria"][0].start_length = attention_masks[0].sum().item()
                    
                    input_ids = prompt_ids[0, :attention_masks[0].sum().item()]
                  
                    input_data = self.model_class.get_tokenizer().decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    batch_generated_ids = self.model_class.get_model().generate(
                        input_ids=input_ids.unsqueeze(0),
                        attention_mask=attention_masks[0, :attention_masks[0].sum().item()].unsqueeze(0), 
                        max_length=max_length, num_return_sequences=sequences_per_chunk, 
                        **gen_kwargs
                    )
                    batch_generated_ids = batch_generated_ids.cpu().numpy()
                    
                    gen_codes = self.model_class.get_tokenizer().batch_decode(batch_generated_ids, 
                                            skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    
                    for i,item in enumerate(gen_codes):
                        result =  remove_last_block(item)
                        if model_name == "Seq2SeqModel":
                            result = f"{input_data} {result}"
                        
                        solutions_per_chunk.append(result)

            solutions.append(solutions_per_chunk)
            dataloader_pbar.set_description(f"Processing step {step+1}/{len(data_loader)}")
        
        pass_at_k, _ = self.code_eval.compute(
            references=unit_tests, predictions=solutions, k=k, num_workers=num_workers
        )

        return pass_at_k
