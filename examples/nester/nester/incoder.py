from typing import List

import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizers_version = tuple(int(n) for n in tokenizers.__version__.split('.'))
if tokenizers_version < (0, 12, 1):
    print("warning: Your tokenizers version looks old and you will likely have formatting issues. We recommend installing tokenizers >= 0.12.1")



# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"

def make_sentinel(i):
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"

def generate(input: str, model, tokenizer, max_to_generate: int=128, ):
    """
    Do standard left-to-right completion of the prefix `input` by sampling from the model
    """
    model = model.half().to(DEVICE)
    input_ids = tokenizer(input, return_tensors="pt", max_length = 1028, truncation=True).input_ids.to(DEVICE)
    max_length = max_to_generate + input_ids.flatten().size(0)
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, num_beams = 5, num_return_sequences = 5, max_length=max_length)
    # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
    predictions = tokenizer.batch_decode(outputs)
    predictions = [p[len(BOS):] if p.startswith(BOS) else p for p in predictions]
    return predictions

def infill(parts: List[str], model, tokenizer, max_to_generate: int=128, extra_sentinel: bool=True, max_retries: int=1):
    """
    Generate infills to complete a partial document, e.g.
    [A C E] -> [A B C D E], where B and D are infills that have been generated.
    parts: List[str]. list of parts of the document. One string will be
            inserted in between each element, i.e. infilling N-1 locations for a list
            of length N.
    max_to_generate: int. maximum number of tokens to generate. Keep in mind
            that the model context size is 2048.
    temperature: float. temperature parameter for sampling.
    extra_sentinel: bool. we recommend setting this to True, as it makes it
            easier for the model to end generated infills. See the footnote in 
            section 2.2 of our paper for details.
    max_retries: int. if > 1, use rejection sampling to keep sampling infills until
            all infills sample a completion token.
    returns a dictionary containing the following:
        text:  str, the completed document (with infills inserted)
        parts:  List[str], length N. Same as passed to the method
        infills:  List[str], length N-1. The list of infills generated
        retries_attempted:  number of retries used (if max_retries > 1)
    """
    assert isinstance(parts, list)
    retries_attempted = 0
    done = False

    while (not done) and (retries_attempted < max_retries):
        retries_attempted += 1
        
        ## (1) build the prompt
        if len(parts) == 1:
            prompt = parts[0]
        else:
            prompt = ""
            # encode parts separated by sentinel
            for sentinel_ix, part in enumerate(parts):
                prompt += part
                if extra_sentinel or (sentinel_ix < len(parts) - 1):
                    prompt += make_sentinel(sentinel_ix)
        
        infills = []

        done = True

        ## (2) generate infills
        for sentinel_ix, part in enumerate(parts[:-1]):
            prompt += make_sentinel(sentinel_ix)
            # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
            completions = generate(prompt, model, tokenizer, max_to_generate)
            completions = [completion[len(prompt):] + EOM if EOM not in completion[len(prompt):] else completion[len(prompt):] for completion in completions]
            completions = [completion[:completion.index(EOM) + len(EOM)] for completion in completions]
            infilled = [completion[:-len(EOM)] for completion in completions]
            infills.append(infilled)
            prompt += completions[0]
    
    return infills
